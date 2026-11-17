"""L2-L3-L4를 한 주기로 묶은 제어기. 측정값을 받아 로봇별 속도 명령을 낸다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from tiggle_nl.config import FleetConfig
from tiggle_nl.schema import TaskSpec

from .cbf import CBFConfig, safety_filter
from .config import ControlGains, RobotLimits
from .control import clamp_speed, heading_rate, nominal_velocity, pairwise
from .planner import PlanExecutor, PlanError, Reference, SwarmState
from .policy import ResidualActor, body_to_world, build_obs
from .sim import DomainRand, SwarmSim


@dataclass
class ControlOutput:
    velocity: np.ndarray  # (N, 2) 경기장 좌표 속도 명령 (m/s). 몸체 좌표 변환은 L5
    omega: np.ndarray  # (N,) rad/s
    reference: Reference
    intervention: np.ndarray  # (N,) 안전 필터가 바꾼 크기 (m/s)


def _batched(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)[None]


class SwarmController:
    def __init__(
        self,
        spec: TaskSpec,
        state: SwarmState,
        fleet: FleetConfig,
        limits: RobotLimits = RobotLimits(),
        gains: ControlGains = ControlGains(),
        cbf: CBFConfig = CBFConfig(iters=100),
        actor: Optional[ResidualActor] = None,
        residual_limit: float = 0.05,
    ):
        self.executor = PlanExecutor(spec, state, fleet, limits)
        self.fleet, self.limits, self.gains, self.cbf = fleet, limits, gains, cbf
        self.actor, self.residual_limit = actor, residual_limit
        self.lam: Optional[torch.Tensor] = None
        self.last_cmd = torch.zeros((1, len(fleet.robot_ids), 2))

    @torch.no_grad()
    def step(self, t: float, pos: np.ndarray, heading: np.ndarray, vel_est: np.ndarray) -> ControlOutput:
        ref = self.executor.update(t, SwarmState(pos, heading))
        fleet, limits, gains = self.fleet, self.limits, self.gains
        p, theta, v_hat = _batched(pos), _batched(heading), _batched(vel_est)
        p_ref, v_ref = _batched(ref.pos), _batched(ref.vel)
        group = torch.as_tensor(ref.group, dtype=torch.long)[None]
        safe_dist = fleet.robot_diameter_m + limits.safety_margin_m

        u = nominal_velocity(p, p_ref, v_ref, group, gains, limits.v_max)
        if self.actor is not None:
            obs = build_obs(p, theta, v_hat, p_ref, v_ref, u, self.last_cmd, group, safe_dist, limits.v_max, gains.comm_radius_m)
            action = self.actor.mean(obs) * self.residual_limit
            u = clamp_speed(u + body_to_world(action, theta), limits.v_max)

        min_dist = safe_dist * (1.0 - torch.as_tensor(ref.docked, dtype=torch.float32))[None]
        safe = safety_filter(p, u, min_dist, limits.v_max, self.cbf, self.lam)
        self.lam, self.last_cmd = safe.lam, safe.u
        omega = heading_rate(theta, _batched(ref.heading), _batched(ref.omega), gains, limits.omega_max)
        return ControlOutput(safe.u[0].numpy(), omega[0].numpy(), ref, safe.intervention[0].numpy())


@dataclass
class RunResult:
    ok: bool
    error: Optional[str]
    seconds: float
    final_error: float  # 마지막 기준 대비 평균 위치 오차 (m)
    worst_error: float
    min_distance: float  # 로봇 중심 간 최소 거리 (m)
    contacts: int
    intervention: float
    warnings: list[str]
    trace: np.ndarray  # (T, N, 2) 실제 위치
    final_reference: np.ndarray  # (N, 2)


def random_start(n: int, fleet: FleetConfig, seed: int, limits: RobotLimits = RobotLimits()) -> SwarmState:
    from .train import _scatter

    rng = np.random.default_rng(seed)
    pos = _scatter(rng, n, fleet, fleet.robot_diameter_m + limits.safety_margin_m + 0.02)
    return SwarmState(pos, rng.uniform(-np.pi, np.pi, n))


def run_in_sim(
    spec: TaskSpec,
    fleet: FleetConfig,
    start: SwarmState,
    limits: RobotLimits = RobotLimits(),
    rand: DomainRand = DomainRand(),
    actor: Optional[ResidualActor] = None,
    residual_limit: float = 0.05,
    seed: int = 0,
    max_seconds: float = 120.0,
    extra_settle_s: float = 2.0,
) -> RunResult:
    """TaskSpec 전체를 추상 시뮬레이터에서 실행한다. 제어기는 측정값(지연·노이즈 포함)만 본다."""
    n = len(fleet.robot_ids)
    gen = torch.Generator().manual_seed(seed)
    sim = SwarmSim(1, n, fleet, limits, rand, generator=gen)
    sim.reset(_batched(start.pos), _batched(start.heading))
    empty = np.zeros((0, n, 2))

    def result(ok, error, secs, trace, ref, contacts, interv, mind, warnings) -> RunResult:
        final = np.linalg.norm(trace[-1] - ref, axis=1) if len(trace) else np.zeros(n)
        return RunResult(ok, error, secs, float(final.mean()), float(final.max()), mind, contacts, interv, warnings, trace, ref)

    try:
        ctrl = SwarmController(spec, start, fleet, limits, actor=actor, residual_limit=residual_limit)
    except PlanError as e:
        return result(False, str(e), 0.0, empty, start.pos, 0, 0.0, float("inf"), [])

    trace, contacts, interv, mind = [], 0, 0.0, float("inf")
    done_at: Optional[float] = None
    k, ref = 0, start.pos
    while True:
        t = k * limits.dt
        p_hat, theta_hat, v_hat = sim.observe()
        try:
            out = ctrl.step(t, p_hat[0].numpy(), theta_hat[0].numpy(), v_hat[0].numpy())
        except PlanError as e:
            return result(False, str(e), t, np.array(trace) if trace else empty, ref, contacts, interv / max(k, 1), mind, ctrl.executor.warnings)
        ref = out.reference.pos
        sim.exempt = torch.as_tensor(out.reference.docked, dtype=torch.bool)[None]
        sim.step(_batched(out.velocity), _batched(out.omega))
        trace.append(sim.pos[0].numpy().copy())
        contacts += int(sim.contacts.sum()) // 2
        interv += float(out.intervention.mean())
        if n > 1:
            d = pairwise(sim.pos)[1]
            d = d.masked_fill(sim.exempt, float("inf"))
            mind = min(mind, float(d.min()))
        k += 1
        t = k * limits.dt
        if out.reference.done and done_at is None:
            done_at = t
        if (done_at is not None and t >= done_at + extra_settle_s) or t >= max_seconds:
            break

    timed_out = done_at is None
    error = None if not timed_out else f"{max_seconds:.0f}초 안에 끝나지 않았습니다"
    return result(not timed_out and contacts == 0, error, t, np.array(trace), ref, contacts, interv / k, mind, ctrl.executor.warnings)
