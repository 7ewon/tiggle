"""MAPPO로 L3 보정 정책을 학습하고, 기본 제어기와 같은 시나리오에서 비교한다.

  python -m tiggle_swarm.train --out checkpoints/residual.pt
  python -m tiggle_swarm.train --eval checkpoints/residual.pt

시나리오는 무작위 시작 배치에서 대형 1~2개로 옮겨 가는 전환이며, L2와 같은 슬롯과 배정, 시간표를 쓴다.
로봇 수를 2 -> 4 -> 8 -> 15대로 늘리는 커리큘럼으로 학습한다.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from tiggle_nl.config import FleetConfig

from .cbf import CBFConfig, safety_filter
from .config import ControlGains, RobotLimits
from .control import clamp_speed, heading_rate, nominal_velocity, pairwise
from .formations import rotate, slot_template
from .policy import GLOBAL_DIM, CentralCritic, Obs, ResidualActor, body_to_world, build_obs
from .sim import DomainRand, SwarmSim

SHAPES = ("line", "column", "circle", "v", "triangle", "square", "grid", "cluster")
MAX_PLAN_S = 20.0
SETTLE_S = 5.0


@dataclass
class TrainConfig:
    stages: tuple[tuple[int, int], ...] = ((2, 15), (4, 25), (8, 50), (15, 70))  # (로봇 수, 반복 수)
    batch: int = 48
    lr: float = 3e-4
    gamma: float = 0.99
    gae: float = 0.95
    clip: float = 0.2
    epochs: int = 3
    minibatch: int = 8192
    entropy: float = 1e-3
    value_coef: float = 0.5
    max_grad: float = 0.5
    residual_limit: float = 0.05  # 보정값 최대 크기 (m/s)
    seed: int = 0


@dataclass
class Scenario:
    start: torch.Tensor  # (B, N, 2)
    goal: torch.Tensor  # (B, N, 2)
    heading: torch.Tensor  # (B, N)
    duration: torch.Tensor  # (B, N)
    group: torch.Tensor  # (B, N)
    steps: int


def _scatter(rng: np.random.Generator, n: int, fleet: FleetConfig, separation: float) -> np.ndarray:
    """서로 separation 이상 떨어진 시작 위치. 일부러 좁은 영역에 몰아 혼잡한 경우를 만든다."""
    (x0, x1), (y0, y1) = fleet.arena_x, fleet.arena_y
    margin = fleet.robot_diameter_m
    shrink = rng.uniform(0.6, 1.0)
    for _ in range(50):
        w, h = (x1 - x0 - 2 * margin) * shrink, (y1 - y0 - 2 * margin) * shrink
        cx = rng.uniform(x0 + margin + w / 2, x1 - margin - w / 2)
        cy = rng.uniform(y0 + margin + h / 2, y1 - margin - h / 2)
        pts: list[np.ndarray] = []
        for _ in range(4000):
            c = np.array([rng.uniform(cx - w / 2, cx + w / 2), rng.uniform(cy - h / 2, cy + h / 2)])
            if all(np.hypot(*(c - q)) >= separation for q in pts):
                pts.append(c)
                if len(pts) == n:
                    return np.array(pts)
        shrink = min(1.0, shrink + 0.1)
    raise RuntimeError("시작 위치를 배치하지 못했습니다")


def _slots_for(rng, m: int, start: np.ndarray, fleet: FleetConfig) -> Optional[np.ndarray]:
    allowed = [s for s in SHAPES if s == "cluster" or fleet.min_robots[s] <= m]
    shape = allowed[int(rng.integers(len(allowed)))]
    spacing = rng.uniform(fleet.min_spacing_m, 0.32) if shape != "cluster" else fleet.min_spacing_m * 1.05
    tpl = rotate(slot_template(shape, m, spacing), rng.uniform(-math.pi, math.pi))
    (x0, x1), (y0, y1) = fleet.arena_x, fleet.arena_y
    margin = fleet.robot_diameter_m / 2
    low = np.array([x0 + margin, y0 + margin]) - tpl.min(axis=0)
    high = np.array([x1 - margin, y1 - margin]) - tpl.max(axis=0)
    if (low > high).any():
        return None
    return tpl + np.clip(start.mean(axis=0) + rng.uniform(-0.4, 0.4, 2), low, high)


def _sample_one(rng, n: int, fleet: FleetConfig, limits: RobotLimits):
    start = _scatter(rng, n, fleet, fleet.robot_diameter_m + limits.safety_margin_m + 0.02)
    for _ in range(100):
        order = rng.permutation(n)
        two = n >= 6 and rng.random() < 0.5
        cut = n // 2 if two else n
        groups = [order[:cut]] + ([order[cut:]] if two else [])
        goal, group, duration, placed, ok = np.zeros((n, 2)), np.zeros(n, dtype=int), np.zeros(n), [], True
        speed = limits.v_normal if rng.random() < 0.6 else limits.v_fast
        for g, idx in enumerate(groups):
            slots = _slots_for(rng, len(idx), start[idx], fleet)
            if slots is None or any(
                np.linalg.norm(slots[:, None] - other[None], axis=-1).min() < fleet.min_spacing_m for other in placed
            ):
                ok = False
                break
            placed.append(slots)
            _, col = linear_sum_assignment(((start[idx][:, None] - slots[None]) ** 2).sum(-1))
            goal[idx] = slots[col]
            group[idx] = g
            duration[idx] = max(1.5 * np.linalg.norm(goal[idx] - start[idx], axis=1).max() / speed, 1.0)
        if ok and duration.max() <= MAX_PLAN_S:
            return start, goal, rng.uniform(-math.pi, math.pi, n), duration, group
    raise RuntimeError("시나리오를 만들지 못했습니다")


def sample_scenario(rng, batch: int, n: int, fleet: FleetConfig, limits: RobotLimits, device: str = "cpu") -> Scenario:
    items = [_sample_one(rng, n, fleet, limits) for _ in range(batch)]

    def to(k: int, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.stack([it[k] for it in items]), dtype=dtype, device=device)

    duration = to(3)
    steps = int(math.ceil((float(duration.max()) + SETTLE_S) / limits.dt))
    return Scenario(to(0), to(1), to(2), duration, to(4, torch.long), steps)


def scenario_reference(s: Scenario, t: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = (t / s.duration).clamp(0, 1)
    beta = x * x * (3 - 2 * x)
    rate = 6 * x * (1 - x) / s.duration
    delta = s.goal - s.start
    return s.start + beta[..., None] * delta, rate[..., None] * delta


@dataclass
class Rollout:
    ego: list = field(default_factory=list)
    nbr: list = field(default_factory=list)
    mask: list = field(default_factory=list)
    state: list = field(default_factory=list)
    action: list = field(default_factory=list)
    logp: list = field(default_factory=list)
    value: list = field(default_factory=list)
    reward: list = field(default_factory=list)


@dataclass
class EpisodeStats:
    track_err: float
    final_err: float
    worst_final: float
    success: float
    contacts: float
    min_dist: float
    intervention: float
    reward: float


def run_episode(
    sim: SwarmSim,
    s: Scenario,
    actor: Optional[ResidualActor],
    cfg: TrainConfig,
    fleet: FleetConfig,
    limits: RobotLimits = RobotLimits(),
    gains: ControlGains = ControlGains(),
    cbf: CBFConfig = CBFConfig(),
    critic: Optional[CentralCritic] = None,
    deterministic: bool = False,
) -> tuple[EpisodeStats, Optional[Rollout]]:
    """한 에피소드를 돌린다. critic을 주면 학습용 표본을 함께 돌려준다."""
    B, N, dev = sim.B, sim.N, sim.device
    sim.reset(s.start, s.heading)
    safe_dist = fleet.robot_diameter_m + limits.safety_margin_m
    lam = None
    last_cmd = torch.zeros((B, N, 2), device=dev)
    last_action = torch.zeros((B, N, 2), device=dev)
    roll = Rollout() if critic is not None else None
    err_sum, contacts, intervention, reward_sum, ok_contacts = 0.0, 0.0, 0.0, 0.0, torch.zeros(B, device=dev)
    min_dist = float("inf")

    for k in range(s.steps):
        t = k * limits.dt
        p_ref, v_ref = scenario_reference(s, t)
        p_hat, theta_hat, v_hat = sim.observe()
        u_nom = nominal_velocity(p_hat, p_ref, v_ref, s.group, gains, limits.v_max)
        u_cmd, action = u_nom, last_action
        obs = None
        if actor is not None:
            obs = build_obs(p_hat, theta_hat, v_hat, p_ref, v_ref, u_nom, last_cmd, s.group, safe_dist, limits.v_max, gains.comm_radius_m)
            dist = actor.dist(obs)
            action = dist.mean if deterministic else dist.sample()
            residual = body_to_world(action.clamp(-1, 1) * cfg.residual_limit, theta_hat)
            u_cmd = clamp_speed(u_nom + residual, limits.v_max)
        safe = safety_filter(p_hat, u_cmd, safe_dist, limits.v_max, cbf, lam)
        lam = safe.lam
        omega = heading_rate(theta_hat, s.heading, torch.zeros_like(theta_hat), gains, limits.omega_max)

        if roll is not None:
            state = _global_state(sim.pos, p_ref, safe.intervention, safe_dist, t, s, limits, N)
            roll.ego.append(obs.ego)
            roll.nbr.append(obs.nbr)
            roll.mask.append(obs.mask)
            roll.state.append(state)
            roll.action.append(action)
            roll.logp.append(dist.log_prob(action).sum(-1))
            roll.value.append(critic(obs, state))

        sim.step(safe.u, omega)
        last_cmd = safe.u

        err = (p_ref - sim.pos).norm(dim=-1)
        own = (
            -(err / 0.1).clamp(max=3.0)
            - 0.5 * safe.intervention / limits.v_max
            - 2.0 * sim.contacts
            - 0.25 * (action - last_action).pow(2).sum(-1)
            - 0.02 * action.pow(2).sum(-1)
        )
        reward = 0.1 * (0.7 * own + 0.3 * own.mean(dim=1, keepdim=True))
        last_action = action

        if roll is not None:
            roll.reward.append(reward)
        err_sum += float(err.mean())
        contacts += float(sim.contacts.sum()) / 2
        intervention += float(safe.intervention.mean())
        reward_sum += float(reward.mean())
        ok_contacts += sim.contacts.sum(dim=1)
        if N > 1:
            min_dist = min(min_dist, float(pairwise(sim.pos)[1].min()))

    final = (s.goal - sim.pos).norm(dim=-1)
    success = (final.max(dim=1).values <= limits.pos_tolerance_m) & (ok_contacts == 0)
    stats = EpisodeStats(
        track_err=err_sum / s.steps,
        final_err=float(final.mean()),
        worst_final=float(final.max(dim=1).values.mean()),
        success=float(success.float().mean()),
        contacts=contacts / B,
        min_dist=min_dist,
        intervention=intervention / s.steps,
        reward=reward_sum,
    )
    return stats, roll


def _global_state(pos, p_ref, intervention, safe_dist, t, s: Scenario, limits, N) -> torch.Tensor:
    err = (p_ref - pos).norm(dim=-1)
    B = pos.shape[0]
    gap = ((pairwise(pos)[1].reshape(B, -1).min(dim=1).values - safe_dist) / 0.2).clamp(-1, 3)
    env = torch.stack(
        [err.mean(dim=1) / 0.2, err.max(dim=1).values / 0.2, gap, intervention.mean(dim=1) / limits.v_max, torch.full((B,), N / 15, device=pos.device)],
        dim=-1,
    )
    togo = ((s.duration - t) / MAX_PLAN_S).clamp(-1, 1)[..., None]
    return torch.cat([env[:, None].expand(-1, N, -1), togo], dim=-1)


def ppo_update(actor, critic, optimizer, roll: Rollout, last_value: torch.Tensor, cfg: TrainConfig) -> dict[str, float]:
    reward = torch.stack(roll.reward)
    value = torch.stack(roll.value).detach()
    adv = torch.zeros_like(reward)
    running = torch.zeros_like(last_value)
    for k in reversed(range(len(roll.reward))):
        next_value = last_value if k == len(roll.reward) - 1 else value[k + 1]
        delta = reward[k] + cfg.gamma * next_value - value[k]
        running = delta + cfg.gamma * cfg.gae * running
        adv[k] = running
    ret = adv + value

    def flat(items: list) -> torch.Tensor:
        x = torch.stack(items).detach()
        return x.reshape(-1, *x.shape[3:])

    ego, nbr, mask, state, action, old_logp = map(flat, (roll.ego, roll.nbr, roll.mask, roll.state, roll.action, roll.logp))
    adv_flat, ret_flat = adv.reshape(-1), ret.reshape(-1)
    adv_flat = (adv_flat - adv_flat.mean()) / adv_flat.std().clamp(min=1e-6)

    size = ego.shape[0]
    log = {"policy": 0.0, "value": 0.0, "n": 0}
    for _ in range(cfg.epochs):
        perm = torch.randperm(size)
        for start in range(0, size - cfg.minibatch + 1, cfg.minibatch):
            mb = perm[start : start + cfg.minibatch]
            obs = Obs(ego[mb], nbr[mb], mask[mb])
            dist = actor.dist(obs)
            logp = dist.log_prob(action[mb]).sum(-1)
            ratio = (logp - old_logp[mb]).exp()
            surr = torch.min(ratio * adv_flat[mb], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv_flat[mb])
            policy_loss = -surr.mean() - cfg.entropy * dist.entropy().sum(-1).mean()
            value_loss = (critic(obs, state[mb]) - ret_flat[mb]).pow(2).mean()
            optimizer.zero_grad()
            (policy_loss + cfg.value_coef * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), cfg.max_grad)
            optimizer.step()
            log["policy"] += policy_loss.item()
            log["value"] += value_loss.item()
            log["n"] += 1
    n = max(log.pop("n"), 1)
    return {k: v / n for k, v in log.items()}


def train(cfg: TrainConfig, out: Path, fleet_factory=FleetConfig, device: str = "cpu", log_every: int = 5) -> ResidualActor:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    limits, gains = RobotLimits(), ControlGains()
    actor, critic = ResidualActor().to(device), CentralCritic().to(device)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=cfg.lr)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".csv")
    started = time.time()
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "n", "reward", "track_err", "success", "contacts", "intervention", "std", "seconds"])
        iteration = 0
        for n, count in cfg.stages:
            fleet = fleet_factory(robot_ids=tuple(range(1, n + 1)))
            sim = SwarmSim(cfg.batch, n, fleet, limits, DomainRand(), device)
            for _ in range(count):
                scen = sample_scenario(rng, cfg.batch, n, fleet, limits, device)
                with torch.no_grad():
                    stats, roll = run_episode(sim, scen, actor, cfg, fleet, limits, gains, critic=critic)
                    p_ref, _ = scenario_reference(scen, scen.steps * limits.dt)
                    p_hat, theta_hat, v_hat = sim.observe()
                    u_nom = nominal_velocity(p_hat, p_ref, torch.zeros_like(p_ref), scen.group, gains, limits.v_max)
                    safe_dist = fleet.robot_diameter_m + limits.safety_margin_m
                    obs = build_obs(p_hat, theta_hat, v_hat, p_ref, torch.zeros_like(p_ref), u_nom, u_nom, scen.group, safe_dist, limits.v_max, gains.comm_radius_m)
                    state = _global_state(sim.pos, p_ref, torch.zeros((cfg.batch, n), device=device), safe_dist, scen.steps * limits.dt, scen, limits, n)
                    last_value = critic(obs, state)
                ppo_update(actor, critic, optimizer, roll, last_value, cfg)
                iteration += 1
                std = float(actor.log_std.exp().mean())
                elapsed = time.time() - started
                writer.writerow([iteration, n, f"{stats.reward:.3f}", f"{stats.track_err:.4f}", f"{stats.success:.3f}", f"{stats.contacts:.3f}", f"{stats.intervention:.4f}", f"{std:.3f}", f"{elapsed:.0f}"])
                f.flush()
                if iteration % log_every == 0 or iteration == 1:
                    print(f"[{iteration:4d}] N={n:2d} 보상 {stats.reward:8.2f}  추종오차 {stats.track_err * 100:5.2f}cm  성공 {stats.success:4.2f}  접촉 {stats.contacts:5.2f}  std {std:.2f}  {elapsed:5.0f}s", flush=True)
                if iteration % 20 == 0:
                    save_actor(actor, cfg, out)
    save_actor(actor, cfg, out)
    return actor


def save_actor(actor: ResidualActor, cfg: TrainConfig, path: Path) -> None:
    torch.save({"state_dict": actor.state_dict(), "residual_limit": cfg.residual_limit}, path)


def load_actor(path: Path, device: str = "cpu") -> tuple[ResidualActor, float]:
    data = torch.load(path, map_location=device, weights_only=True)
    actor = ResidualActor().to(device)
    actor.load_state_dict(data["state_dict"])
    actor.eval()
    return actor, float(data["residual_limit"])


def evaluate(actor: Optional[ResidualActor], residual_limit: float, sizes, batch: int = 48, episodes: int = 4, seed: int = 1234, device: str = "cpu") -> dict[int, EpisodeStats]:
    """같은 시드면 기본 제어기와 학습 정책이 같은 시나리오·같은 로봇 파라미터로 평가된다."""
    limits, gains = RobotLimits(), ControlGains()
    cfg = TrainConfig(residual_limit=residual_limit)
    results: dict[int, EpisodeStats] = {}
    for n in sizes:
        rng = np.random.default_rng(seed + n)
        fleet = FleetConfig(robot_ids=tuple(range(1, n + 1)))
        gen = torch.Generator(device=device).manual_seed(seed + n)
        sim = SwarmSim(batch, n, fleet, limits, DomainRand(), device, gen)
        collected: list[EpisodeStats] = []
        with torch.no_grad():
            for _ in range(episodes):
                scen = sample_scenario(rng, batch, n, fleet, limits, device)
                stats, _ = run_episode(sim, scen, actor, cfg, fleet, limits, gains, deterministic=True)
                collected.append(stats)
        results[n] = EpisodeStats(
            *(float(np.mean([getattr(c, f) for c in collected])) for f in ("track_err", "final_err", "worst_final", "success", "contacts")),
            min_dist=min(c.min_dist for c in collected),
            intervention=float(np.mean([c.intervention for c in collected])),
            reward=float(np.mean([c.reward for c in collected])),
        )
    return results


def print_comparison(actor: Optional[ResidualActor], residual_limit: float, sizes=(2, 3, 4, 6, 8, 12, 15), **kwargs) -> None:
    base = evaluate(None, residual_limit, sizes, **kwargs)
    learned = evaluate(actor, residual_limit, sizes, **kwargs) if actor is not None else None
    print(f"{'N':>3} | {'방식':<10} | {'추종(cm)':>8} | {'최종평균(cm)':>11} | {'최종최악(cm)':>11} | {'성공률':>6} | {'접촉/판':>7} | {'최소거리(cm)':>11} | {'CBF개입':>7}")
    for n in sizes:
        rows = [("기본+CBF", base[n])] + ([("MAPPO+CBF", learned[n])] if learned else [])
        for name, r in rows:
            print(f"{n:>3} | {name:<10} | {r.track_err * 100:8.2f} | {r.final_err * 100:11.2f} | {r.worst_final * 100:11.2f} | {r.success:6.2f} | {r.contacts:7.2f} | {r.min_dist * 100:11.2f} | {r.intervention:7.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="MAPPO 보정 정책 학습")
    ap.add_argument("--out", type=Path, default=Path("checkpoints/residual.pt"))
    ap.add_argument("--eval", type=Path, default=None, help="학습 없이 이 체크포인트를 평가한다")
    ap.add_argument("--baseline", action="store_true", help="기본 제어기만 평가한다")
    ap.add_argument("--scale", type=float, default=1.0, help="커리큘럼 반복 수에 곱하는 값")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.baseline:
        print_comparison(None, TrainConfig().residual_limit, device=args.device)
        return
    if args.eval:
        actor, limit = load_actor(args.eval, args.device)
        print_comparison(actor, limit, device=args.device)
        return
    cfg = TrainConfig(seed=args.seed)
    cfg.stages = tuple((n, max(1, int(c * args.scale))) for n, c in cfg.stages)
    actor = train(cfg, args.out, device=args.device)
    actor.eval()
    print_comparison(actor, cfg.residual_limit, device=args.device)


if __name__ == "__main__":
    main()
