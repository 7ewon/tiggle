"""메카넘 로봇 무리의 추상 모델 (PyTorch 벡터화).

롤러 물리 대신 '속도 명령 -> 지연 -> 데드밴드/축별 이득 -> 1차 지연'으로 구동을 근사한다.
파라미터는 DomainRand 범위에서 뽑는다. Real Track 실측값이 나오면 범위를 그 주변으로 좁힌다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from tiggle_nl.config import FleetConfig

from .config import RobotLimits
from .control import pairwise


@dataclass(frozen=True)
class DomainRand:
    """TODO 전부 가정값. 측정 항목은 PWM-속도 곡선, 스텝 응답, 횡이동 효율, 통신 지연, 측위 노이즈."""

    gain_forward: tuple[float, float] = (0.8, 1.1)
    gain_strafe: tuple[float, float] = (0.6, 0.95)
    tau: tuple[float, float] = (0.06, 0.20)
    deadband: tuple[float, float] = (0.0, 0.015)
    cmd_delay: tuple[int, int] = (0, 3)  # 주기 수
    obs_delay: tuple[int, int] = (1, 3)
    pos_noise: tuple[float, float] = (0.0, 0.003)
    drop: tuple[float, float] = (0.0, 0.1)
    hw_v_max: float = 0.15


IDEAL = DomainRand(
    gain_forward=(1.0, 1.0), gain_strafe=(1.0, 1.0), tau=(0.1, 0.1), deadband=(0.0, 0.0),
    cmd_delay=(0, 0), obs_delay=(0, 0), pos_noise=(0.0, 0.0), drop=(0.0, 0.0),
)

_HISTORY = 8  # 지연 버퍼 길이 (주기 수)
_VEL_FILTER = 0.2  # 속도 추정 저역통과 계수


class SwarmSim:
    def __init__(
        self,
        batch: int,
        n: int,
        fleet: FleetConfig,
        limits: RobotLimits = RobotLimits(),
        rand: DomainRand = DomainRand(),
        device: str = "cpu",
        generator: Optional[torch.Generator] = None,
    ):
        self.B, self.N, self.fleet, self.limits, self.rand = batch, n, fleet, limits, rand
        self.device = device
        self.gen = generator or torch.Generator(device=device)
        self.env = torch.arange(batch, device=device)

    def _uniform(self, span: tuple[float, float], shape: tuple[int, ...]) -> torch.Tensor:
        return span[0] + (span[1] - span[0]) * torch.rand(shape, device=self.device, generator=self.gen)

    def _steps(self, span: tuple[int, int]) -> torch.Tensor:
        return torch.randint(span[0], span[1] + 1, (self.B,), device=self.device, generator=self.gen)

    def reset(self, pos: torch.Tensor, heading: torch.Tensor) -> None:
        B, N, r = self.B, self.N, self.rand
        self.pos, self.heading = pos.clone(), heading.clone()
        self.vel = torch.zeros((B, N, 2), device=self.device)
        self.omega = torch.zeros((B, N), device=self.device)
        self.gain_f = self._uniform(r.gain_forward, (B, N))
        self.gain_s = self._uniform(r.gain_strafe, (B, N))
        self.tau = self._uniform(r.tau, (B, N))
        self.deadband = self._uniform(r.deadband, (B, N))
        self.cmd_delay, self.obs_delay = self._steps(r.cmd_delay), self._steps(r.obs_delay)
        self.noise = self._uniform(r.pos_noise, (B, 1, 1))
        self.drop = self._uniform(r.drop, (B, 1))

        self.t = 0
        self.cmd_hist = torch.zeros((B, _HISTORY, N, 3), device=self.device)
        state = torch.cat([self.pos, self.heading[..., None]], dim=-1)
        self.obs_hist = state[:, None].repeat(1, _HISTORY, 1, 1)
        self.applied = torch.zeros((B, N, 3), device=self.device)
        self.prev_obs: Optional[torch.Tensor] = None
        self.vel_est = torch.zeros((B, N, 2), device=self.device)
        self.exempt = torch.zeros((B, N, N), dtype=torch.bool, device=self.device)
        self.contacts = torch.zeros((B, N), device=self.device)

    def observe(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """천장 카메라 측위를 흉내 낸 (위치, 방향, 속도 추정). 지연과 노이즈가 들어간다."""
        delayed = self.obs_hist[self.env, (self.t - self.obs_delay) % _HISTORY]
        noise = torch.randn((self.B, self.N, 2), device=self.device, generator=self.gen) * self.noise
        pos = delayed[..., :2] + noise
        if self.prev_obs is not None:
            raw = (pos - self.prev_obs) / self.limits.dt
            self.vel_est = (1 - _VEL_FILTER) * self.vel_est + _VEL_FILTER * raw
        self.prev_obs = pos
        return pos, delayed[..., 2], self.vel_est

    def step(self, u_world: torch.Tensor, omega_cmd: torch.Tensor) -> None:
        dt, r = self.limits.dt, self.rand
        self.cmd_hist[:, self.t % _HISTORY] = torch.cat([u_world, omega_cmd[..., None]], dim=-1)
        delayed = self.cmd_hist[self.env, (self.t - self.cmd_delay) % _HISTORY]
        lost = torch.rand((self.B, self.N), device=self.device, generator=self.gen) < self.drop
        self.applied = torch.where(lost[..., None], self.applied, delayed)
        ux, uy, w = self.applied[..., 0], self.applied[..., 1], self.applied[..., 2]

        c, s = torch.cos(self.heading), torch.sin(self.heading)
        bx, by = c * ux + s * uy, -s * ux + c * uy
        moving = (bx * bx + by * by).sqrt() >= self.deadband
        bx, by = bx * moving * self.gain_f, by * moving * self.gain_s
        target = torch.stack([c * bx - s * by, s * bx + c * by], dim=-1)
        speed = target.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        target = target * (r.hw_v_max / speed).clamp(max=1.0)

        alpha = 1 - torch.exp(-dt / self.tau)
        self.vel = self.vel + alpha[..., None] * (target - self.vel)
        self.omega = self.omega + alpha * (w - self.omega)
        self.pos = self.pos + self.vel * dt
        self.heading = self.heading + self.omega * dt
        self._resolve_contacts()

        self.t += 1
        self.obs_hist[:, self.t % _HISTORY] = torch.cat([self.pos, self.heading[..., None]], dim=-1)

    def _resolve_contacts(self) -> None:
        diff, dist = pairwise(self.pos)
        d0 = self.fleet.robot_diameter_m
        touching = (dist < d0) & ~self.exempt
        overlap = ((d0 - dist).clamp(min=0) * touching)[..., None]
        n = diff / dist.clamp(min=1e-6)[..., None]
        self.pos = self.pos + (0.5 * overlap * n).sum(dim=2)
        along = (self.vel[:, :, None, :] * n).sum(-1)
        self.vel = self.vel - ((along * ((along < 0) & touching))[..., None] * n).sum(dim=2)
        self.contacts = touching.sum(dim=2).float()

        (x0, x1), (y0, y1) = self.fleet.arena_x, self.fleet.arena_y
        m = d0 / 2
        low = torch.tensor([x0 + m, y0 + m], device=self.device)
        high = torch.tensor([x1 - m, y1 - m], device=self.device)
        clamped = self.pos.clamp(min=low, max=high)
        self.vel = self.vel * (clamped == self.pos)
        self.pos = clamped
