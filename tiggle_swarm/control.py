"""L3 기본 제어기: 학습 없이 동작하는 속도 명령 계산. MAPPO 보정값은 이 위에 더해진다.

텐서는 (B, N, ...) 형태다. B는 환경 수, N은 로봇 수이고, 실제 운용에서는 B=1로 부른다.
"""

from __future__ import annotations

import torch

from .config import ControlGains


def clamp_speed(u: torch.Tensor, v_max: float) -> torch.Tensor:
    """방향을 유지한 채 로봇별 속력을 v_max로 제한한다."""
    speed = u.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return u * (v_max / speed).clamp(max=1.0)


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def pairwise(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(p_i - p_j)와 그 크기. 형태는 (B, N, N, 2), (B, N, N)이고 대각 거리는 inf."""
    diff = p[:, :, None, :] - p[:, None, :, :]
    dist = diff.norm(dim=-1)
    n = p.shape[1]
    return diff, dist + torch.diag(torch.full((n,), float("inf"), device=p.device))


def nominal_velocity(
    p: torch.Tensor,
    p_ref: torch.Tensor,
    v_ref: torch.Tensor,
    group: torch.Tensor,
    gains: ControlGains,
    v_max: float,
) -> torch.Tensor:
    """기준 속도 + 위치 오차 P + 같은 무리 이웃과의 오차 합의항."""
    err = p_ref - p
    _, dist = pairwise(p)
    same = (group[:, :, None] == group[:, None, :]) & (group[:, :, None] >= 0)
    adj = same & (dist <= gains.comm_radius_m)
    count = adj.sum(dim=2, keepdim=True).clamp(min=1)
    relative = err[:, :, None, :] - err[:, None, :, :]
    consensus = (relative * adj[..., None]).sum(dim=2) / count
    return clamp_speed(v_ref + gains.k_pos * err + gains.k_consensus * consensus, v_max)


def heading_rate(
    theta: torch.Tensor,
    theta_ref: torch.Tensor,
    omega_ref: torch.Tensor,
    gains: ControlGains,
    omega_max: float,
) -> torch.Tensor:
    return (omega_ref + gains.k_heading * wrap_angle(theta_ref - theta)).clamp(-omega_max, omega_max)
