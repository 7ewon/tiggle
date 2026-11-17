"""L4 안전 필터: CBF-QP (Wang, Ames, Egerstedt, TRO 2017).

평면 이동 로봇을 속도 입력 모델로 보고, 로봇 쌍마다 h = d^2 - D^2 >= 0 을 유지한다.

    min sum ||u_i - u_nom_i||^2   s.t.  n_ij . (u_i - u_j) >= -gamma h_ij / (2 d_ij)

쌍대 문제를 FISTA로 풀어 환경 수만큼 한 번에 처리한다. 학습과 실제 운용이 같은 코드를 쓴다.
속력 제한은 풀이 후 전체를 같은 비율로 줄여 맞춘다. h >= 0인 쌍의 제약은 이 축소로 깨지지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from .control import pairwise


@dataclass(frozen=True)
class CBFConfig:
    gamma: float = 1.2  # 경계로 다가가는 속도를 줄이는 비율 (1/s). 구동 지연의 역수보다 충분히 작게
    iters: int = 40
    yield_gain: float = 0.4  # 정면으로 마주칠 때 오른쪽으로 비키는 정도
    yield_range_m: float = 0.08


@dataclass
class SafetyResult:
    u: torch.Tensor  # (B, N, 2) 안전한 속도 명령
    lam: torch.Tensor  # (B, N, N) 다음 주기 warm start용
    intervention: torch.Tensor  # (B, N) 입력을 얼마나 바꿨는지 (m/s)


def safety_filter(
    p: torch.Tensor,
    u_nom: torch.Tensor,
    min_dist: Union[float, torch.Tensor],
    v_max: float,
    cfg: CBFConfig = CBFConfig(),
    lam: Optional[torch.Tensor] = None,
) -> SafetyResult:
    """min_dist는 스칼라 또는 (B, N, N). 값이 0인 쌍(도킹)에는 제약을 걸지 않는다."""
    B, N = p.shape[:2]
    diff, dist = pairwise(p)
    safe_dist = dist.clamp(min=1e-6)
    n = diff / safe_dist[..., None]
    D = min_dist if torch.is_tensor(min_dist) else torch.full((B, N, N), float(min_dist), device=p.device)

    # 이 거리 밖의 쌍은 둘 다 최고 속도로 다가와도 제약에 걸리지 않으므로 뺀다
    a = 2 * v_max / cfg.gamma
    active = (dist < a + torch.sqrt(a * a + D * D)) & (D > 0)

    # 정면 교착 방지: 상대를 향해 가는 로봇은 자기 오른쪽으로 조금 비킨다 (consistent perturbation)
    toward = -n
    right = torch.stack([toward[..., 1], -toward[..., 0]], dim=-1)
    approaching = (u_nom[:, :, None, :] * toward).sum(-1) > 0
    near = active & approaching & (dist < D + cfg.yield_range_m)
    speed = u_nom.norm(dim=-1, keepdim=True)
    u_biased = u_nom + cfg.yield_gain * speed * (right * near[..., None]).sum(dim=2)

    # 이미 침범한 쌍을 밀어내는 속도는 구동 한계 안으로 제한한다
    limit = (-cfg.gamma * (dist * dist - D * D) / (2 * safe_dist)).clamp(max=v_max)
    b = torch.where(active, limit, torch.zeros_like(dist))  # 대각은 거리가 inf라 여기서 걸러진다

    degree = active.sum(dim=2).max(dim=1).values.clamp(min=1)
    step = (0.5 / degree)[:, None, None]
    current = torch.zeros((B, N, N), device=p.device) if lam is None else lam * active
    z, t = current, 1.0
    for _ in range(cfg.iters):
        u = u_biased + (z[..., None] * n).sum(dim=2)
        gap = ((u[:, :, None, :] - u[:, None, :, :]) * n).sum(-1) - b
        following = (z - step * gap).clamp(min=0) * active
        t_next = (1 + (1 + 4 * t * t) ** 0.5) / 2
        z = following + (t - 1) / t_next * (following - current)
        current, t = following, t_next

    u = u_biased + (current[..., None] * n).sum(dim=2)
    top = u.norm(dim=-1).max(dim=1).values.clamp(min=1e-9)
    u = u * (v_max / top).clamp(max=1.0)[:, None, None]
    return SafetyResult(u, current, (u - u_nom).norm(dim=-1))
