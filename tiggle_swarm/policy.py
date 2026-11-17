"""L3 학습 정책: 가까운 이웃을 attention으로 묶는 MAPPO 보정값 신경망.

입력은 각 로봇 주변 정보뿐이고(분산 실행), 이웃 수를 입력 크기에 넣지 않아
2대로 학습한 정책을 15대에 그대로 쓸 수 있다. 학습할 때만 critic이 전역 정보를 본다(CTDE).
벡터는 전부 로봇 몸체 좌표계로 돌려서 넣는다. 메카넘은 앞뒤와 좌우의 구동 특성이 다르기 때문이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .control import pairwise

K_NEIGHBORS = 6
EGO_DIM = 10
NBR_DIM = 8
GLOBAL_DIM = 6
_ERR_SCALE = 0.2  # m
_POS_SCALE = 0.5  # m


@dataclass
class Obs:
    ego: torch.Tensor  # (B, N, EGO_DIM)
    nbr: torch.Tensor  # (B, N, K, NBR_DIM)
    mask: torch.Tensor  # (B, N, K) 실제 이웃이면 True


def _to_body(x: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return torch.stack([c * x[..., 0] + s * x[..., 1], -s * x[..., 0] + c * x[..., 1]], dim=-1)


def body_to_world(a: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([c * a[..., 0] - s * a[..., 1], s * a[..., 0] + c * a[..., 1]], dim=-1)


def build_obs(
    p: torch.Tensor,
    theta: torch.Tensor,
    v_hat: torch.Tensor,
    p_ref: torch.Tensor,
    v_ref: torch.Tensor,
    u_nom: torch.Tensor,
    last_cmd: torch.Tensor,
    group: torch.Tensor,
    safe_dist: float,
    v_max: float,
    comm_radius: float,
) -> Obs:
    B, N = p.shape[:2]
    c, s = torch.cos(theta), torch.sin(theta)
    err = p_ref - p
    ego = torch.cat(
        [
            (_to_body(err, c, s) / _ERR_SCALE).clamp(-3, 3),
            _to_body(v_ref, c, s) / v_max,
            _to_body(v_hat, c, s) / v_max,
            _to_body(u_nom, c, s) / v_max,
            _to_body(last_cmd, c, s) / v_max,
        ],
        dim=-1,
    )

    k = min(K_NEIGHBORS, N - 1)
    _, dist = pairwise(p)
    dist_k, idx = torch.topk(dist, k, dim=2, largest=False)
    pick = idx[..., None].expand(-1, -1, -1, 2)

    def gather(x: torch.Tensor) -> torch.Tensor:
        return torch.take_along_dim(x[:, None].expand(-1, N, -1, -1), pick, dim=2)

    ck, sk = c[..., None], s[..., None]
    rel_pos = gather(p) - p[:, :, None]
    rel_vel = gather(v_hat) - v_hat[:, :, None]
    other = torch.take_along_dim(group, idx.reshape(B, -1), dim=1).reshape(B, N, k)
    same = (other == group[:, :, None]) & (group[:, :, None] >= 0)
    nbr = torch.cat(
        [
            _to_body(rel_pos, ck, sk) / _POS_SCALE,
            _to_body(rel_vel, ck, sk) / v_max,
            (_to_body(gather(err), ck, sk) / _ERR_SCALE).clamp(-3, 3),
            ((dist_k - safe_dist) / _ERR_SCALE).clamp(-1, 3)[..., None],
            same[..., None].float(),
        ],
        dim=-1,
    )
    mask = dist_k <= comm_radius
    if k < K_NEIGHBORS:  # 로봇이 적으면 빈 칸을 채워 입력 크기를 맞춘다
        pad = K_NEIGHBORS - k
        nbr = torch.cat([nbr, torch.zeros((B, N, pad, NBR_DIM), device=p.device)], dim=2)
        mask = torch.cat([mask, torch.zeros((B, N, pad), dtype=torch.bool, device=p.device)], dim=2)
    return Obs(ego, nbr * mask[..., None], mask)


class NeighborEncoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.ego = nn.Sequential(nn.Linear(EGO_DIM, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU())
        self.nbr = nn.Sequential(nn.Linear(NBR_DIM, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU())
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.scale = math.sqrt(hidden)

    def forward(self, ego: torch.Tensor, nbr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h_ego = self.ego(ego)
        h_nbr = self.nbr(nbr)
        score = (self.query(h_ego)[..., None, :] * self.key(h_nbr)).sum(-1) / self.scale
        weight = torch.softmax(score.masked_fill(~mask, -1e9), dim=-1)
        context = (weight[..., None] * h_nbr).sum(-2) * mask.any(-1, keepdim=True)
        return torch.cat([h_ego, context], dim=-1)


class ResidualActor(nn.Module):
    """출력은 [-1, 1] 범위의 몸체 좌표 속도 보정값. 초기에는 거의 0이라 기본 제어기와 같게 시작한다."""

    def __init__(self, hidden: int = 96):
        super().__init__()
        self.encoder = NeighborEncoder(hidden)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ELU(), nn.Linear(hidden, 2))
        nn.init.orthogonal_(self.head[-1].weight, gain=0.01)
        nn.init.zeros_(self.head[-1].bias)
        self.log_std = nn.Parameter(torch.full((2,), math.log(0.3)))

    def mean(self, obs: Obs) -> torch.Tensor:
        return torch.tanh(self.head(self.encoder(obs.ego, obs.nbr, obs.mask)))

    def dist(self, obs: Obs) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.mean(obs), self.log_std.exp())


class CentralCritic(nn.Module):
    def __init__(self, hidden: int = 96):
        super().__init__()
        self.encoder = NeighborEncoder(hidden)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + GLOBAL_DIM, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1)
        )

    def forward(self, obs: Obs, state: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.encoder(obs.ego, obs.nbr, obs.mask), state], dim=-1)).squeeze(-1)
