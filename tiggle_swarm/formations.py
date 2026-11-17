"""대형 슬롯 템플릿. L1 검증기와 L2 계획기가 같은 좌표를 쓴다.

대형 좌표계는 +x가 대형의 앞, +y가 왼쪽이고, 슬롯을 감싸는 상자의 중심이 원점이다.
모든 대형에서 이웃 슬롯 간격은 spacing 이상이다.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def rotate(points: np.ndarray, heading_rad: float) -> np.ndarray:
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    return points @ np.array([[c, s], [-s, c]])


def _centered(points: np.ndarray) -> np.ndarray:
    return points - (points.max(axis=0) + points.min(axis=0)) / 2


def _polygon(n: int, corners: int, spacing: float, first_angle: float) -> np.ndarray:
    """꼭짓점을 슬롯으로 고정하고 남은 슬롯을 변에 고르게 나눈다.

    둘레를 등간격으로 걷지 않는 이유: 꼭짓점을 끼고 있는 두 슬롯의 거리가 spacing의 절반까지 줄어든다.
    """
    inner = n - corners
    per_side = [inner // corners + (1 if s < inner % corners else 0) for s in range(corners)]
    side = (max(per_side) + 1) * spacing
    radius = side / (2 * math.sin(math.pi / corners))
    angles = [first_angle + 2 * math.pi * k / corners for k in range(corners)]
    verts = [np.array([radius * math.cos(a), radius * math.sin(a)]) for a in angles]
    pts = []
    for s, count in enumerate(per_side):
        a, b = verts[s], verts[(s + 1) % corners]
        pts += [a + (b - a) * j / (count + 1) for j in range(count + 1)]
    return np.array(pts)


def _grid(n: int, spacing: float) -> np.ndarray:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    pts = []
    for r in range(rows):
        count = min(cols, n - r * cols)
        pts += [(-r * spacing, (j - (count - 1) / 2) * spacing) for j in range(count)]
    return np.array(pts, dtype=float)


def _v(n: int, spacing: float) -> np.ndarray:
    """꼭짓점이 앞이고 두 팔이 뒤로 45도씩 벌어진다. 짝수 대면 왼팔이 하나 더 길다."""
    pts = [(0.0, 0.0)]
    d = spacing * math.sqrt(0.5)
    for k in range(1, n):
        arm = (k + 1) // 2
        side = 1 if k % 2 else -1
        pts.append((-arm * d, side * arm * d))
    return np.array(pts)


def _cluster(n: int, spacing: float) -> np.ndarray:
    """육각 밀집 배치 중 원점에 가까운 n개. '모여' 명령에 쓴다."""
    reach = math.ceil(math.sqrt(n)) + 2
    pts = []
    for a in range(-reach, reach + 1):
        for b in range(-reach, reach + 1):
            x, y = (a + b / 2) * spacing, b * math.sqrt(3) / 2 * spacing
            pts.append((round(math.hypot(x, y), 9), round(math.atan2(y, x), 9), x, y))
    pts.sort()
    return np.array([(x, y) for _, _, x, y in pts[:n]])


def slot_template(shape: str, n: int, spacing: float, radius: Optional[float] = None) -> np.ndarray:
    """(n, 2) 슬롯 좌표. shape은 TaskSpec의 대형 이름 또는 'cluster'."""
    if shape == "line":
        pts = np.column_stack([np.zeros(n), (np.arange(n) - (n - 1) / 2) * spacing])
    elif shape == "column":
        pts = np.column_stack([((n - 1) / 2 - np.arange(n)) * spacing, np.zeros(n)])
    elif shape == "circle":
        r = radius if radius is not None else spacing / (2 * math.sin(math.pi / n))
        a = 2 * np.pi * np.arange(n) / n
        return np.column_stack([r * np.cos(a), r * np.sin(a)])  # 원은 원의 중심이 곧 대형 중심
    elif shape == "triangle":
        pts = _polygon(n, 3, spacing, 0.0)
    elif shape == "square":
        pts = _polygon(n, 4, spacing, math.pi / 4)
    elif shape == "v":
        pts = _v(n, spacing)
    elif shape == "grid":
        pts = _grid(n, spacing)
    elif shape == "cluster":
        pts = _cluster(n, spacing)
    else:
        raise ValueError(f"알 수 없는 대형: {shape}")
    return _centered(pts)


def half_extents(shape: str, n: int, spacing: float, heading_deg: Optional[float]) -> tuple[float, float]:
    """대형을 경기장 좌표로 돌렸을 때 중심에서 x, y 방향으로 뻗는 최대 길이 (m)."""
    pts = rotate(slot_template(shape, n, spacing), math.radians(heading_deg or 0.0))
    hx, hy = np.abs(pts).max(axis=0)
    return float(hx), float(hy)
