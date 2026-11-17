from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotLimits:
    """구동 한계와 제어 주기. TODO 표시는 Real Track 실측값으로 교체할 것."""

    dt: float = 0.05  # 제어 주기 (s)
    v_max: float = 0.12  # m/s. N20 60RPM + 48mm 휠의 이론 최대는 0.15. TODO 실측
    omega_max: float = 1.5  # rad/s. TODO 실측

    # 기준 궤적의 최고 속도
    v_slow: float = 0.05
    v_normal: float = 0.08
    v_fast: float = 0.11
    omega_ref: float = 0.8

    safety_margin_m: float = 0.03  # 로봇 지름에 더하는 안전 여유 (L4)
    dock_distance_m: float = 0.12  # 도킹했을 때 중심 간 거리. TODO 실측
    pos_tolerance_m: float = 0.03  # 단계 완료로 보는 위치 오차
    settle_timeout_s: float = 6.0  # 궤적이 끝난 뒤 오차가 줄기를 기다리는 시간

    def v_ref(self, speed: str) -> float:
        return {"slow": self.v_slow, "normal": self.v_normal, "fast": self.v_fast}[speed]


@dataclass(frozen=True)
class ControlGains:
    k_pos: float = 1.2  # 위치 오차 이득 (1/s)
    k_consensus: float = 0.3  # 같은 무리 이웃과 오차를 맞추는 이득 (1/s)
    k_heading: float = 2.0  # 방향 오차 이득 (1/s)
    comm_radius_m: float = 1.0  # 이웃으로 보는 거리
