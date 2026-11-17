from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FleetConfig:
    """로봇 구성과 경기장 정보. TODO 표시는 Real Track 실측값으로 교체할 것."""

    robot_ids: tuple[int, ...] = tuple(range(1, 16))
    arena_x: tuple[float, float] = (0.0, 3.0)  # TODO 실제 경기장 크기 (m)
    arena_y: tuple[float, float] = (0.0, 2.0)  # TODO
    robot_diameter_m: float = 0.12  # TODO 실측 (범퍼 포함 외접원 지름)
    max_spacing_m: float = 1.0
    default_spacing_m: float = 0.25  # 대형 명령에 간격이 없을 때 L2가 쓰는 값

    # 명령에 값이 없을 때 규칙 파서가 채우는 기본값
    default_move_m: float = 0.3
    default_rotate_deg: float = 90.0

    # 대형별 최소 로봇 수
    min_robots: dict[str, int] = field(
        default_factory=lambda: {
            "line": 2, "column": 2, "circle": 3, "v": 3,
            "triangle": 3, "square": 4, "grid": 4,
        }
    )

    @property
    def min_spacing_m(self) -> float:
        return self.robot_diameter_m * 1.5

    @property
    def arena_center(self) -> tuple[float, float]:
        return (sum(self.arena_x) / 2, sum(self.arena_y) / 2)

    @property
    def signature(self) -> str:
        """캐시 키에 쓰는 로봇 구성 요약. 로봇 ID가 바뀌면 캐시가 갈린다."""
        return ",".join(map(str, self.robot_ids))
