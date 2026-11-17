"""결정적 검증기. 규칙 파서 결과든 LLM 결과든 여기를 통과해야 실행한다.

스키마는 모양만 보장하므로, 로봇 ID가 실제로 있는지, 대형이 경기장에 들어가는지 같은
'실행 가능성'은 여기서 확인한다. 문제를 찾으면 사람이 읽을 수 있는 한국어 메시지를 돌려준다.
"""

from __future__ import annotations

import math

from tiggle_swarm.formations import half_extents

from .config import FleetConfig
from .schema import (
    Command, Dock, Follow, Formation, Gather, Move, Point, Rotate, TaskSpec,
)


def _inside(p: Point, fleet: FleetConfig, mx: float, my: float | None = None) -> bool:
    my = mx if my is None else my
    (x0, x1), (y0, y1) = fleet.arena_x, fleet.arena_y
    return x0 + mx <= p.x <= x1 - mx and y0 + my <= p.y <= y1 - my


def _check_command(cmd: Command, fleet: FleetConfig) -> list[str]:
    errs: list[str] = []
    name = cmd.primitive
    robots = cmd.robots
    if not robots:
        errs.append(f"{name}: 대상 로봇이 비어 있습니다")
    unknown = sorted(set(robots) - set(fleet.robot_ids))
    if unknown:
        errs.append(f"{name}: 없는 로봇 ID {unknown}")
    if len(set(robots)) != len(robots):
        errs.append(f"{name}: 같은 로봇이 두 번 지정되었습니다")
    n = len(robots)
    r_robot = fleet.robot_diameter_m / 2
    span_x = fleet.arena_x[1] - fleet.arena_x[0]
    span_y = fleet.arena_y[1] - fleet.arena_y[0]

    if isinstance(cmd, Formation):
        need = fleet.min_robots[cmd.shape]
        if n < need:
            errs.append(f"{cmd.shape} 대형에는 로봇이 {need}대 이상 필요합니다 (지정 {n}대)")
            return errs
        if cmd.radius_m is not None and cmd.spacing_m is not None:
            errs.append("spacing_m과 radius_m을 동시에 지정할 수 없습니다")
        if cmd.radius_m is not None and cmd.shape != "circle":
            errs.append("radius_m은 circle 대형에만 쓸 수 있습니다")
        if cmd.radius_m is not None and cmd.shape == "circle":
            spacing = 2 * cmd.radius_m * math.sin(math.pi / n)
        else:
            spacing = cmd.spacing_m if cmd.spacing_m is not None else fleet.default_spacing_m
        if spacing < fleet.min_spacing_m:
            errs.append(f"로봇 간 간격 {spacing:.2f} m가 최소 {fleet.min_spacing_m:.2f} m보다 좁습니다")
        if spacing > fleet.max_spacing_m:
            errs.append(f"로봇 간 간격 {spacing:.2f} m가 최대 {fleet.max_spacing_m:.2f} m보다 넓습니다")
        hx, hy = half_extents(cmd.shape, n, spacing, cmd.heading_deg)
        hx, hy = hx + r_robot, hy + r_robot
        if 2 * hx > span_x or 2 * hy > span_y:
            errs.append(
                f"{n}대 {cmd.shape} 대형(약 {2 * hx:.2f} x {2 * hy:.2f} m)이 경기장"
                f"({span_x:.2f} x {span_y:.2f} m)에 들어가지 않습니다"
            )
        elif cmd.center is not None and not _inside(cmd.center, fleet, hx, hy):
            errs.append("대형 중심이 경기장 가장자리에 너무 가까워 대형이 밖으로 나갑니다")
    elif isinstance(cmd, Move):
        if cmd.mode == "absolute" and not _inside(Point(x=cmd.x, y=cmd.y), fleet, r_robot):
            errs.append(f"이동 목표 ({cmd.x}, {cmd.y})가 경기장 밖입니다")
        if cmd.mode == "relative" and math.hypot(cmd.x, cmd.y) > math.hypot(span_x, span_y):
            errs.append("이동 거리가 경기장 대각선보다 깁니다")
        if cmd.mode == "relative" and cmd.x == 0 and cmd.y == 0:
            errs.append("이동 거리가 0입니다")
    elif isinstance(cmd, Rotate):
        if not 0 < abs(cmd.angle_deg) <= 720:
            errs.append(f"회전 각도 {cmd.angle_deg}도는 허용 범위(0 초과 720 이하)를 벗어납니다")
    elif isinstance(cmd, Gather):
        if cmd.center is not None and not _inside(cmd.center, fleet, r_robot):
            errs.append("모일 지점이 경기장 밖입니다")
    elif isinstance(cmd, Follow):
        if cmd.leader not in fleet.robot_ids:
            errs.append(f"따라갈 로봇 {cmd.leader}번이 없습니다")
        if cmd.leader in robots:
            errs.append(f"{cmd.leader}번이 자기 자신을 따라갈 수 없습니다")
        if cmd.spacing_m is not None and cmd.spacing_m < fleet.min_spacing_m:
            errs.append(f"따라가기 간격 {cmd.spacing_m:.2f} m가 최소 간격보다 좁습니다")
    elif isinstance(cmd, Dock):
        if n < 2:
            errs.append("도킹에는 로봇이 2대 이상 필요합니다")
    return errs


def validate(spec: TaskSpec, fleet: FleetConfig) -> list[str]:
    """실행 불가능한 이유 목록. 빈 목록이면 통과."""
    errs: list[str] = []
    if not spec.steps:
        return ["실행할 단계가 없습니다"]
    for si, step in enumerate(spec.steps, 1):
        if not step.commands:
            errs.append(f"{si}단계: 명령이 없습니다")
        seen: set[int] = set()
        for cmd in step.commands:
            errs += [f"{si}단계 {e}" for e in _check_command(cmd, fleet)]
            ids = set(cmd.robots)
            if seen & ids:
                errs.append(f"{si}단계: 동시에 실행하는 명령들이 같은 로봇 {sorted(seen & ids)}을 쓰고 있습니다")
            seen |= ids
    return errs
