import pytest

from tiggle_nl.config import FleetConfig
from tiggle_nl.schema import TaskSpec
from tiggle_nl.validate import validate

FLEET = FleetConfig()  # 15대, 경기장 3 x 2 m
ALL = list(FLEET.robot_ids)


def spec(*steps: list[dict]) -> TaskSpec:
    return TaskSpec.model_validate({"steps": [{"commands": list(s)} for s in steps]})


def formation(robots, shape, **kw):
    return {"primitive": "formation", "robots": robots, "shape": shape, **kw}


def test_valid_circle_of_all():
    assert validate(spec([formation(ALL, "circle")]), FLEET) == []


@pytest.mark.parametrize(
    "cmd, expected",
    [
        (formation(ALL, "line"), "경기장"),                       # 15대 x 0.25 m = 3.5 m
        (formation([1, 2], "circle"), "3대 이상"),
        (formation([1, 2, 3, 4], "square", spacing_m=0.05), "최소"),
        (formation([1, 2, 3, 4], "line", radius_m=0.3), "circle 대형에만"),
        (formation([1, 2, 3], "circle", center={"x": 0.05, "y": 1.0}), "가장자리"),
        ({"primitive": "stop", "robots": [20]}, "없는 로봇 ID"),
        ({"primitive": "move", "robots": ALL, "mode": "absolute", "x": 10, "y": 10}, "경기장 밖"),
        ({"primitive": "rotate", "robots": ALL, "angle_deg": 0, "pivot": "self"}, "허용 범위"),
        ({"primitive": "follow", "robots": [1, 2], "leader": 1}, "자기 자신"),
        ({"primitive": "dock", "robots": [3]}, "2대 이상"),
    ],
)
def test_invalid_command(cmd, expected):
    errs = validate(spec([cmd]), FLEET)
    assert any(expected in e for e in errs), errs


def test_parallel_commands_must_not_share_robots():
    errs = validate(spec([formation([1, 2, 3], "circle"), {"primitive": "stop", "robots": [3, 4]}]), FLEET)
    assert any("같은 로봇 [3]" in e for e in errs), errs


def test_rotated_line_fits_along_long_side():
    # 8대 line은 기본 heading(옆으로 늘어섬, y 방향 1.75 m+)으로는 경기장 폭 2 m에 들어가고,
    # 90도 돌리면 x 방향 3 m에 들어간다. 12대는 y 방향으로는 못 들어간다.
    assert validate(spec([formation(list(range(1, 9)), "line")]), FLEET) == []
    assert validate(spec([formation(list(range(1, 13)), "line", heading_deg=90)]), FLEET) == []
    assert validate(spec([formation(list(range(1, 13)), "line")]), FLEET) != []
