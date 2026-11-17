import pytest

from tiggle_nl.config import FleetConfig
from tiggle_nl.normalize import normalize
from tiggle_nl.rules import RuleMiss, rule_parse

FLEET = FleetConfig(robot_ids=tuple(range(1, 7)))  # 6대: 대형 크기 오류 없이 의미만 본다
ALL = list(FLEET.robot_ids)


def steps(text: str) -> list[list[dict]]:
    """TaskSpec을 비교하기 쉬운 형태로: 값이 null/기본값인 필드는 뺀다."""
    spec = rule_parse(normalize(text), FLEET)
    return [
        [
            {k: v for k, v in cmd.model_dump().items() if v is not None and not (k == "speed" and v == "normal")}
            for cmd in step.commands
        ]
        for step in spec.steps
    ]


@pytest.mark.parametrize(
    "raw, norm",
    [
        ("세 대", "3대"),
        ("앞으로 삼십 센티", "앞으로 30cm"),
        ("구십도로 돌아", "90도로 돌아"),
        ("반 바퀴", "180도"),
        ("두 바퀴", "720도"),
        ("반 미터", "0.5m"),
        ("1번 로봇이랑 3번 로봇", "1번+3번"),
        ("1번부터 4번까지", "1~4번"),
        ("1번, 2번, 5번", "1번+2번+5번"),
        ("(1, 0.5)로", "(1 0.5)로"),
        ("이번에는 원", "이번에는 원"),  # '이번'을 2번으로 바꾸지 않는다
        ("전부 멈춰!", "전부 멈춰"),
    ],
)
def test_normalize(raw, norm):
    assert normalize(raw) == norm


@pytest.mark.parametrize(
    "text, expected",
    [
        ("전부 멈춰!", [[{"robots": ALL, "primitive": "stop"}]]),
        ("앞으로 삼십 센티 가", [[{"robots": ALL, "primitive": "move", "mode": "relative", "x": 0.3, "y": 0.0}]]),
        ("뒤로 와", [[{"robots": ALL, "primitive": "move", "mode": "relative", "x": -0.3, "y": 0.0}]]),
        ("1번 로봇이랑 3번 로봇 왼쪽으로 50cm 이동",
         [[{"robots": [1, 3], "primitive": "move", "mode": "relative", "x": 0.0, "y": 0.5}]]),
        ("가운데로 와", [[{"robots": ALL, "primitive": "move", "mode": "absolute", "x": 1.5, "y": 1.0}]]),
        ("(1, 0.5)로 이동해", [[{"robots": ALL, "primitive": "move", "mode": "absolute", "x": 1.0, "y": 0.5}]]),
        ("왼쪽으로 90도 돌아",
         [[{"robots": ALL, "primitive": "rotate", "angle_deg": 90.0, "pivot": "group_center"}]]),
        ("시계 방향으로 반 바퀴 회전",
         [[{"robots": ALL, "primitive": "rotate", "angle_deg": -180.0, "pivot": "group_center"}]]),
        ("1번 제자리에서 한 바퀴 돌아",
         [[{"robots": [1], "primitive": "rotate", "angle_deg": 360.0, "pivot": "self"}]]),
        ("원 만들어", [[{"robots": ALL, "primitive": "formation", "shape": "circle"}]]),
        ("반지름 50센티 원", [[{"robots": ALL, "primitive": "formation", "shape": "circle", "radius_m": 0.5}]]),
        ("간격 30cm로 V자 대형", [[{"robots": ALL, "primitive": "formation", "shape": "v", "spacing_m": 0.3}]]),
        ("세로로 줄 서", [[{"robots": ALL, "primitive": "formation", "shape": "column"}]]),
        ("한 줄로 서", [[{"robots": ALL, "primitive": "formation", "shape": "line"}]]),
        ("3번 빼고 삼각형 만들어",
         [[{"robots": [1, 2, 4, 5, 6], "primitive": "formation", "shape": "triangle"}]]),
        ("가운데에 모여", [[{"robots": ALL, "primitive": "gather", "center": {"x": 1.5, "y": 1.0}}]]),
        ("1번 따라와", [[{"robots": [2, 3, 4, 5, 6], "primitive": "follow", "leader": 1}]]),
        ("2번이랑 3번은 1번 따라가", [[{"robots": [2, 3], "primitive": "follow", "leader": 1}]]),
        ("1번이랑 2번 합체해", [[{"robots": [1, 2], "primitive": "dock"}]]),
        ("도킹 해제", [[{"robots": ALL, "primitive": "undock"}]]),
        ("모든 로봇 멈춰", [[{"robots": ALL, "primitive": "stop"}]]),
        ("1번만 앞으로", [[{"robots": [1], "primitive": "move", "mode": "relative", "x": 0.3, "y": 0.0}]]),
        ("1번은 멈추고 나머지 로봇은 원 만들어",
         [[{"robots": [1], "primitive": "stop"},
           {"robots": [2, 3, 4, 5, 6], "primitive": "formation", "shape": "circle"}]]),
        # 동시 실행: 절마다 서로 다른 로봇을 명시
        ("1번부터 3번은 원 만들고 나머지는 일렬로 서",
         [[{"robots": [1, 2, 3], "primitive": "formation", "shape": "circle"},
           {"robots": [4, 5, 6], "primitive": "formation", "shape": "line"}]]),
        ("1번은 앞으로, 2번은 뒤로",
         [[{"robots": [1], "primitive": "move", "mode": "relative", "x": 0.3, "y": 0.0},
           {"robots": [2], "primitive": "move", "mode": "relative", "x": -0.3, "y": 0.0}]]),
        # 순차 실행
        ("모여서 원 만들어",
         [[{"robots": ALL, "primitive": "gather"}],
          [{"robots": ALL, "primitive": "formation", "shape": "circle"}]]),
        ("원 만들고 나서 천천히 앞으로 50cm",
         [[{"robots": ALL, "primitive": "formation", "shape": "circle"}],
          [{"robots": ALL, "primitive": "move", "mode": "relative", "x": 0.5, "y": 0.0, "speed": "slow"}]]),
        ("1번은 멈추고 전부 원 만들어",
         [[{"robots": [1], "primitive": "stop"}],
          [{"robots": ALL, "primitive": "formation", "shape": "circle"}]]),
    ],
)
def test_rule_parse(text, expected):
    assert steps(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "3대만 원 만들어",          # 어떤 3대인지 규칙으로 못 정함
        "원 말고 사각형",            # 부정
        "앞을 보고 원 만들어",        # 방향 지정 표현
        "춤 춰",                    # 없는 동작
        "나머지는 멈춰",             # 무엇의 나머지인지 없음
        "50cm로 원 만들어",          # 간격인지 반지름인지 모름
        "원 그리면서 돌아",           # 대형 + 회전이 한 절에
        "따라와",                   # 따라갈 로봇 없음
        "앞뒤로 가",                # 반대 방향
        "가까이 오면 멈춰",           # 조건
        "앞으로 30cm씩 세 번 가",     # 반복
        # 규칙이 모르는 대상 지정: 전체 로봇으로 잘못 해석하면 안 된다
        "짝수 번호만 왼쪽으로 20cm 옮겨",
        "홀수 로봇은 멈춰",
        "왼쪽 로봇들 멈춰",
        "1번은 앞으로, 다른 애들은 뒤로",
    ],
)
def test_rule_miss(text):
    with pytest.raises(RuleMiss):
        rule_parse(normalize(text), FLEET)
