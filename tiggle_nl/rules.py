"""규칙 기반 파서. 확실히 아는 표현만 처리하고, 조금이라도 애매하면 RuleMiss를 던져 LLM에 넘긴다.

처리 흐름 (입력은 normalize()를 거친 문자열)
1. 순서 표지('그 다음', '-고 나서', '.' 등)로 자른다 -> 단계 후보
2. 각 단계 후보를 쉼표·'-고'·'그리고'로 절로 자른다
3. 절마다 로봇 지칭과 동작을 뽑는다
4. 한 단계 후보의 모든 절이 서로 겹치지 않는 로봇을 명시했으면 동시 실행, 아니면 절마다 순차 실행
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional, Union

from .config import FleetConfig
from .schema import (
    Command, Dock, Follow, Formation, Gather, Move, Point, Rotate, Speed, Step, Stop,
    TaskSpec, Undock,
)


class RuleMiss(Exception):
    """규칙으로 확실하게 해석하지 못했다는 신호."""


_SEQ_SPLIT = re.compile(
    r"\s*(?:그\s*다음에?|그러고\s*나서|그리고\s*나서|그\s*(?:후|뒤)에?|고\s*나서"
    r"|(?<=[가-힣])\s+(?:다음|후|뒤)에?\s+|(?<=[어아해여])서\s+|끝나면|이후에?"
    r"|->|→|;|\.(?!\d))\s*"
)
_SOFT_SPLIT = re.compile(r"\s*(?:,|(?:^|\s)그리고(?:\s|$)|(?<=[가-힣])(?<!빼)(?<!외하)고\s+)\s*")

# 문장에 이런 표현이 있으면 규칙으로 처리하지 않는다 (부정, 조건, 반복, 비교 등)
_GUARD = re.compile(r"말고|않|지\s*마|만약|(?<=[가-힣])(?<!끝나)면(?:\s|$)|동안|계속|반복|씩|마다|대로|처럼")
# 규칙이 모르는 방식으로 로봇을 고르는 표현. 놓치면 명령이 전체 로봇에 잘못 적용된다.
_UNKNOWN_TARGET = re.compile(r"짝수|홀수|다른|애들|(?<!번)만(?:\s|$)|(?<!모든)(?<!나머지)\s로봇")
# '3대만', '두 명' 같은 대수 지정은 어떤 로봇을 고를지 규칙으로 정할 수 없다
_COUNT = re.compile(r"\d+\s*(?:대|명)(?!형)")

_ID_TOKEN = r"\d+(?:~\d+)?번"
_ID_GROUP = re.compile(rf"({_ID_TOKEN}(?:\+{_ID_TOKEN})*)(?:만)?(?:\s*(빼고|제외하고|제외))?(?:들)?(?:은|는|이|가|을|를|도)?")
_ALL = re.compile(r"(?:전부|모두|전체|다\s*같이|모든\s*로봇|전원|다들)(?:\s*다)?(?:들)?(?:은|는|이|가)?")
_REST = re.compile(r"나머지(?:\s*로봇)?(?:들)?(?:은|는|이|가|도)?")

_NUM = r"(-?\d+(?:\.\d+)?)"
_DIST = re.compile(rf"{_NUM}\s*(cm|mm|m)(?![a-z])")
_ANGLE = re.compile(rf"{_NUM}\s*도")
_POINT = re.compile(rf"\(\s*{_NUM}\s+{_NUM}\s*\)")
_CENTER_WORD = re.compile(r"한?가운데|중앙|센터")

_STOP = re.compile(r"멈춰|멈춤|멈추|정지|스톱|stop|그만|대기")
_UNDOCK = re.compile(r"분리|해체|떨어져|(?:도킹|합체|결합|연결)\s*해제|풀어")
_DOCK = re.compile(r"합체|결합|도킹|연결|붙어|붙여")
_FOLLOW_LEADER = re.compile(r"(\d+)번\s*(?:을|를)?\s*따라")
_FOLLOW = re.compile(r"따라")
_ROTATE = re.compile(r"회전|돌아(?!\s*(?:와|오))|돌려|돌기|돌자|턴")
_GATHER = re.compile(r"모여|모이|집합|뭉쳐|집결")
_SHAPES: list[tuple[str, re.Pattern]] = [
    ("circle", re.compile(r"동그라미|원형|원\s*모양|(?<![가-힣])원(?=$|\s|으로|을|이|대형|진)")),
    ("v", re.compile(r"v\s*자|브이")),
    ("triangle", re.compile(r"삼각형|세모")),
    ("square", re.compile(r"정?사각형|네모")),
    ("grid", re.compile(r"격자|바둑판")),
    ("column", re.compile(r"세로|종대|줄\s*(?:을\s*)?서")),
    ("line", re.compile(r"가로|횡대|일렬|1\s*줄|일자")),
]
_DIRS: list[tuple[re.Pattern, tuple[float, float]]] = [
    (re.compile(r"앞|전진"), (1.0, 0.0)),
    (re.compile(r"뒤|후진"), (-1.0, 0.0)),
    (re.compile(r"왼쪽|왼편|좌측"), (0.0, 1.0)),
    (re.compile(r"오른쪽|오른편|우측"), (0.0, -1.0)),
]
_DIR_PHRASE = re.compile(r"(?:앞|뒤|왼쪽|오른쪽|왼편|오른편|좌측|우측)\s*(?:쪽)?\s*(?:으로|로)|전진|후진")
_MOVE_VERB = re.compile(r"이동|움직|옮겨|(?:으로|로)\s*(?:가|와|오|이동)")


# Command.robots는 절을 파싱할 때는 비워 두고, rule_parse가 지칭을 풀어서 채운다.
# 절의 로봇 지칭: None(지칭 없음) | "all" | "rest" | ("ids", [...]) | ("except", [...])
Selector = Union[None, str, tuple[str, list[int]]]


@dataclass
class _Clause:
    selector: Selector
    command: Command


def _expand_ids(group: str) -> list[int]:
    ids: list[int] = []
    for tok in group.split("+"):
        tok = tok.rstrip("번")
        if "~" in tok:
            a, b = map(int, tok.split("~"))
            if a > b:
                raise RuleMiss(f"범위가 거꾸로입니다: {tok}")
            ids.extend(range(a, b + 1))
        else:
            ids.append(int(tok))
    return ids


def _take_selector(text: str) -> tuple[Selector, str]:
    groups = list(_ID_GROUP.finditer(text))
    if len(groups) > 1:
        raise RuleMiss("한 절에 로봇 지칭이 여러 군데 있습니다")
    if groups:
        m = groups[0]
        ids = _expand_ids(m.group(1))
        text = text[: m.start()] + " " + text[m.end():]
        if m.group(2):
            text = _REST.sub(" ", _ALL.sub(" ", text))
            return ("except", ids), text
        return ("ids", ids), text
    if m := _REST.search(text):
        return "rest", text[: m.start()] + " " + text[m.end():]
    if m := _ALL.search(text):
        return "all", text[: m.start()] + " " + text[m.end():]
    return None, text


def _to_m(value: str, unit: str) -> float:
    return float(value) * {"m": 1.0, "cm": 0.01, "mm": 0.001}[unit]


def _speed(text: str) -> Speed:
    if re.search(r"천천히|느리게|살살|조심", text):
        return "slow"
    if re.search(r"빨리|빠르게|급히|신속", text):
        return "fast"
    return "normal"


def _target_point(text: str, fleet: FleetConfig) -> Optional[Point]:
    if m := _POINT.search(text):
        return Point(x=float(m.group(1)), y=float(m.group(2)))
    if _CENTER_WORD.search(text):
        cx, cy = fleet.arena_center
        return Point(x=cx, y=cy)
    return None


def _parse_formation(text: str, shape: str, fleet: FleetConfig) -> Formation:
    spacing = radius = None
    used: list[tuple[int, int]] = []
    if m := re.search(rf"간격\s*(?:을|은|이)?\s*{_DIST.pattern}|{_DIST.pattern}\s*간격", text):
        v, u = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        spacing = _to_m(v, u)
        used.append(m.span())
    if m := re.search(rf"(반지름|반경|지름)\s*(?:이|을|은)?\s*{_DIST.pattern}", text):
        radius = _to_m(m.group(2), m.group(3)) / (2 if m.group(1) == "지름" else 1)
        used.append(m.span())
    for d in _DIST.finditer(text):
        if not any(a <= d.start() < b for a, b in used):
            raise RuleMiss("대형 명령에 용도를 알 수 없는 거리가 있습니다 (간격/반지름을 밝혀 주세요)")
    if radius is not None and shape != "circle":
        raise RuleMiss("반지름은 원 대형에만 쓸 수 있습니다")
    heading = None
    if m := re.search(rf"{_NUM}\s*도\s*(?:방향|쪽)", text):
        heading = float(m.group(1))
    elif _ANGLE.search(text):
        raise RuleMiss("대형 명령에 용도를 알 수 없는 각도가 있습니다")
    return Formation(
        primitive="formation", robots=[], shape=shape, spacing_m=spacing, radius_m=radius,
        center=_target_point(text, fleet), heading_deg=heading, speed=_speed(text),
    )


def _parse_rotate(text: str, fleet: FleetConfig, single: bool) -> Rotate:
    angles = _ANGLE.findall(text)
    if len(angles) > 1:
        raise RuleMiss("회전 각도가 여러 개입니다")
    angle = abs(float(angles[0])) if angles else fleet.default_rotate_deg
    if re.search(r"반시계|왼쪽|좌회전", text):
        sign = 1.0
    elif re.search(r"시계|오른쪽|우회전", text):
        sign = -1.0
    else:
        sign = 1.0
    self_pivot = single or bool(re.search(r"제자리|각자|그\s*자리", text))
    return Rotate(
        primitive="rotate", robots=[], angle_deg=sign * angle,
        pivot="self" if self_pivot else "group_center", speed=_speed(text),
    )


def _parse_move(text: str, fleet: FleetConfig) -> Optional[Move]:
    if _DIR_PHRASE.search(text):
        vx = vy = 0.0
        for pat, (dx, dy) in _DIRS:
            if pat.search(text):
                vx, vy = vx + dx, vy + dy
        norm = math.hypot(vx, vy)
        if norm == 0:
            raise RuleMiss("서로 반대 방향이 함께 있습니다")
        dists = _DIST.findall(text)
        if len(dists) > 1:
            raise RuleMiss("이동 거리가 여러 개입니다")
        d = _to_m(*dists[0]) if dists else fleet.default_move_m
        return Move(
            primitive="move", robots=[], mode="relative",
            x=round(d * vx / norm, 4), y=round(d * vy / norm, 4), speed=_speed(text),
        )
    if _MOVE_VERB.search(text) and (p := _target_point(text, fleet)):
        return Move(primitive="move", robots=[], mode="absolute", x=p.x, y=p.y, speed=_speed(text))
    return None


def _parse_clause(raw: str, fleet: FleetConfig) -> _Clause:
    if _COUNT.search(raw):
        raise RuleMiss("대수로 로봇을 고르는 표현은 규칙으로 처리하지 않습니다")

    text = raw
    leader = None
    if m := _FOLLOW_LEADER.search(text):
        leader = int(m.group(1))
        text = text[: m.start()] + " 따라" + text[m.end():]
    selector, text = _take_selector(text)
    single = isinstance(selector, tuple) and selector[0] == "ids" and len(selector[1]) == 1

    shapes = [name for name, pat in _SHAPES if pat.search(text)]
    strong = {
        "stop": bool(_STOP.search(text)),
        "undock": bool(_UNDOCK.search(text)),
        "dock": bool(_DOCK.search(text)) and not _UNDOCK.search(text),
        "follow": bool(_FOLLOW.search(text)),
        "rotate": bool(_ROTATE.search(text)),
        "formation": bool(shapes),
        "gather": bool(_GATHER.search(text)),
    }
    hits = [k for k, v in strong.items() if v]
    if len(hits) > 1 or len(shapes) > 1:
        raise RuleMiss(f"한 절에 동작이 여러 개입니다: {hits or shapes}")

    kind = hits[0] if hits else None
    # 동작마다 쓸 수 있는 수치 종류. 남는 수치가 있으면 무언가를 놓친 것이다.
    numbers = {"dist": bool(_DIST.search(text)), "angle": bool(_ANGLE.search(text))}
    allowed = {
        "stop": (), "undock": (), "dock": (), "gather": (), "follow": ("dist",),
        "rotate": ("angle",), "formation": ("dist", "angle"), None: ("dist",),
    }[kind]
    if any(v and k not in allowed for k, v in numbers.items()):
        raise RuleMiss(f"'{raw}'에 동작과 맞지 않는 수치가 있습니다")
    cmd: Optional[Command]
    if kind == "stop":
        cmd = Stop(primitive="stop", robots=[])
    elif kind == "undock":
        cmd = Undock(primitive="undock", robots=[])
    elif kind == "dock":
        cmd = Dock(primitive="dock", robots=[])
    elif kind == "follow":
        if leader is None:
            raise RuleMiss("누구를 따라갈지 없습니다")
        sp = _DIST.search(text)
        cmd = Follow(primitive="follow", robots=[], leader=leader, spacing_m=_to_m(*sp.groups()) if sp else None)
    elif kind == "rotate":
        cmd = _parse_rotate(text, fleet, single)
    elif kind == "formation":
        cmd = _parse_formation(text, shapes[0], fleet)
    elif kind == "gather":
        cmd = Gather(primitive="gather", robots=[], center=_target_point(text, fleet), speed=_speed(text))
    else:
        cmd = _parse_move(text, fleet)
    if cmd is None:
        raise RuleMiss(f"동작을 찾지 못했습니다: '{raw}'")
    if leader is not None and kind != "follow":
        raise RuleMiss("'따라'가 아닌 동작에 기준 로봇이 붙었습니다")
    return _Clause(selector=selector, command=cmd)


def _resolve(selector: Selector, fleet: FleetConfig, rest: list[int], cmd: Command) -> list[int]:
    if selector is None or selector == "all":
        if isinstance(cmd, Follow):  # '1번 따라와' = 1번을 뺀 전부
            return [i for i in fleet.robot_ids if i != cmd.leader]
        return list(fleet.robot_ids)
    if selector == "rest":
        return rest
    kind, ids = selector
    if kind == "except":
        return [i for i in fleet.robot_ids if i not in ids]
    return ids


def rule_parse(norm: str, fleet: FleetConfig) -> TaskSpec:
    """정규화된 문장을 TaskSpec으로 바꾼다. 확실하지 않으면 RuleMiss."""
    if _GUARD.search(norm):
        raise RuleMiss("부정·조건·반복 표현은 규칙으로 처리하지 않습니다")
    if _UNKNOWN_TARGET.search(norm):
        raise RuleMiss("규칙이 모르는 방식으로 로봇을 고르고 있습니다")
    steps: list[Step] = []
    segments = [s for s in _SEQ_SPLIT.split(norm) if s and s.strip()]
    if not segments:
        raise RuleMiss("빈 명령")
    for seg in segments:
        clauses = [_parse_clause(c, fleet) for c in _SOFT_SPLIT.split(seg) if c and c.strip()]
        explicit = [c for c in clauses if isinstance(c.selector, tuple)]
        named = [set(_resolve(c.selector, fleet, [], c.command)) for c in explicit]
        disjoint = sum(map(len, named)) == len(set().union(*named)) if named else False
        rest_clauses = [c for c in clauses if c.selector == "rest"]
        parallel = (
            len(clauses) > 1
            and all(c.selector is not None and c.selector != "all" for c in clauses)
            and len(rest_clauses) <= 1
            and disjoint
        )
        if parallel:
            taken = set().union(*named)
            rest = [i for i in fleet.robot_ids if i not in taken]
            cmds = []
            for c in clauses:
                c.command.robots = _resolve(c.selector, fleet, rest, c.command)
                cmds.append(c.command)
            steps.append(Step(commands=cmds))
        else:
            for c in clauses:
                if c.selector == "rest":
                    raise RuleMiss("'나머지'가 무엇의 나머지인지 알 수 없습니다")
                c.command.robots = _resolve(c.selector, fleet, [], c.command)
                steps.append(Step(commands=[c.command]))
    return TaskSpec(steps=steps)
