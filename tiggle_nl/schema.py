"""TaskSpec: 자연어 명령을 해석한 결과를 L2 계획기로 넘기는 JSON 형식.

좌표 규약 (경기장 좌표계, 단위 m / deg)
- +x = 조작자 기준 '앞', +y = '왼쪽', 원점은 경기장 왼쪽 아래 모서리
- 각도는 반시계(왼쪽)가 +, 시계(오른쪽)가 -
- heading 0도 = +x 방향

구조
- TaskSpec.steps 는 순서대로 실행한다.
- 한 Step 안의 commands 는 동시에 실행하며, 대상 로봇 집합이 서로 겹치면 안 된다.

수치 범위(간격·경기장 경계 등)는 로봇 대수와 경기장 크기에 따라 달라지므로
여기서 제약을 걸지 않고 validate.py 에서 검사한다.

Claude 구조화 출력은 스키마 전체에서 union(anyOf) 파라미터 16개, 선택 파라미터 24개까지만 받는다.
넘으면 모든 요청이 400으로 실패하므로, Optional 필드를 늘릴 때는 tests/test_parser.py의 한도 테스트를 확인할 것.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

Speed = Literal["slow", "normal", "fast"]
Shape = Literal["line", "column", "circle", "v", "triangle", "square", "grid"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Point(_Strict):
    x: float = Field(description="경기장 좌표 x (m)")
    y: float = Field(description="경기장 좌표 y (m)")


class _Command(_Strict):
    robots: list[int] = Field(description="대상 로봇 ID 목록. 전체를 뜻하면 모든 ID를 적는다")


class Stop(_Command):
    primitive: Literal["stop"]


class Move(_Command):
    primitive: Literal["move"]
    mode: Literal["relative", "absolute"] = Field(
        description="relative: 무리 전체를 (x, y)만큼 평행이동 / absolute: 무리 중심을 (x, y) 지점으로 이동"
    )
    x: float = Field(description="m")
    y: float = Field(description="m")
    speed: Speed = "normal"


class Rotate(_Command):
    primitive: Literal["rotate"]
    angle_deg: float = Field(description="+ 반시계(왼쪽), - 시계(오른쪽)")
    pivot: Literal["group_center", "self"] = Field(
        description="group_center: 무리 중심 기준 공전 / self: 각자 제자리 회전"
    )
    speed: Speed = "normal"


class Formation(_Command):
    primitive: Literal["formation"]
    shape: Shape = Field(
        description="line: 가로 일렬(옆으로 나란히) / column: 세로 일렬(앞뒤로 줄서기) / circle / v / triangle / square / grid"
    )
    spacing_m: Optional[float] = Field(
        default=None, description="이웃 로봇 간 간격 (m). null이면 기본값"
    )
    radius_m: Optional[float] = Field(
        default=None, description="circle 전용 반지름 (m). spacing_m과 동시에 쓰지 않는다"
    )
    center: Optional[Point] = Field(
        default=None, description="대형 중심. null이면 현재 무게중심"
    )
    heading_deg: Optional[float] = Field(
        default=None, description="대형이 바라보는 방향. null이면 0도(앞)"
    )
    speed: Speed = "normal"


class Gather(_Command):
    primitive: Literal["gather"]
    center: Optional[Point] = Field(
        default=None, description="모일 지점. null이면 현재 무게중심"
    )
    speed: Speed = "normal"


class Follow(_Command):
    primitive: Literal["follow"]
    leader: int = Field(description="따라갈 로봇 ID. robots에 포함되면 안 된다")
    spacing_m: Optional[float] = Field(default=None, description="앞 로봇과의 간격 (m)")


class Dock(_Command):
    primitive: Literal["dock"]
    # robots 순서대로 자석 커넥터를 체인 형태로 연결한다.


class Undock(_Command):
    primitive: Literal["undock"]


Command = Union[Stop, Move, Rotate, Formation, Gather, Follow, Dock, Undock]


class Step(_Strict):
    commands: list[Command] = Field(description="동시에 실행할 명령들")


class TaskSpec(_Strict):
    version: Literal["1.0"] = SCHEMA_VERSION
    steps: list[Step] = Field(description="순서대로 실행할 단계들")


class LLMParse(_Strict):
    """LLM 폴백의 출력. 해석할 수 없으면 spec 대신 이유를 돌려준다."""

    status: Literal["ok", "unsupported", "ambiguous"] = Field(
        description="ok: 해석 성공 / unsupported: 지원하지 않는 동작 / ambiguous: 필수 정보가 빠져 해석 불가"
    )
    reason: str = Field(description="status가 ok가 아닐 때 이유 (한국어 한 문장). ok이면 빈 문자열")
    spec: Optional[TaskSpec] = Field(description="status가 ok일 때만 채운다")
