"""규칙 파서가 놓친 문장을 Claude Haiku 구조화 출력으로 TaskSpec에 맞춘다.

일관성은 temperature가 아니라 (1) JSON 스키마 강제 (2) validate.py 검증 (3) 캐시로 확보한다.
Claude Haiku 5.5는 기본값 외 temperature를 받지 않는다 (보내면 400).
모델은 환경변수 TIGGLE_LLM_MODEL로 바꿀 수 있다 (예: claude-haiku-4-5).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Optional

import anthropic
from pydantic import ValidationError

from .config import FleetConfig
from .schema import LLMParse

DEFAULT_MODEL = "claude-haiku-5-5"


class LLMError(Exception):
    """LLM 호출이나 출력 해석이 실패했다."""


@lru_cache(maxsize=1)
def _output_schema() -> dict:
    # 스키마에서 API가 지원하지 않는 제약을 SDK가 정리해 준다
    return anthropic.transform_schema(LLMParse.model_json_schema())


def _examples(ids: list[int]) -> list[tuple[str, dict]]:
    rest = [i for i in ids if i not in (1, 2, 3)]
    formation = {"spacing_m": None, "radius_m": None, "center": None, "heading_deg": None, "speed": "normal"}
    return [
        (
            "1번부터 3번은 원 만들고 나머지는 일렬로 서",
            {"status": "ok", "reason": "", "spec": {"version": "1.0", "steps": [{"commands": [
                {"primitive": "formation", "robots": [1, 2, 3], "shape": "circle", **formation},
                {"primitive": "formation", "robots": rest, "shape": "line", **formation},
            ]}]}},
        ),
        (
            "다 같이 모인 다음에 천천히 왼쪽으로 반 바퀴 돌아",
            {"status": "ok", "reason": "", "spec": {"version": "1.0", "steps": [
                {"commands": [{"primitive": "gather", "robots": ids, "center": None, "speed": "normal"}]},
                {"commands": [{"primitive": "rotate", "robots": ids, "angle_deg": 180,
                               "pivot": "group_center", "speed": "slow"}]},
            ]}},
        ),
        (
            "춤 좀 춰 봐",
            {"status": "unsupported", "reason": "춤 동작은 지원하는 명령에 없습니다.", "spec": None},
        ),
    ]


def build_system_prompt(fleet: FleetConfig) -> str:
    cx, cy = fleet.arena_center
    ids = list(fleet.robot_ids)
    examples = "\n\n".join(
        f"입력: {u}\n출력: {json.dumps(o, ensure_ascii=False)}" for u, o in _examples(ids)
    )
    return f"""너는 TIGGLE 군집 로봇의 명령 해석기다. 사용자의 한국어 명령을 주어진 JSON 스키마(TaskSpec)로만 바꾼다.

## 로봇과 경기장
- 로봇 ID: {ids} (이 밖의 ID는 절대 만들지 않는다)
- 메카넘 휠 평면 이동 로봇. 경기장 x {fleet.arena_x[0]}~{fleet.arena_x[1]} m, y {fleet.arena_y[0]}~{fleet.arena_y[1]} m
- 좌표 규약: +x = 조작자 기준 앞, +y = 왼쪽. '가운데/중앙' = ({cx}, {cy})
- 각도는 deg, 반시계(왼쪽)가 +, 시계(오른쪽)가 -. 거리는 m (30cm -> 0.3)

## 동작 (primitive)
- stop: 멈춤
- move: mode=relative는 무리 전체를 (x, y)만큼 평행이동, absolute는 무리 중심을 (x, y)로 이동. 거리가 없으면 {fleet.default_move_m} m
- rotate: angle_deg만큼 회전. pivot=group_center(무리 중심 기준) 또는 self(각자 제자리). 각도가 없으면 {fleet.default_rotate_deg}도. 로봇 1대면 self
- formation: shape = line(옆으로 나란히) / column(앞뒤로 줄서기) / circle / v / triangle / square / grid.
  spacing_m(이웃 간격)과 radius_m(원 반지름)은 말한 경우에만, 둘 중 하나만 채운다
- gather: 한 지점으로 모이기
- follow: robots가 leader 로봇을 줄지어 따라감. leader는 robots에 넣지 않는다
- dock: robots 순서대로 자석 커넥터로 연결 / undock: 연결 해제

## 규칙
1. steps는 순서대로, 한 step의 commands는 동시에 실행한다. 동시에 실행하는 명령끼리는 로봇이 겹치면 안 된다.
2. robots는 항상 ID 정수 목록이다. 전체를 뜻하면 위 로봇 ID를 모두 적고, '나머지'는 같은 step에서 이미 지정된 로봇을 뺀 ID를 적는다.
3. 대수만 말하면('3대만') 아직 지정되지 않은 ID 중 작은 번호부터 고른다.
4. 말하지 않은 선택 값은 null, speed는 'normal'. '천천히'는 slow, '빨리'는 fast.
5. 위 동작으로 표현할 수 없으면 status=unsupported, 기본값으로도 채울 수 없는 필수 정보가 빠졌으면
   (예: 누구를 따라갈지 없음) status=ambiguous. 이때 spec은 null, reason에 한국어 한 문장. status=ok이면 reason은 빈 문자열.
6. 실행 가능성(경기장 크기, 간격)은 따로 검사하므로 사용자가 말한 값을 그대로 옮긴다.

## 예시
{examples}"""


class HaikuFallback:
    def __init__(
        self,
        model: Optional[str] = None,
        client: Optional[anthropic.Anthropic] = None,
        timeout: float = 20.0,
    ):
        self.model = model or os.environ.get("TIGGLE_LLM_MODEL", DEFAULT_MODEL)
        self.client = client or anthropic.Anthropic(timeout=timeout, max_retries=2)

    def parse(self, utterance: str, fleet: FleetConfig) -> LLMParse:
        output_config: dict = {"format": {"type": "json_schema", "schema": _output_schema()}}
        # Haiku 4.5는 effort를 받지 않는다. Haiku 5.5는 기본 effort가 medium이라 낮춘다.
        if not self.model.startswith("claude-haiku-4-5"):
            output_config["effort"] = "low"
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=4096,  # 생각(thinking)에 쓰일 여유 포함
                system=build_system_prompt(fleet),
                messages=[{"role": "user", "content": utterance}],
                output_config=output_config,
            )
        except anthropic.RateLimitError as e:
            raise LLMError("API 요청 한도를 넘었습니다. 잠시 후 다시 시도하세요") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"API 오류 {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("API 서버에 연결하지 못했습니다") from e
        except TypeError as e:  # SDK는 인증 정보가 없으면 요청 전에 TypeError를 낸다
            if "authentication" not in str(e):
                raise
            raise LLMError("Claude API 인증 정보가 없습니다 (ANTHROPIC_API_KEY 설정 또는 `ant auth login`)") from e

        if resp.stop_reason == "refusal":
            raise LLMError("모델이 요청을 거절했습니다")
        if resp.stop_reason == "max_tokens":
            raise LLMError("모델 출력이 max_tokens에서 잘렸습니다")
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if text is None:
            raise LLMError("모델 응답에 텍스트가 없습니다")
        try:
            out = LLMParse.model_validate_json(text)
        except ValidationError as e:
            raise LLMError(f"모델 출력이 TaskSpec 스키마와 맞지 않습니다: {e.error_count()}개 오류") from e
        if out.status == "ok" and out.spec is None:
            raise LLMError("status가 ok인데 spec이 비어 있습니다")
        return out
