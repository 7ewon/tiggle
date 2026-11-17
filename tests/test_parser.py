"""파서 전체 흐름과 LLM 폴백. 실제 API는 부르지 않고 가짜 클라이언트로 확인한다."""

import json
from types import SimpleNamespace

import anthropic
import pytest

from tiggle_nl.config import FleetConfig
from tiggle_nl.llm import HaikuFallback, LLMError, build_system_prompt
from tiggle_nl.parser import CommandParser
from tiggle_nl.schema import LLMParse

FLEET = FleetConfig(robot_ids=tuple(range(1, 7)))

CIRCLE_OK = LLMParse.model_validate({
    "status": "ok", "reason": "",
    "spec": {"steps": [{"commands": [{"primitive": "formation", "robots": [1, 2, 3, 4, 5, 6], "shape": "circle", "heading_deg": 0}]}]},
})


class FakeFallback:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def parse(self, utterance, fleet):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_rule_hit_does_not_call_llm(tmp_path):
    fb = FakeFallback(CIRCLE_OK)
    r = CommandParser(FLEET, tmp_path / "c.json", fb).parse("원 만들어")
    assert (r.ok, r.source, fb.calls) == (True, "rule", 0)


def test_infeasible_rule_result_is_not_sent_to_llm(tmp_path):
    fb = FakeFallback(CIRCLE_OK)
    r = CommandParser(FleetConfig(), tmp_path / "c.json", fb).parse("전부 일렬로")
    assert (r.ok, r.source, fb.calls) == (False, "rule", 0)
    assert r.errors


def test_llm_result_is_cached(tmp_path):
    fb = FakeFallback(CIRCLE_OK)
    cache = tmp_path / "c.json"
    p = CommandParser(FLEET, cache, fb)
    first = p.parse("앞을 보고 원 만들어")
    second = p.parse("앞을 보고   원 만들어!")  # 정규화하면 같은 문장
    assert (first.source, second.source, fb.calls) == ("llm", "cache", 1)
    assert first.spec == second.spec
    # 다른 프로세스에서도 같은 결과
    third = CommandParser(FLEET, cache, FakeFallback(LLMError("호출되면 안 됨"))).parse("앞을 보고 원 만들어")
    assert third.source == "cache" and third.spec == first.spec
    assert "앞을 보고 원 만들어" in json.loads(cache.read_text(encoding="utf-8"))["1.0|1,2,3,4,5,6|앞을 보고 원 만들어"]["utterance"]


def test_invalid_llm_result_is_rejected_and_not_cached(tmp_path):
    bad = LLMParse.model_validate({
        "status": "ok", "reason": "",
        "spec": {"steps": [{"commands": [{"primitive": "stop", "robots": [99]}]}]},
    })
    fb = FakeFallback(bad)
    p = CommandParser(FLEET, tmp_path / "c.json", fb)
    assert not p.parse("구십구번 세워").ok
    assert not p.parse("구십구번 세워").ok
    assert fb.calls == 2


@pytest.mark.parametrize(
    "result, expected",
    [
        (LLMParse(status="unsupported", reason="춤은 지원하지 않습니다.", spec=None), "unsupported"),
        (LLMError("API 서버에 연결하지 못했습니다"), "연결"),
    ],
)
def test_llm_failures_are_reported(tmp_path, result, expected):
    r = CommandParser(FLEET, tmp_path / "c.json", FakeFallback(result)).parse("춤 춰")
    assert not r.ok and expected in r.errors[0]


def test_no_llm_mode(tmp_path):
    r = CommandParser(FLEET, tmp_path / "c.json", use_llm=False).parse("춤 춰")
    assert (r.ok, r.source) == (False, "none")


# --- HaikuFallback: 요청 형태와 응답 처리 ---

class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def fake_client(text, stop_reason="end_turn"):
    content = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(messages=FakeMessages(SimpleNamespace(stop_reason=stop_reason, content=content)))


def test_haiku_request_shape():
    client = fake_client(CIRCLE_OK.model_dump_json())
    out = HaikuFallback(model="claude-haiku-5-5", client=client).parse("앞을 보고 원", FLEET)
    kw = client.messages.kwargs
    assert out == CIRCLE_OK
    assert kw["model"] == "claude-haiku-5-5"
    assert kw["output_config"]["effort"] == "low"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in kw  # Haiku 5.5는 기본값 외 temperature를 받지 않는다


def test_haiku_4_5_omits_effort():
    client = fake_client(CIRCLE_OK.model_dump_json())
    HaikuFallback(model="claude-haiku-4-5", client=client).parse("앞을 보고 원", FLEET)
    assert "effort" not in client.messages.kwargs["output_config"]


@pytest.mark.parametrize(
    "text, stop_reason, expected",
    [
        ("", "refusal", "거절"),
        ('{"status": "ok"', "max_tokens", "잘렸"),
        ('{"status": "ok", "reason": "", "spec": {"steps": [{"commands": [{"primitive": "jump", "robots": [1]}]}]}}',
         "end_turn", "스키마"),
    ],
)
def test_haiku_bad_responses(text, stop_reason, expected):
    with pytest.raises(LLMError, match=expected):
        HaikuFallback(client=fake_client(text, stop_reason)).parse("x", FLEET)


def test_missing_credentials_becomes_llm_error(monkeypatch, tmp_path):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # 디스크의 ant 프로필도 읽지 않게
    client = anthropic.Anthropic(base_url="http://127.0.0.1:9", max_retries=0, timeout=1)
    with pytest.raises(LLMError, match="인증"):
        HaikuFallback(client=client).parse("x", FLEET)


def test_output_schema_within_api_limits():
    """구조화 출력 한도: union 파라미터 16개, 선택 파라미터 24개. 넘으면 실제 API가 400을 낸다."""
    schema = anthropic.transform_schema(LLMParse.model_json_schema())

    def nodes(node):
        if isinstance(node, dict):
            yield node
            for v in node.values():
                yield from nodes(v)
        elif isinstance(node, list):
            for v in node:
                yield from nodes(v)

    every = list(nodes(schema))
    objs = [n for n in every if n.get("type") == "object"]
    unions = [n for n in every if "anyOf" in n or isinstance(n.get("type"), list)]
    optional = sum(len(set(o.get("properties", {})) - set(o.get("required", []))) for o in objs)
    assert objs and all(o.get("additionalProperties") is False for o in objs)
    assert len(unions) <= 12, len(unions)  # 16이 한도지만 복잡도 오류를 피하려 여유를 둔다
    assert optional <= 24, optional


def test_system_prompt_lists_fleet():
    prompt = build_system_prompt(FLEET)
    assert "[1, 2, 3, 4, 5, 6]" in prompt and "(1.5, 1.0)" in prompt
