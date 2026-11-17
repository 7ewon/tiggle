"""실제 Claude API로 LLM 폴백을 시험한다. 규칙 파서가 놓치는 문장만 보낸다.

  export ANTHROPIC_API_KEY=...     (또는 ant auth login)
  python scripts/live_test.py [--repeat 3] [--model claude-haiku-4-5]

같은 문장을 여러 번 보내 결과가 매번 같은지도 확인한다. 캐시는 쓰지 않는다.
종료 코드: 0 전부 통과, 1 일부 실패, 2 API 호출 자체가 실패.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic

from tiggle_nl.config import FleetConfig
from tiggle_nl.llm import HaikuFallback, LLMError
from tiggle_nl.normalize import normalize
from tiggle_nl.rules import RuleMiss, rule_parse
from tiggle_nl.schema import LLMParse
from tiggle_nl.validate import validate

# 100만 토큰당 (입력, 출력) USD. 추정용이며 실제 청구액은 콘솔에서 확인할 것.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-haiku-4-5": (1.00, 5.00)}
ALL = list(range(1, 16))


@dataclass
class Case:
    text: str
    accept: tuple[str, ...]  # 허용하는 status
    primitive: Optional[str] = None  # status가 ok일 때 첫 명령의 동작
    robots: Optional[list[int]] = None  # status가 ok일 때 첫 명령의 대상
    shape: Optional[str] = None


CASES = [
    Case("앞을 보고 원 만들어", ("ok",), "formation", ALL, "circle"),
    Case("3대만 원 만들어", ("ok",), "formation", [1, 2, 3], "circle"),
    Case("원 말고 사각형으로 서", ("ok",), "formation", ALL, "square"),
    Case("짝수 번호만 왼쪽으로 20센치 옮겨줘", ("ok",), "move", [2, 4, 6, 8, 10, 12, 14]),
    Case("1번이 앞장서고 나머지는 줄줄이 따라가", ("ok",), "follow", ALL[1:]),
    Case("홀수 로봇은 삼각형 만들고 짝수 로봇은 제자리에서 멈춰", ("ok",), "formation", ALL[::2], "triangle"),
    Case("50cm로 원 만들어", ("ok", "ambiguous"), "formation", ALL, "circle"),
    Case("나머지는 멈춰", ("ambiguous", "unsupported")),
    Case("춤 춰", ("unsupported",)),
    Case("커피 좀 타 와", ("unsupported",)),
]


class RecordingClient:
    """HaikuFallback이 부르는 messages.create를 감싸 토큰 사용량과 시간을 모은다."""

    def __init__(self, client: anthropic.Anthropic):
        self._client = client
        self.messages = self
        self.usages: list = []
        self.seconds: list[float] = []

    def create(self, **kwargs):
        start = time.perf_counter()
        resp = self._client.messages.create(**kwargs)
        self.seconds.append(time.perf_counter() - start)
        self.usages.append(resp.usage)
        return resp


def judge(case: Case, out: LLMParse, fleet: FleetConfig) -> list[str]:
    if out.status not in case.accept:
        return [f"status={out.status} ({out.reason})"]
    if out.status != "ok":
        return []
    problems = list(validate(out.spec, fleet))
    first = out.spec.steps[0].commands[0]
    if case.primitive and first.primitive != case.primitive:
        problems.append(f"동작 {first.primitive} (기대 {case.primitive})")
    if case.robots is not None and sorted(first.robots) != case.robots:
        problems.append(f"대상 {first.robots} (기대 {case.robots})")
    if case.shape and getattr(first, "shape", None) != case.shape:
        problems.append(f"모양 {getattr(first, 'shape', None)} (기대 {case.shape})")
    return problems


def describe(out: LLMParse) -> str:
    if out.status != "ok":
        return f"{out.status}: {out.reason}"
    steps = []
    for step in out.spec.steps:
        cmds = []
        for cmd in step.commands:
            fields = cmd.model_dump(exclude={"primitive", "robots"}, exclude_none=True)
            if fields.get("speed") == "normal":
                del fields["speed"]
            cmds.append(f"{cmd.primitive}{cmd.robots} {fields}" if fields else f"{cmd.primitive}{cmd.robots}")
        steps.append(" & ".join(cmds))
    return " -> ".join(steps)


def main() -> int:
    ap = argparse.ArgumentParser(description="실제 API로 LLM 폴백 시험")
    ap.add_argument("--repeat", type=int, default=3, help="문장당 호출 횟수")
    ap.add_argument("--model", default=None, help="기본: TIGGLE_LLM_MODEL 또는 claude-haiku-5-5")
    args = ap.parse_args()

    fleet = FleetConfig()
    client = RecordingClient(anthropic.Anthropic(timeout=30.0, max_retries=2))
    fallback = HaikuFallback(model=args.model, client=client)
    print(f"모델 {fallback.model}, 문장 {len(CASES)}개 x {args.repeat}회\n")

    passed = consistent = tested = 0
    for case in CASES:
        try:
            rule_parse(normalize(case.text), fleet)
        except RuleMiss:
            pass
        else:
            print(f"[건너뜀] {case.text}\n    규칙 파서가 처리하므로 LLM을 부르지 않음\n")
            continue

        tested += 1
        problems: list[str] = []
        outputs: list[LLMParse] = []
        for _ in range(args.repeat):
            try:
                outputs.append(fallback.parse(case.text, fleet))
            except LLMError as e:
                if not client.usages:  # 첫 호출부터 실패하면 설정 문제이므로 더 부르지 않는다
                    print(f"API 호출 실패: {e}", file=sys.stderr)
                    return 2
                problems.append(str(e))

        for out in outputs:
            problems += [p for p in judge(case, out, fleet) if p not in problems]
        variants = {o.model_dump_json(exclude={"reason"}) for o in outputs}
        ok = not problems
        passed += ok
        consistent += len(variants) == 1 and len(outputs) == args.repeat

        print(f"[{'통과' if ok else '실패'}] {case.text}")
        print(f"    동일한 결과 {args.repeat - len(variants) + 1 if variants else 0}/{args.repeat} (서로 다른 결과 {len(variants)}종)")
        if outputs:
            print(f"    {describe(outputs[0])}")
        for p in problems:
            print(f"    ! {p}")
        print()

    in_tok = sum(u.input_tokens for u in client.usages)
    out_tok = sum(u.output_tokens for u in client.usages)
    n = len(client.usages)
    print(f"통과 {passed}/{tested}, 결과가 매번 같음 {consistent}/{tested}")
    print(f"호출 {n}회, 평균 {sum(client.seconds) / max(n, 1):.2f}초, 입력 {in_tok:,} / 출력 {out_tok:,} 토큰")
    if fallback.model in PRICES:
        pi, po = PRICES[fallback.model]
        print(f"추정 비용 ${(in_tok * pi + out_tok * po) / 1e6:.4f}")
    return 0 if passed == tested else 1


if __name__ == "__main__":
    sys.exit(main())
