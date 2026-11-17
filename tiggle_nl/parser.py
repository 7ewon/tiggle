"""L1 파서 진입점: 정규화 -> 규칙 -> 캐시 -> LLM 폴백 -> 검증.

규칙이 해석은 했지만 검증에서 막힌 명령(예: 경기장보다 큰 대형)은 LLM으로 넘기지 않는다.
뜻은 이미 확실하고, 실행할 수 없다는 사실은 LLM이 바꿀 수 없기 때문이다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Protocol

from .cache import SpecCache
from .config import FleetConfig
from .normalize import normalize
from .rules import RuleMiss, rule_parse
from .schema import LLMParse, TaskSpec
from .validate import validate

Source = Literal["rule", "cache", "llm", "none"]


class Fallback(Protocol):
    def parse(self, utterance: str, fleet: FleetConfig) -> LLMParse: ...


@dataclass
class ParseResult:
    ok: bool
    source: Source
    normalized: str
    spec: Optional[TaskSpec] = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "source": self.source,
            "normalized": self.normalized,
            "errors": self.errors,
            "spec": self.spec.model_dump(mode="json") if self.spec else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class CommandParser:
    def __init__(
        self,
        fleet: FleetConfig = FleetConfig(),
        cache_path: Optional[Path] = None,
        fallback: Optional[Fallback] = None,
        use_llm: bool = True,
    ):
        self.fleet = fleet
        self.cache = SpecCache(cache_path)
        self.use_llm = use_llm
        self._fallback = fallback

    @property
    def fallback(self) -> Fallback:
        if self._fallback is None:  # API 키가 없어도 규칙 파서는 쓸 수 있도록 늦게 만든다
            from .llm import HaikuFallback

            self._fallback = HaikuFallback()
        return self._fallback

    def parse(self, utterance: str) -> ParseResult:
        norm = normalize(utterance)
        try:
            spec = rule_parse(norm, self.fleet)
        except RuleMiss as miss:
            rule_reason = str(miss)
        else:
            errs = validate(spec, self.fleet)
            return ParseResult(ok=not errs, source="rule", normalized=norm, spec=spec, errors=errs)

        if (cached := self.cache.get(norm, self.fleet)) is not None:
            errs = validate(cached, self.fleet)
            return ParseResult(ok=not errs, source="cache", normalized=norm, spec=cached, errors=errs)

        if not self.use_llm:
            return ParseResult(ok=False, source="none", normalized=norm, errors=[f"규칙으로 해석하지 못했습니다: {rule_reason}"])

        from .llm import LLMError

        try:
            out = self.fallback.parse(utterance, self.fleet)
        except LLMError as e:
            return ParseResult(ok=False, source="llm", normalized=norm, errors=[str(e)])
        if out.status != "ok" or out.spec is None:
            return ParseResult(ok=False, source="llm", normalized=norm, errors=[f"{out.status}: {out.reason}"])

        errs = validate(out.spec, self.fleet)
        if not errs:  # 검증을 통과한 해석만 캐시한다
            self.cache.put(norm, self.fleet, out.spec, utterance=utterance, source="llm")
        return ParseResult(ok=not errs, source="llm", normalized=norm, spec=out.spec, errors=errs)
