"""사용법
  python -m tiggle_nl "1~3번은 원, 나머지는 일렬로"     해석 결과 JSON 출력
  python -m tiggle_nl --no-llm "..."                  규칙 파서만 사용
  python -m tiggle_nl --export-schema schema/         TaskSpec JSON 스키마 내보내기
  python -m tiggle_nl                                 대화형 (빈 줄로 종료)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .parser import CommandParser
from .schema import LLMParse, TaskSpec


def main() -> None:
    ap = argparse.ArgumentParser(prog="tiggle_nl", description="TIGGLE 자연어 명령 -> TaskSpec")
    ap.add_argument("command", nargs="*", help="해석할 명령 (없으면 대화형)")
    ap.add_argument("--no-llm", action="store_true", help="LLM 폴백을 끈다")
    ap.add_argument("--cache", type=Path, default=Path("cache/taskspec_cache.json"))
    ap.add_argument("--export-schema", type=Path, metavar="DIR")
    args = ap.parse_args()

    if args.export_schema:
        args.export_schema.mkdir(parents=True, exist_ok=True)
        for name, model in (("taskspec", TaskSpec), ("llm_parse", LLMParse)):
            path = args.export_schema / f"{name}.schema.json"
            path.write_text(json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"wrote {path}")
        return

    parser = CommandParser(cache_path=args.cache, use_llm=not args.no_llm)
    if args.command:
        print(parser.parse(" ".join(args.command)).to_json())
        return
    try:
        while (line := input("명령> ").strip()):
            print(parser.parse(line).to_json())
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
