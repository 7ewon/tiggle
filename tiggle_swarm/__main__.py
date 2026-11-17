"""자연어 명령을 L1~L4까지 통과시켜 추상 시뮬레이터에서 실행한다.

  python -m tiggle_swarm "1~3번은 원 만들고 나머지는 V자로" --n 8
  python -m tiggle_swarm "모여서 원 만들어" --policy checkpoints/residual.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tiggle_nl import CommandParser, FleetConfig

from .pipeline import random_start, run_in_sim
from .sim import IDEAL, DomainRand


def main() -> None:
    ap = argparse.ArgumentParser(prog="tiggle_swarm", description="자연어 명령 -> L1~L4 -> 시뮬레이션")
    ap.add_argument("command", nargs="+")
    ap.add_argument("--n", type=int, default=15, help="로봇 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", type=Path, default=None, help="학습한 MAPPO 보정 정책")
    ap.add_argument("--ideal", action="store_true", help="지연·미끄러짐·노이즈 없는 이상적인 로봇으로 실행")
    ap.add_argument("--llm", action="store_true", help="규칙이 못 푸는 문장에 Claude API를 쓴다")
    args = ap.parse_args()

    fleet = FleetConfig(robot_ids=tuple(range(1, args.n + 1)))
    parsed = CommandParser(fleet, use_llm=args.llm).parse(" ".join(args.command))
    print(f"[L1] {parsed.source}: {parsed.normalized}")
    if not parsed.ok:
        print("해석 실패:", *parsed.errors, sep="\n  ")
        raise SystemExit(1)

    actor, limit = None, 0.05
    if args.policy:
        from .train import load_actor

        actor, limit = load_actor(args.policy)
    result = run_in_sim(
        parsed.spec, fleet, random_start(args.n, fleet, args.seed),
        rand=IDEAL if args.ideal else DomainRand(), actor=actor, residual_limit=limit, seed=args.seed,
    )
    print(f"[L2~L4] {'성공' if result.ok else '실패'}{': ' + result.error if result.error else ''}")
    print(f"  소요 {result.seconds:.1f}초, 최종 위치 오차 평균 {result.final_error * 100:.1f}cm / 최대 {result.worst_error * 100:.1f}cm")
    print(f"  로봇 간 최소 거리 {result.min_distance * 100:.1f}cm, 접촉 {result.contacts}회, CBF 개입 평균 {result.intervention:.4f} m/s")
    for w in result.warnings:
        print("  경고:", w)


if __name__ == "__main__":
    main()
