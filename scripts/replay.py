"""명령 하나를 L1~L4 + 추상 시뮬레이터로 실행하고, 브라우저에서 재생하는 HTML 한 파일로 저장한다.

  python scripts/replay.py "모여서 원 만들어" --n 15 --seed 3 -o replays/demo.html

기본은 지연·미끄러짐·노이즈를 넣은 시뮬레이터(DomainRand)와 기본 제어기+CBF다.
--ideal은 이상적인 로봇, --policy는 학습한 MAPPO 보정 정책을 쓴다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from tiggle_nl import CommandParser, FleetConfig
from tiggle_nl.schema import Command, Dock, Follow, Formation, Gather, Move, Rotate, Stop, TaskSpec, Undock
from tiggle_swarm.config import RobotLimits
from tiggle_swarm.control import pairwise
from tiggle_swarm.pipeline import SwarmController, random_start
from tiggle_swarm.planner import FollowChain, PlanError
from tiggle_swarm.sim import IDEAL, DomainRand, SwarmSim

SHAPE_NAMES = {"line": "가로 일렬", "column": "세로 일렬", "circle": "원", "v": "V자", "triangle": "삼각형", "square": "사각형", "grid": "격자"}


def robots_text(ids: list[int], n: int) -> str:
    ids = sorted(ids)
    if ids == list(range(1, n + 1)):
        return "전체"
    runs, start, prev = [], ids[0], ids[0]
    for i in ids[1:]:
        if i != prev + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return ", ".join(f"{a}~{b}" if b > a else f"{a}" for a, b in runs) + "번"


def describe(cmd: Command, n: int) -> str:
    who = robots_text(cmd.robots, n)
    if isinstance(cmd, Formation):
        return f"{who} {SHAPE_NAMES[cmd.shape]} 대형"
    if isinstance(cmd, Rotate):
        turn = "반시계" if cmd.angle_deg > 0 else "시계"
        return f"{who} {turn} {abs(cmd.angle_deg):.0f}도 회전" + (" (제자리)" if cmd.pivot == "self" else "")
    if isinstance(cmd, Move):
        if cmd.mode == "absolute":
            return f"{who} ({cmd.x:.2f}, {cmd.y:.2f})로 이동"
        return f"{who} 이동 ({cmd.x:+.2f}, {cmd.y:+.2f}) m"
    if isinstance(cmd, Gather):
        return f"{who} 모이기"
    if isinstance(cmd, Follow):
        return f"{who} {cmd.leader}번 따라가기"
    if isinstance(cmd, Dock):
        return f"{who} 도킹"
    if isinstance(cmd, Undock):
        return f"{who} 분리"
    if isinstance(cmd, Stop):
        return f"{who} 정지"
    return cmd.primitive


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)[None]


def _step_info(ctrl: SwarmController, spec: TaskSpec, step: int, n: int, frame: int, fleet: FleetConfig) -> dict:
    """이번 단계에서 로봇마다 몇 번째 명령에 속하는지와 도착 위치."""
    commands = spec.steps[step].commands
    member, goals = [-1] * n, [None] * n
    for active in ctrl.executor._step_active:
        motion = active.motion
        ids = {fleet.robot_ids[int(i)] for i in motion.indices}
        order = next(k for k, c in enumerate(commands) if set(c.robots) == ids)
        for i in motion.indices:
            member[int(i)] = order
        if not isinstance(motion, FollowChain):
            for i, g in zip(motion.indices, motion.final()):
                goals[int(i)] = [round(float(g[0]) * 1000), round(float(g[1]) * 1000)]
    return {"frame": frame, "labels": [describe(c, n) for c in commands], "member": member, "goals": goals}


def record(text: str, n: int, seed: int, rand: DomainRand, actor=None, residual_limit: float = 0.05, every: int = 2) -> dict:
    fleet = FleetConfig(robot_ids=tuple(range(1, n + 1)))
    parsed = CommandParser(fleet, use_llm=False).parse(text)
    if not parsed.ok:
        raise SystemExit("해석 실패: " + "; ".join(parsed.errors))
    spec, limits = parsed.spec, RobotLimits()
    start = random_start(n, fleet, seed)
    sim = SwarmSim(1, n, fleet, limits, rand, generator=torch.Generator().manual_seed(seed))
    sim.reset(_t(start.pos), _t(start.heading))
    ctrl = SwarmController(spec, start, fleet, actor=actor, residual_limit=residual_limit)

    frames, hud, steps, docks = [], [], [], []
    last_step, last_dock, contacts, closest, error = None, None, 0, float("inf"), None
    done_at, k, ref_pos = None, 0, start.pos
    while True:
        t = k * limits.dt
        p, th, v = sim.observe()
        try:
            out = ctrl.step(t, p[0].numpy(), th[0].numpy(), v[0].numpy())
        except PlanError as e:
            error = str(e)
            break
        ref = out.reference
        ref_pos = ref.pos
        if not ref.done and ref.step != last_step:
            last_step = ref.step
            steps.append(_step_info(ctrl, spec, ref.step, n, len(frames), fleet))
        pairs = np.argwhere(np.triu(ref.docked)).tolist()
        if pairs != last_dock:
            docks.append([len(frames), pairs])
            last_dock = pairs
        sim.exempt = torch.as_tensor(ref.docked, dtype=torch.bool)[None]
        sim.step(_t(out.velocity), _t(out.omega))
        contacts += int(sim.contacts.sum()) // 2
        dist = float(pairwise(sim.pos)[1].masked_fill(sim.exempt, float("inf")).min()) if n > 1 else float("inf")
        closest = min(closest, dist)
        if k % every == 0:
            pos, heading = sim.pos[0].numpy(), sim.heading[0].numpy()
            frames.append([v for i in range(n) for v in (round(float(pos[i, 0]) * 1000), round(float(pos[i, 1]) * 1000), round(float(heading[i]) * 1000))])
            hud.append([round(dist * 1000) if dist < 1e9 else -1, contacts])
        k += 1
        if ref.done and done_at is None:
            done_at = k * limits.dt
        if (done_at is not None and k * limits.dt >= done_at + 2.0) or k * limits.dt >= 180:
            break

    final = np.linalg.norm(sim.pos[0].numpy() - ref_pos, axis=1)
    return {
        "text": text,
        "normalized": parsed.normalized,
        "source": parsed.source,
        "n": n,
        "arena": [round((fleet.arena_x[1] - fleet.arena_x[0]) * 1000), round((fleet.arena_y[1] - fleet.arena_y[0]) * 1000)],
        "diameter": round(fleet.robot_diameter_m * 1000),
        "dt": limits.dt * every,
        "steps": steps,
        "docks": docks,
        "frames": frames,
        "hud": hud,
        "summary": {
            "ok": error is None and done_at is not None and contacts == 0,
            "error": error or ("시간 안에 끝나지 않음" if done_at is None else None),
            "seconds": round(k * limits.dt, 1),
            "final_mean_cm": round(float(final.mean()) * 100, 1),
            "final_max_cm": round(float(final.max()) * 100, 1),
            "closest_cm": round(closest * 100, 1),
            "contacts": contacts,
            "warnings": ctrl.executor.warnings,
            "controller": "기본 제어 + MAPPO 보정 + CBF" if actor is not None else "기본 제어 + CBF",
            "dynamics": "이상적인 로봇" if rand is IDEAL else "지연·미끄러짐·노이즈 포함 (가정값)",
            "seed": seed,
        },
    }


PAGE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TIGGLE 군집 재생</title>
<style>
:root { --bg:#fbfaf7; --fg:#1f1e1c; --muted:#6b6a65; --line:#dcdad2; --arena:#ffffff; --grid:#efede6; --hold:#9c9a92; --chip:#f1efe8; --accent:#534AB7; }
@media (prefers-color-scheme: dark) { :root { --bg:#1d1d1b; --fg:#ecebe6; --muted:#a6a49c; --line:#3a3936; --arena:#262624; --grid:#30302d; --hold:#75736c; --chip:#2c2c2a; --accent:#AFA9EC; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.6 -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif; }
main { max-width:960px; margin:0 auto; padding:20px 16px 32px; }
h1 { font-size:20px; font-weight:500; margin:0 0 4px; }
.meta { color:var(--muted); font-size:13px; margin:0 0 12px; }
.stats { display:flex; flex-wrap:wrap; gap:6px 16px; font-size:13px; margin-bottom:12px; }
.stats b { font-weight:500; }
.bad { color:#c0392b; }
.steps { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px; }
.steps button { border:1px solid var(--line); background:var(--chip); color:var(--fg); border-radius:8px; padding:4px 10px; font:inherit; font-size:13px; cursor:pointer; text-align:left; }
.steps button.on { border-color:var(--accent); outline:1px solid var(--accent); }
svg { width:100%; height:auto; display:block; background:var(--arena); border:1px solid var(--line); border-radius:12px; }
.bar { display:flex; align-items:center; gap:10px; margin-top:10px; flex-wrap:wrap; font-size:13px; }
.bar button, .bar select { font:inherit; font-size:13px; padding:4px 10px; border-radius:8px; border:1px solid var(--line); background:var(--chip); color:var(--fg); cursor:pointer; }
.bar input[type=range] { flex:1; min-width:160px; }
.hud { font-variant-numeric: tabular-nums; color:var(--muted); font-size:13px; margin-top:6px; }
.legend { display:flex; flex-wrap:wrap; gap:6px 16px; font-size:12px; color:var(--muted); margin-top:10px; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:-1px; }
</style>
</head>
<body>
<main>
<h1 id="title"></h1>
<p class="meta" id="meta"></p>
<div class="stats" id="stats"></div>
<div class="steps" id="steps"></div>
<svg id="arena" role="img" aria-label="로봇 위치 재생"></svg>
<div class="bar">
  <button id="play" type="button">재생</button>
  <select id="speed" aria-label="재생 속도"><option value="0.5">0.5배</option><option value="1">1배</option><option value="2" selected>2배</option><option value="4">4배</option></select>
  <input id="scrub" type="range" min="0" value="0" step="1" aria-label="시간">
  <label><input id="trails" type="checkbox" checked> 궤적</label>
  <label><input id="goals" type="checkbox" checked> 목표</label>
</div>
<div class="hud" id="hud"></div>
<div class="legend" id="legend"></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const N = D.n, F = D.frames.length, W = D.arena[0], H = D.arena[1], R = D.diameter / 2;
const COLORS = ['#7F77DD', '#1D9E75', '#D85A30', '#D4537E'];
const NS = 'http://www.w3.org/2000/svg';
const svg = document.getElementById('arena');
const pad = 80;
svg.setAttribute('viewBox', `${-pad} ${-pad} ${W + 2 * pad} ${H + 2 * pad}`);
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); (parent || svg).appendChild(e); return e; };
const Y = y => H - y;

for (let x = 0; x <= W; x += 500) el('line', { x1: x, y1: 0, x2: x, y2: H, stroke: 'var(--grid)', 'stroke-width': 4 });
for (let y = 0; y <= H; y += 500) el('line', { x1: 0, y1: y, x2: W, y2: y, stroke: 'var(--grid)', 'stroke-width': 4 });
el('rect', { x: 0, y: 0, width: W, height: H, fill: 'none', stroke: 'var(--muted)', 'stroke-width': 6 });
const axis = el('text', { x: W, y: H + 60, 'text-anchor': 'end', fill: 'var(--muted)', 'font-size': 44 });
axis.textContent = `${(W / 1000).toFixed(1)} x ${(H / 1000).toFixed(1)} m · 격자 0.5 m · 위쪽이 +y(왼쪽), 오른쪽이 +x(앞)`;

const goalLayer = el('g', {}), trailLayer = el('g', {}), dockLayer = el('g', {}), robotLayer = el('g', {});
const goals = [], trails = [], robots = [];
for (let i = 0; i < N; i++) {
  goals.push(el('circle', { r: R, fill: 'none', 'stroke-width': 6, 'stroke-dasharray': '14 10' }, goalLayer));
  trails.push(el('polyline', { fill: 'none', 'stroke-width': 8, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', opacity: 0.35 }, trailLayer));
  const g = el('g', {}, robotLayer);
  const body = el('circle', { r: R, stroke: 'var(--arena)', 'stroke-width': 6 }, g);
  const nose = el('line', { x1: 0, y1: 0, x2: R, y2: 0, stroke: '#fff', 'stroke-width': 10, 'stroke-linecap': 'round' }, g);
  const label = el('text', { 'text-anchor': 'middle', 'dominant-baseline': 'central', fill: '#fff', 'font-size': 46, 'font-weight': 600 }, g);
  label.textContent = i + 1;
  robots.push({ g, body, nose, label });
}

const s = D.summary;
document.getElementById('title').textContent = `“${D.text}”`;
document.getElementById('meta').textContent = `로봇 ${N}대 · ${s.controller} · ${s.dynamics} · 시드 ${s.seed} · L1 ${D.source === 'rule' ? '규칙 파서' : D.source}`;
const stats = [
  [s.ok ? '성공' : '실패' + (s.error ? `: ${s.error}` : ''), !s.ok],
  [`소요 ${s.seconds}초`],
  [`최종 오차 평균 ${s.final_mean_cm}cm / 최대 ${s.final_max_cm}cm`],
  [`최소 중심 거리 ${s.closest_cm}cm (로봇 지름 ${D.diameter / 10}cm)`],
  [`접촉 ${s.contacts}회`, s.contacts > 0],
].concat(s.warnings.map(w => [`경고: ${w}`, true]));
document.getElementById('stats').innerHTML = stats.map(([t, bad]) => `<span class="${bad ? 'bad' : ''}"><b>${t}</b></span>`).join('');

const stepBox = document.getElementById('steps');
const chips = D.steps.map((st, k) => {
  const b = document.createElement('button');
  b.type = 'button';
  b.innerHTML = `${k + 1}단계 · ` + st.labels.map((l, c) => `<span style="color:${COLORS[c % COLORS.length]}">●</span> ${l}`).join(' / ');
  b.onclick = () => { cur = st.frame; render(); };
  stepBox.appendChild(b);
  return b;
});
document.getElementById('legend').innerHTML =
  `<span><span class="dot" style="background:${COLORS[0]}"></span>단계의 첫 번째 명령</span>` +
  `<span><span class="dot" style="background:${COLORS[1]}"></span>두 번째 명령</span>` +
  `<span><span class="dot" style="background:var(--hold)"></span>이번 단계 명령 없음 (이전 동작 유지)</span>` +
  `<span>점선 원: 목표 위치 · 흰 막대: 로봇 앞쪽 · 굵은 선: 도킹</span>`;

const scrub = document.getElementById('scrub');
scrub.max = F - 1;
let cur = 0, playing = false, last = null;
const stepAt = f => { let k = 0; for (let i = 0; i < D.steps.length; i++) if (D.steps[i].frame <= f) k = i; return k; };
const dockAt = f => { let p = []; for (const [fr, pairs] of D.docks) if (fr <= f) p = pairs; return p; };
const at = (f, i) => {
  const a = Math.min(Math.floor(f), F - 1), b = Math.min(a + 1, F - 1), w = f - a;
  const A = D.frames[a], B = D.frames[b];
  let dh = (B[3 * i + 2] - A[3 * i + 2]) / 1000;
  dh = Math.atan2(Math.sin(dh), Math.cos(dh));
  return [A[3 * i] + w * (B[3 * i] - A[3 * i]), A[3 * i + 1] + w * (B[3 * i + 1] - A[3 * i + 1]), A[3 * i + 2] / 1000 + w * dh];
};

function render() {
  const f = Math.max(0, Math.min(cur, F - 1)), fi = Math.floor(f), k = stepAt(fi), st = D.steps[k];
  const showTrails = document.getElementById('trails').checked, showGoals = document.getElementById('goals').checked;
  const pts = [];
  for (let i = 0; i < N; i++) {
    const [x, y, h] = at(f, i);
    pts.push([x, y]);
    const m = st ? st.member[i] : -1;
    const color = m < 0 ? 'var(--hold)' : COLORS[m % COLORS.length];
    const r = robots[i];
    r.g.setAttribute('transform', `translate(${x} ${Y(y)})`);
    r.body.setAttribute('fill', color);
    r.nose.setAttribute('x2', R * Math.cos(h));
    r.nose.setAttribute('y2', -R * Math.sin(h));
    const goal = st && st.goals[i];
    goals[i].setAttribute('visibility', showGoals && goal ? 'visible' : 'hidden');
    if (goal) { goals[i].setAttribute('cx', goal[0]); goals[i].setAttribute('cy', Y(goal[1])); goals[i].setAttribute('stroke', color); }
    trails[i].setAttribute('visibility', showTrails ? 'visible' : 'hidden');
    if (showTrails) {
      const from = Math.max(0, fi - Math.round(12 / D.dt)), path = [];
      for (let j = from; j <= fi; j += 2) path.push(`${D.frames[j][3 * i]},${Y(D.frames[j][3 * i + 1])}`);
      path.push(`${x},${Y(y)}`);
      trails[i].setAttribute('points', path.join(' '));
      trails[i].setAttribute('stroke', color);
    }
  }
  dockLayer.replaceChildren();
  for (const [a, b] of dockAt(fi)) el('line', { x1: pts[a][0], y1: Y(pts[a][1]), x2: pts[b][0], y2: Y(pts[b][1]), stroke: 'var(--fg)', 'stroke-width': 26, 'stroke-linecap': 'round', opacity: 0.55 }, dockLayer);
  chips.forEach((c, j) => c.classList.toggle('on', j === k));
  scrub.value = fi;
  const [dmin, contacts] = D.hud[fi];
  document.getElementById('hud').textContent =
    `t = ${(f * D.dt).toFixed(1)}초 / ${((F - 1) * D.dt).toFixed(1)}초 · ${D.steps.length ? `${k + 1}/${D.steps.length}단계` : ''} · 현재 최소 중심 거리 ${dmin < 0 ? '-' : (dmin / 10).toFixed(1) + 'cm'} · 누적 접촉 ${contacts}회`;
}

function tick(now) {
  if (playing) {
    if (last !== null) cur += (now - last) / 1000 / D.dt * Number(document.getElementById('speed').value);
    if (cur >= F - 1) { cur = F - 1; setPlaying(false); }
    render();
  }
  last = now;
  requestAnimationFrame(tick);
}
function setPlaying(on) { playing = on; if (on && cur >= F - 1) cur = 0; document.getElementById('play').textContent = on ? '일시정지' : '재생'; }
document.getElementById('play').onclick = () => setPlaying(!playing);
scrub.oninput = () => { cur = Number(scrub.value); render(); };
document.getElementById('trails').onchange = render;
document.getElementById('goals').onchange = render;
document.addEventListener('keydown', e => { if (e.code === 'Space' && e.target.tagName !== 'INPUT') { e.preventDefault(); setPlaying(!playing); } });
render();
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="명령 실행 결과를 HTML 재생 파일로 저장")
    ap.add_argument("command", nargs="+")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--ideal", action="store_true")
    ap.add_argument("--policy", type=Path, default=None)
    ap.add_argument("-o", "--out", type=Path, default=Path("replays/replay.html"))
    args = ap.parse_args()

    actor, limit = None, 0.05
    if args.policy:
        from tiggle_swarm.train import load_actor

        actor, limit = load_actor(args.policy)
    data = record(" ".join(args.command), args.n, args.seed, IDEAL if args.ideal else DomainRand(), actor, limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    args.out.write_text(PAGE.replace("__DATA__", payload), encoding="utf-8")
    s = data["summary"]
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB, {len(data['frames'])} 프레임)")
    print(f"{'성공' if s['ok'] else '실패'} · {s['seconds']}초 · 최종 오차 {s['final_mean_cm']}/{s['final_max_cm']}cm · 최소 거리 {s['closest_cm']}cm · 접촉 {s['contacts']}회")


if __name__ == "__main__":
    main()
