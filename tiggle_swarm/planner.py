"""L2 계획기: TaskSpec을 로봇별 기준 궤적으로 바꾼다.

- 대형 슬롯은 헝가리안 배정(거리 제곱합 최소)으로 나눈다.
- 무리 안의 로봇은 같은 시간표로 직선 이동하며 동시에 도착한다 (CAPT, Turpin et al. 2014).
  시작·도착 간격이 로봇 반경의 2√2배 이상이면 경로끼리 충돌하지 않는다. 남는 위험은 L4가 막는다.
- 명령받지 않은 로봇은 이전 동작을 유지한다. follow는 다른 명령을 받을 때까지 계속된다.

경기장 이탈이나 도킹 상태 위반처럼 현재 위치를 알아야 판단되는 문제는 여기서 PlanError로 알린다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, QhullError

from tiggle_nl.config import FleetConfig
from tiggle_nl.schema import (
    Command, Dock, Follow, Formation, Gather, Move, Rotate, Stop, TaskSpec, Undock,
)

from .config import RobotLimits
from .formations import rotate, slot_template


class PlanError(Exception):
    """현재 상태에서 실행할 수 없는 명령."""


@dataclass
class SwarmState:
    pos: np.ndarray  # (N, 2) m
    heading: np.ndarray  # (N,) rad


def wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _smooth(s: float) -> tuple[float, float]:
    """0→1로 부드럽게 가는 진행률과 그 미분. 미분의 최댓값은 1.5."""
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s), 6 * s * (1 - s)


@dataclass
class Trajectory:
    """무리 전체가 같은 진행률로 움직이는 궤적. pivot이 있으면 그 점을 중심으로 회전한다."""

    indices: np.ndarray
    start: np.ndarray
    goal: np.ndarray
    heading0: np.ndarray
    dheading: np.ndarray
    duration: float
    pivot: Optional[np.ndarray] = None
    angle: float = 0.0
    coupled: bool = True

    def final(self) -> np.ndarray:
        if self.pivot is None:
            return self.goal
        return self.pivot + rotate(self.start - self.pivot, self.angle)

    def reference(self, t: float, state: SwarmState):
        if self.duration > 0:
            beta, dbeta = _smooth(t / self.duration)
            dbeta = dbeta / self.duration if t < self.duration else 0.0
        else:
            beta, dbeta = 1.0, 0.0
        heading = self.heading0 + beta * self.dheading
        omega = dbeta * self.dheading
        if self.pivot is None:
            return self.start + beta * (self.goal - self.start), dbeta * (self.goal - self.start), heading, omega
        arm = rotate(self.start - self.pivot, beta * self.angle)
        vel = dbeta * self.angle * np.column_stack([-arm[:, 1], arm[:, 0]])
        return self.pivot + arm, vel, heading, omega


@dataclass
class Phased:
    """궤적 여러 개를 이어 붙인 동작 (도킹의 접근 + 결합)."""

    phases: list[Trajectory]
    coupled: bool = True

    @property
    def indices(self) -> np.ndarray:
        return self.phases[0].indices

    @property
    def duration(self) -> float:
        return sum(p.duration for p in self.phases)

    def final(self) -> np.ndarray:
        return self.phases[-1].final()

    def reference(self, t: float, state: SwarmState):
        for p in self.phases[:-1]:
            if t < p.duration:
                return p.reference(t, state)
            t -= p.duration
        return self.phases[-1].reference(t, state)


@dataclass
class FollowChain:
    """각 로봇이 앞 로봇 뒤 spacing 거리의 점을 쫓는다. 끝나는 시각이 없다."""

    indices: np.ndarray  # 사슬 순서 (leader 제외)
    leader: int
    spacing: float
    duration: float = 0.0
    coupled: bool = False

    def reference(self, t: float, state: SwarmState):
        pos, n = state.pos, len(self.indices)
        ref = np.zeros((n, 2))
        prev = pos[self.leader]
        for k, i in enumerate(self.indices):
            d = pos[i] - prev
            r = float(np.hypot(*d))
            ref[k] = prev + (d / r if r > 1e-6 else np.array([-1.0, 0.0])) * self.spacing
            prev = pos[i]
        return ref, np.zeros((n, 2)), state.heading[self.indices], np.zeros(n)


Motion = Union[Trajectory, Phased, FollowChain]


@dataclass
class _Active:
    motion: Motion
    t0: float
    group: int


@dataclass
class Reference:
    pos: np.ndarray  # (N, 2)
    vel: np.ndarray  # (N, 2)
    heading: np.ndarray  # (N,)
    omega: np.ndarray  # (N,)
    group: np.ndarray  # (N,) 같이 움직이는 무리 번호. 묶이지 않은 로봇은 -1
    docked: np.ndarray  # (N, N) bool. L4가 이 쌍에는 거리 제약을 걸지 않는다
    step: int
    done: bool


class PlanExecutor:
    """TaskSpec을 단계별로 실행하며 매 주기 기준값을 낸다. 위치는 측정값을 넣는다."""

    def __init__(self, spec: TaskSpec, state: SwarmState, fleet: FleetConfig, limits: RobotLimits = RobotLimits()):
        self.spec, self.fleet, self.limits = spec, fleet, limits
        self.index = {rid: i for i, rid in enumerate(fleet.robot_ids)}
        self.n = len(fleet.robot_ids)
        self.docked: set[tuple[int, int]] = set()
        self.warnings: list[str] = []
        self.active: list[_Active] = []
        self.owner = np.zeros(self.n, dtype=int)
        self._next_group = 0
        self.step = -1
        self.done = False
        self._step_active: list[_Active] = []
        self._dock_later: list[tuple[float, list[tuple[int, int]]]] = []
        self._undock_at_end: list[tuple[int, int]] = []

        everyone = np.arange(self.n)
        self.active.append(_Active(self._hold(everyone, state), 0.0, -1))
        self._begin_next_step(state, 0.0)

    # --- 매 주기 실행 ---

    def update(self, t: float, state: SwarmState) -> Reference:
        ref = self._compute(t, state)
        while not self.done and self._step_finished(t, state, ref):
            self._finish_step()
            self._begin_next_step(state, t)
            ref = self._compute(t, state)
        for when, pairs in list(self._dock_later):
            if t >= when:
                self.docked.update(pairs)
                self._dock_later.remove((when, pairs))
                ref = self._compute(t, state)
        return ref

    def _compute(self, t: float, state: SwarmState) -> Reference:
        pos, vel = np.zeros((self.n, 2)), np.zeros((self.n, 2))
        heading, omega, group = np.zeros(self.n), np.zeros(self.n), -np.ones(self.n, dtype=int)
        for a_id, a in enumerate(self.active):
            mine = self.owner[a.motion.indices] == a_id
            if not mine.any():
                continue
            p, v, h, w = a.motion.reference(t - a.t0, state)
            idx = a.motion.indices[mine]
            pos[idx], vel[idx], heading[idx], omega[idx] = p[mine], v[mine], h[mine], w[mine]
            group[idx] = a.group if a.motion.coupled else -1
        docked = np.zeros((self.n, self.n), dtype=bool)
        for i, j in self.docked:
            docked[i, j] = docked[j, i] = True
        return Reference(pos, vel, heading, omega, group, docked, self.step, self.done)

    def _step_finished(self, t: float, state: SwarmState, ref: Reference) -> bool:
        # 대기 시간은 가장 늦게 끝난 궤적부터 잰다. 먼저 끝난 무리 기준으로 재면 다른 무리가 도착하기 전에 끊긴다
        since_done = float("inf")
        for a in self._step_active:
            late = t - a.t0 - a.motion.duration
            if late < 0:
                return False
            since_done = min(since_done, late)
        moving = np.concatenate([a.motion.indices for a in self._step_active]) if self._step_active else []
        err = np.linalg.norm(state.pos[moving] - ref.pos[moving], axis=1) if len(moving) else np.zeros(1)
        if float(err.max()) <= self.limits.pos_tolerance_m:
            return True
        if since_done >= self.limits.settle_timeout_s:
            self.warnings.append(f"{self.step + 1}단계: 오차 {float(err.max()):.3f} m가 남은 채로 다음 단계로 넘어갑니다")
            return True
        return False

    def _finish_step(self) -> None:
        self.docked.difference_update(self._undock_at_end)
        self._undock_at_end = []
        for _, pairs in self._dock_later:  # 시간 전에 끝난 도킹도 빠뜨리지 않는다
            self.docked.update(pairs)
        self._dock_later = []

    def _begin_next_step(self, state: SwarmState, t: float) -> None:
        self.step += 1
        if self.step >= len(self.spec.steps):
            self.done = True
            self._step_active = []
            return
        self._step_active = []
        cmds = self.spec.steps[self.step].commands
        commanded = {self.index[r] for cmd in cmds for r in cmd.robots}
        following = {
            int(i) for a_id, a in enumerate(self.active) if isinstance(a.motion, FollowChain)
            for i in a.motion.indices if self.owner[i] == a_id
        }
        held = [i for i in range(self.n) if i not in commanded and i not in following]
        self._occupied = state.pos[held].reshape(-1, 2)
        ends: dict[int, np.ndarray] = {i: state.pos[i] for i in held}
        moved: set[int] = set()

        # 위치가 정해진 명령을 먼저 계획해서, 위치를 정하지 않은 대형이 그 자리를 피하게 한다
        for cmd in sorted(cmds, key=lambda c: isinstance(c, (Formation, Gather)) and c.center is None):
            idx = np.array([self.index[r] for r in cmd.robots])
            self._check_docking(cmd, idx)
            motion = self._motion_for(cmd, idx, state, t)
            group = self._next_group
            self._next_group += 1
            active = _Active(motion, t, group)
            self.active.append(active)
            self.owner[idx] = len(self.active) - 1
            self._step_active.append(active)
            if not isinstance(motion, FollowChain):
                final = motion.final()
                ends.update({int(i): final[k] for k, i in enumerate(motion.indices)})
                self._occupied = np.vstack([self._occupied, final])
                if not isinstance(cmd, Stop):
                    moved.update(int(i) for i in motion.indices)
        self._check_destinations(ends, moved)

    # --- 명령 -> 궤적 ---

    def _motion_for(self, cmd: Command, idx: np.ndarray, state: SwarmState, t: float) -> Motion:
        if isinstance(cmd, Stop):
            return self._hold(idx, state)
        if isinstance(cmd, Move):
            return self._move(cmd, idx, state)
        if isinstance(cmd, Rotate):
            return self._rotate(cmd, idx, state)
        if isinstance(cmd, Formation):
            return self._formation(cmd, idx, state)
        if isinstance(cmd, Gather):
            return self._gather(cmd, idx, state)
        if isinstance(cmd, Follow):
            return self._follow(cmd, idx, state)
        if isinstance(cmd, Dock):
            return self._dock(idx, state, t)
        if isinstance(cmd, Undock):
            return self._undock(idx, state)
        raise PlanError(f"처리할 수 없는 명령: {cmd.primitive}")

    def _hold(self, idx: np.ndarray, state: SwarmState) -> Trajectory:
        p, h = state.pos[idx], state.heading[idx]
        return Trajectory(idx, p.copy(), p.copy(), h.copy(), np.zeros(len(idx)), 0.0, coupled=False)

    def _time(self, dist: float, speed: float, dheading: float = 0.0) -> float:
        """진행률 미분의 최댓값이 1.5이므로 최고 속도가 speed가 되도록 1.5를 곱한다."""
        return 1.5 * max(dist / speed, abs(dheading) / self.limits.omega_ref)

    def _straight(self, idx, state, goal, speed, dheading=None) -> Trajectory:
        start, h0 = state.pos[idx], state.heading[idx]
        dh = np.zeros(len(idx)) if dheading is None else dheading
        dist = float(np.linalg.norm(goal - start, axis=1).max())
        return Trajectory(idx, start.copy(), goal, h0.copy(), dh, self._time(dist, speed, float(np.abs(dh).max())))

    def _move(self, cmd: Move, idx, state) -> Trajectory:
        start = state.pos[idx]
        offset = np.array([cmd.x, cmd.y]) - (0 if cmd.mode == "relative" else start.mean(axis=0))
        goal = start + offset
        self._require_inside(goal, "이동 목표")
        return self._straight(idx, state, goal, self.limits.v_ref(cmd.speed))

    def _rotate(self, cmd: Rotate, idx, state) -> Trajectory:
        start, h0 = state.pos[idx], state.heading[idx]
        angle = math.radians(cmd.angle_deg)
        dh = np.full(len(idx), angle)
        if cmd.pivot == "self":
            return Trajectory(idx, start.copy(), start.copy(), h0.copy(), dh, self._time(0.0, 1.0, angle))
        pivot = start.mean(axis=0)
        reach = float(np.linalg.norm(start - pivot, axis=1).max())
        for beta in np.linspace(0, 1, 25):
            self._require_inside(pivot + rotate(start - pivot, beta * angle), "회전 경로")
        duration = self._time(reach * abs(angle), self.limits.v_ref(cmd.speed), angle)
        return Trajectory(idx, start.copy(), start.copy(), h0.copy(), dh, duration, pivot=pivot, angle=angle)

    def _assign(self, start: np.ndarray, slots: np.ndarray) -> np.ndarray:
        cost = ((start[:, None] - slots[None]) ** 2).sum(axis=-1)
        _, col = linear_sum_assignment(cost)
        return slots[col]

    def _formation(self, cmd: Formation, idx, state) -> Trajectory:
        spacing = cmd.spacing_m if cmd.spacing_m is not None else self.fleet.default_spacing_m
        heading = math.radians(cmd.heading_deg or 0.0)
        tpl = rotate(slot_template(cmd.shape, len(idx), spacing, cmd.radius_m), heading)
        center = None if cmd.center is None else np.array([cmd.center.x, cmd.center.y])
        slots = self._place(tpl, center, state.pos[idx].mean(axis=0), avoid=True, others=self._others(idx, state))
        dh = None if cmd.heading_deg is None else wrap(heading - state.heading[idx])
        return self._straight(idx, state, self._assign(state.pos[idx], slots), self.limits.v_ref(cmd.speed), dh)

    def _gather(self, cmd: Gather, idx, state) -> Trajectory:
        tpl = slot_template("cluster", len(idx), self.fleet.min_spacing_m * 1.05)
        center = None if cmd.center is None else np.array([cmd.center.x, cmd.center.y])
        slots = self._place(tpl, center, state.pos[idx].mean(axis=0), avoid=True, others=self._others(idx, state))
        return self._straight(idx, state, self._assign(state.pos[idx], slots), self.limits.v_ref(cmd.speed))

    def _follow(self, cmd: Follow, idx, state) -> FollowChain:
        leader = self.index[cmd.leader]
        order, rest, prev = [], list(idx), state.pos[leader]
        while rest:
            k = int(np.argmin([np.linalg.norm(state.pos[i] - prev) for i in rest]))
            order.append(rest.pop(k))
            prev = state.pos[order[-1]]
        spacing = cmd.spacing_m if cmd.spacing_m is not None else self.fleet.default_spacing_m
        return FollowChain(np.array(order), leader, spacing)

    def _dock(self, idx, state, t: float) -> Phased:
        """cmd.robots 순서대로 일렬 접근 후 천천히 붙는다. 결합 단계부터 해당 쌍의 거리 제약을 푼다."""
        n, start = len(idx), state.pos[idx]
        axis = start[-1] - start[0]
        norm = float(np.hypot(*axis))
        axis = axis / norm if norm > 1e-6 else np.array([1.0, 0.0])
        psi = math.atan2(axis[1], axis[0])
        offset = (np.arange(n) - (n - 1) / 2)[:, None] * axis

        ready = self._place(offset * self.fleet.min_spacing_m, None, start.mean(axis=0))
        approach = self._straight(idx, state, ready, self.limits.v_ref("normal"), wrap(psi - state.heading[idx]))
        final = ready.mean(axis=0) + offset * self.limits.dock_distance_m
        close = Trajectory(
            idx, ready, final, state.heading[idx] + approach.dheading, np.zeros(n),
            self._time(float(np.linalg.norm(final - ready, axis=1).max()), self.limits.v_slow),
        )
        pairs = [self._pair(int(a), int(b)) for a, b in zip(idx[:-1], idx[1:])]
        self._dock_later.append((t + approach.duration, pairs))
        return Phased([approach, close])

    def _undock(self, idx, state) -> Trajectory:
        chosen = set(int(i) for i in idx)
        pairs = [p for p in self.docked if p[0] in chosen or p[1] in chosen]
        if not pairs:
            self.warnings.append(f"{self.step + 1}단계: 도킹된 로봇이 없어 분리를 건너뜁니다")
            return self._hold(idx, state)
        start = state.pos[idx]
        center = start.mean(axis=0)
        scale = self.fleet.min_spacing_m * 1.05 / self.limits.dock_distance_m
        goal = self._place((start - center) * scale, None, center)
        self._undock_at_end += pairs
        return self._straight(idx, state, goal, self.limits.v_slow)

    # --- 검사 ---

    @staticmethod
    def _pair(i: int, j: int) -> tuple[int, int]:
        return (i, j) if i < j else (j, i)

    def _margin(self) -> float:
        return self.fleet.robot_diameter_m / 2

    def _require_inside(self, points: np.ndarray, what: str) -> None:
        (x0, x1), (y0, y1) = self.fleet.arena_x, self.fleet.arena_y
        m = self._margin()
        inside = (points[:, 0] >= x0 + m) & (points[:, 0] <= x1 - m) & (points[:, 1] >= y0 + m) & (points[:, 1] <= y1 - m)
        if not inside.all():
            raise PlanError(f"{self.step + 1}단계: {what}가 경기장을 벗어납니다")

    def _others(self, idx: np.ndarray, state: SwarmState) -> np.ndarray:
        mask = np.ones(self.n, dtype=bool)
        mask[idx] = False
        return state.pos[mask]

    def _encloses(self, tpl: np.ndarray, centers: np.ndarray, others: np.ndarray) -> np.ndarray:
        """후보 중심마다, 대형 테두리 안(로봇 지름만큼 여유)에 다른 로봇이 지금 있는지.

        안에 있던 로봇은 대형이 닫히면 빠져나오지 못한다. 일렬 대형처럼 면적이 없으면 검사하지 않는다.
        """
        try:
            hull = ConvexHull(tpl)
        except (QhullError, ValueError):
            return np.zeros(len(centers), dtype=bool)
        normal, offset = hull.equations[:, :2], hull.equations[:, 2]
        rel = others[None] - centers[:, None]
        inside = (rel @ normal.T + offset <= self.fleet.robot_diameter_m).all(axis=-1)
        return inside.any(axis=-1)

    def _place(
        self,
        tpl: np.ndarray,
        center: Optional[np.ndarray],
        fallback: np.ndarray,
        avoid: bool = False,
        others: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """사용자가 위치를 지정했으면 그대로, 아니면 현재 위치에 두되 경기장 안으로 밀어 넣는다.

        avoid면 이미 자리가 정해진 로봇과 겹치지 않고, 다른 로봇(others)을 안에 가두지 않는 곳 중
        현재 위치에서 가장 가까운 곳을 찾는다. 가두지 않는 자리가 없으면 겹침만 피한다.
        """
        if center is not None:
            slots = tpl + center
            self._require_inside(slots, "대형 위치")
            return slots
        (x0, x1), (y0, y1) = self.fleet.arena_x, self.fleet.arena_y
        m = self._margin()
        low = np.array([x0 + m, y0 + m]) - tpl.min(axis=0)
        high = np.array([x1 - m, y1 - m]) - tpl.max(axis=0)
        if (low > high).any():
            raise PlanError(f"{self.step + 1}단계: 대형이 경기장보다 큽니다")
        first = np.clip(fallback, low, high)
        if not avoid:
            return tpl + first
        xs = np.arange(low[0], high[0] + 1e-9, 0.05)
        ys = np.arange(low[1], high[1] + 1e-9, 0.05)
        grid = np.vstack([first, np.array(np.meshgrid(xs, ys)).reshape(2, -1).T])
        ok = np.ones(len(grid), dtype=bool)
        if len(self._occupied):
            slots = grid[:, None] + tpl
            ok = np.linalg.norm(slots[:, :, None] - self._occupied[None, None], axis=-1).min(axis=(1, 2)) >= self.fleet.min_spacing_m
        if others is not None and len(others):
            open_ = ok & ~self._encloses(tpl, grid, others)
            if open_.any():
                ok = open_
        free = np.flatnonzero(ok)
        if len(free) == 0:
            raise PlanError(f"{self.step + 1}단계: 다른 로봇과 겹치지 않게 대형을 놓을 자리가 없습니다")
        return tpl + grid[free[np.argmin(np.linalg.norm(grid[free] - first, axis=1))]]

    def _check_destinations(self, ends: dict[int, np.ndarray], moved: set[int]) -> None:
        """이번 단계에 움직이는 로봇의 도착 위치가 겹치면 L4가 끝내 막아서 도착하지 못한다."""
        safe = self.fleet.robot_diameter_m + self.limits.safety_margin_m
        allowed = set(self.docked) | {p for _, pairs in self._dock_later for p in pairs}
        keys = sorted(ends)
        for a, i in enumerate(keys):
            for j in keys[a + 1:]:
                if i not in moved and j not in moved:
                    continue
                if self._pair(i, j) not in allowed and np.linalg.norm(ends[i] - ends[j]) < safe - 1e-6:
                    ids = self.fleet.robot_ids
                    raise PlanError(f"{self.step + 1}단계: {ids[i]}번과 {ids[j]}번의 도착 위치가 겹칩니다")

    def _check_docking(self, cmd: Command, idx: np.ndarray) -> None:
        if isinstance(cmd, (Stop, Undock)):
            return
        chosen = set(int(i) for i in idx)
        ids = self.fleet.robot_ids
        reshapes = isinstance(cmd, (Formation, Gather, Follow, Dock)) or (isinstance(cmd, Rotate) and cmd.pivot == "self")
        for i, j in self.docked:
            if (i in chosen) != (j in chosen):
                raise PlanError(f"{ids[i]}번과 {ids[j]}번은 도킹되어 있어 함께 지정해야 합니다")
            if i in chosen and reshapes:
                raise PlanError(f"{ids[i]}번과 {ids[j]}번이 도킹 상태입니다. 먼저 분리하세요")
