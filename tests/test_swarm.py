"""L2~L4와 추상 시뮬레이터, 학습 경로 테스트."""

import math

import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

from tiggle_nl import CommandParser
from tiggle_nl.config import FleetConfig
from tiggle_nl.schema import TaskSpec
from tiggle_swarm.cbf import CBFConfig, safety_filter
from tiggle_swarm.config import ControlGains, RobotLimits
from tiggle_swarm.control import clamp_speed, nominal_velocity
from tiggle_swarm.formations import half_extents, slot_template
from tiggle_swarm.pipeline import random_start, run_in_sim
from tiggle_swarm.planner import PlanError, PlanExecutor, SwarmState
from tiggle_swarm.policy import K_NEIGHBORS, ResidualActor, build_obs
from tiggle_swarm.sim import IDEAL, DomainRand, SwarmSim
from tiggle_swarm.train import TrainConfig, evaluate, load_actor, sample_scenario, train

LIMITS = RobotLimits()
GAINS = ControlGains()
D_SAFE = 0.15


def fleet(n: int, **kw) -> FleetConfig:
    return FleetConfig(robot_ids=tuple(range(1, n + 1)), **kw)


def spec(*steps: list[dict]) -> TaskSpec:
    return TaskSpec.model_validate({"steps": [{"commands": list(s)} for s in steps]})


def formation(robots, shape, **kw):
    return {"primitive": "formation", "robots": list(robots), "shape": shape, **kw}


def state(points, heading=None) -> SwarmState:
    pos = np.array(points, dtype=float)
    return SwarmState(pos, np.zeros(len(pos)) if heading is None else np.array(heading, dtype=float))


# --- 대형 템플릿 ---

@pytest.mark.parametrize("shape, low", [("line", 2), ("column", 2), ("circle", 3), ("v", 3), ("triangle", 3), ("square", 4), ("grid", 4), ("cluster", 2)])
def test_slots_keep_spacing_for_every_size(shape, low):
    for n in range(low, 16):
        pts = slot_template(shape, n, 0.25)
        assert len(pts) == n
        d = np.linalg.norm(pts[:, None] - pts[None], axis=-1) + np.eye(n) * 9
        assert d.min() >= 0.25 - 1e-9, (shape, n, d.min())


def test_half_extents_follow_heading():
    hx, hy = half_extents("line", 5, 0.25, None)
    assert (hx, hy) == pytest.approx((0.0, 0.5), abs=1e-9)
    hx, hy = half_extents("line", 5, 0.25, 90)
    assert (hx, hy) == pytest.approx((0.5, 0.0), abs=1e-9)


# --- L2 계획기 ---

def kinematic_run(spec_, st, fl, seconds=60.0):
    """측정이 완벽하고 기준을 그대로 따라가는 로봇으로 계획기만 돌린다."""
    ex = PlanExecutor(spec_, st, fl)
    dt, t = LIMITS.dt, 0.0
    pos, heading = st.pos.copy(), st.heading.copy()
    ref = ex.update(t, SwarmState(pos, heading))
    while not ref.done and t < seconds:
        pos, heading = ref.pos.copy(), ref.heading.copy()
        t += dt
        ref = ex.update(t, SwarmState(pos, heading))
    return ex, ref, t


def test_assignment_avoids_crossing():
    fl = fleet(2)
    st = state([[1.0, 0.5], [1.0, 1.5]])  # 2번이 위(왼쪽)
    ex = PlanExecutor(spec([formation([1, 2], "line", spacing_m=0.4, center={"x": 1.0, "y": 1.0})]), st, fl)
    goal = ex.active[-1].motion.goal
    assert goal[0][1] < goal[1][1]  # 아래쪽 로봇이 아래쪽 슬롯으로


def test_formation_is_pushed_inside_arena_when_center_not_given():
    fl = fleet(4)
    st = state([[0.1, 0.1], [0.3, 0.1], [0.1, 0.3], [0.3, 0.3]])
    ex = PlanExecutor(spec([formation([1, 2, 3, 4], "square", spacing_m=0.5)]), st, fl)
    goal = ex.active[-1].motion.goal
    assert goal.min() >= 0.06 - 1e-9


def test_explicit_center_outside_is_error():
    fl = fleet(4)
    st = state([[1, 1], [1.3, 1], [1, 1.3], [1.3, 1.3]])
    with pytest.raises(PlanError, match="경기장"):
        PlanExecutor(spec([formation([1, 2, 3, 4], "square", center={"x": 0.1, "y": 0.1})]), st, fl)


def test_relative_move_off_arena_is_error():
    fl = fleet(2)
    st = state([[2.8, 1.0], [2.8, 1.4]])
    with pytest.raises(PlanError, match="이동 목표"):
        PlanExecutor(spec([[{"primitive": "move", "robots": [1, 2], "mode": "relative", "x": 0.5, "y": 0.0}][0]]), st, fl)


def test_groups_without_center_do_not_overlap():
    fl = fleet(8)
    pos = [[1.0 + 0.2 * (i % 4), 0.7 + 0.25 * (i // 4)] for i in range(8)]
    a = formation([1, 3, 5, 7], "square")
    b = formation([2, 4, 6, 8], "grid")
    ex = PlanExecutor(TaskSpec.model_validate({"steps": [{"commands": [a, b]}]}), state(pos), fl)
    one, two = [m.motion.final() for m in ex._step_active]
    assert np.linalg.norm(one[:, None] - two[None], axis=-1).min() >= fl.min_spacing_m - 1e-9


def test_formation_avoids_robots_that_hold_position():
    fl = fleet(5)
    pos = [[1.0, 1.0], [1.25, 1.0], [1.0, 1.25], [1.1, 1.1], [1.2, 1.2]]
    ex = PlanExecutor(spec([formation([1, 2, 3], "circle", spacing_m=0.3)]), state(pos), fl)
    goal = ex._step_active[0].motion.final()
    assert np.linalg.norm(goal[:, None] - state(pos).pos[3:][None], axis=-1).min() >= fl.min_spacing_m - 1e-9


def test_formation_does_not_trap_other_group():
    """두 무리가 섞여 있을 때 원이 다른 무리 로봇을 안에 가둔 채 닫히면 그 로봇은 빠져나오지 못한다."""
    fl = fleet(10)
    pos = [[1.2 + 0.2 * (i % 5), 0.8 + 0.2 * (i // 5)] for i in range(10)]
    ring, other = [1, 3, 5, 7, 9], [2, 4, 6, 8, 10]
    cmds = [formation(ring, "circle"), formation(other, "v")]
    ex = PlanExecutor(TaskSpec.model_validate({"steps": [{"commands": cmds}]}), state(pos), fl)
    circle = next(a.motion for a in ex._step_active if len(a.motion.indices) == 5 and a.motion.indices[0] == 0).final()
    hull = ConvexHull(circle)
    now = state(pos).pos[[i - 1 for i in other]]
    inside = (now @ hull.equations[:, :2].T + hull.equations[:, 2] <= fl.robot_diameter_m).all(axis=1)
    assert not inside.any()


def test_mixed_groups_split_without_getting_stuck():
    fl = fleet(15)
    parsed = CommandParser(fl, use_llm=False).parse("모여서 1~7번은 원 만들고 나머지는 V자로")
    r = run_in_sim(parsed.spec, fl, random_start(15, fl, 3), rand=DomainRand(), seed=3)
    assert r.error is None and not r.warnings, (r.error, r.warnings)
    assert r.contacts == 0 and r.worst_error < 0.03


def test_move_onto_another_robot_is_error():
    fl = fleet(2)
    st = state([[1.0, 1.0], [1.3, 1.0]])
    with pytest.raises(PlanError, match="겹칩니다"):
        PlanExecutor(spec([{"primitive": "move", "robots": [1], "mode": "relative", "x": 0.3, "y": 0.0}]), st, fl)


def test_explicit_center_on_held_robot_is_error():
    fl = fleet(4)
    st = state([[0.4, 0.4], [0.7, 0.4], [0.4, 0.7], [1.5, 1.0]])
    with pytest.raises(PlanError, match="겹칩니다"):
        PlanExecutor(spec([formation([1, 2, 3], "line", spacing_m=0.3, center={"x": 1.5, "y": 1.0})]), st, fl)


def test_reference_speed_stays_within_requested_speed():
    fl = fleet(3)
    st = state([[0.5, 0.5], [0.5, 1.0], [0.5, 1.5]])
    cmd = {"primitive": "move", "robots": [1, 2, 3], "mode": "relative", "x": 1.0, "y": 0.0, "speed": "slow"}
    ex = PlanExecutor(spec([cmd]), st, fl)
    motion = ex.active[-1].motion
    peak = max(np.linalg.norm(motion.reference(t, st)[1], axis=1).max() for t in np.linspace(0, motion.duration, 200))
    assert peak == pytest.approx(LIMITS.v_slow, rel=0.01)


def test_rotate_group_about_centroid():
    fl = fleet(2)
    st = state([[1.0, 1.0], [1.4, 1.0]])
    cmd = {"primitive": "rotate", "robots": [1, 2], "angle_deg": 90, "pivot": "group_center"}
    _, ref, _ = kinematic_run(spec([cmd]), st, fl)
    assert ref.pos == pytest.approx(np.array([[1.2, 0.8], [1.2, 1.2]]), abs=1e-6)
    assert ref.heading == pytest.approx(np.full(2, math.pi / 2), abs=1e-6)


def test_steps_run_in_order():
    fl = fleet(2)
    st = state([[1.0, 1.0], [1.4, 1.0]])
    a = {"primitive": "move", "robots": [1, 2], "mode": "relative", "x": 0.2, "y": 0.0}
    b = {"primitive": "move", "robots": [1, 2], "mode": "relative", "x": 0.0, "y": 0.3}
    _, ref, _ = kinematic_run(spec([a], [b]), st, fl)
    assert ref.pos == pytest.approx(np.array([[1.2, 1.3], [1.6, 1.3]]), abs=1e-6)


def test_settle_timer_starts_after_slowest_group():
    fl = fleet(2)
    st = state([[0.5, 0.5], [0.5, 1.5]])
    short = {"primitive": "move", "robots": [1], "mode": "relative", "x": 0.05, "y": 0.0}
    long = {"primitive": "move", "robots": [2], "mode": "relative", "x": 1.0, "y": 0.0}
    ex = PlanExecutor(TaskSpec.model_validate({"steps": [{"commands": [short, long]}]}), st, fl)
    t_long = max(a.motion.duration for a in ex._step_active)
    pos, t = st.pos.copy(), 0.0
    while t < t_long + 1.0:  # 2번은 기준보다 20cm 뒤처져 따라간다
        ref = ex.update(t, SwarmState(pos, st.heading))
        pos = ref.pos.copy()
        pos[1, 0] = max(0.5, ref.pos[1, 0] - 0.2)
        t += LIMITS.dt
    assert not ref.done and not ex.warnings


def test_follow_keeps_going_in_next_step():
    fl = fleet(3)
    st = state([[1.0, 1.0], [0.7, 1.0], [0.4, 1.0]])
    follow = {"primitive": "follow", "robots": [2, 3], "leader": 1, "spacing_m": 0.25}
    move = {"primitive": "move", "robots": [1], "mode": "relative", "x": 0.5, "y": 0.0}
    ex = PlanExecutor(spec([follow], [move]), st, fl)
    pos, t = st.pos.copy(), 0.0
    for _ in range(int(40 / LIMITS.dt)):
        ref = ex.update(t, SwarmState(pos, st.heading))
        pos = pos + np.clip(ref.pos - pos, -0.01, 0.01)
        t += LIMITS.dt
    assert ref.done
    assert np.linalg.norm(pos[0] - pos[1]) == pytest.approx(0.25, abs=0.02)
    assert pos[1][0] > 1.2  # 2번이 1번을 따라 앞으로 왔다
    assert pos[2][0] > 0.9


def test_stop_cancels_follow():
    fl = fleet(2)
    st = state([[1.0, 1.0], [0.7, 1.0]])
    follow = {"primitive": "follow", "robots": [2], "leader": 1, "spacing_m": 0.2}
    stop = {"primitive": "stop", "robots": [2]}
    ex, ref, _ = kinematic_run(spec([follow], [stop]), st, fl)
    assert ref.group[1] == -1
    assert isinstance(ex.active[ex.owner[1]].motion.goal, np.ndarray)  # 고정 목표로 바뀌었다


def test_dock_rules():
    fl = fleet(3)
    st = state([[1.0, 1.0], [1.3, 1.0], [1.6, 1.0]])
    dock = {"primitive": "dock", "robots": [1, 2]}
    ex, ref, _ = kinematic_run(spec([dock]), st, fl)
    assert ref.docked[0, 1] and not ref.docked[1, 2]
    assert np.linalg.norm(ref.pos[0] - ref.pos[1]) == pytest.approx(LIMITS.dock_distance_m, abs=1e-6)
    # 도킹한 로봇 하나만 움직이면 안 된다
    ex = PlanExecutor(spec([dock], [{"primitive": "move", "robots": [1], "mode": "relative", "x": 0.1, "y": 0.0}]), st, fl)
    pos, t = st.pos.copy(), 0.0
    with pytest.raises(PlanError, match="함께 지정"):
        for _ in range(4000):
            ref = ex.update(t, SwarmState(pos, st.heading))
            pos, t = ref.pos.copy(), t + LIMITS.dt
    # 둘을 함께 옮기는 건 괜찮고, 분리하면 간격이 벌어지고 제약이 돌아온다
    both = {"primitive": "move", "robots": [1, 2], "mode": "relative", "x": 0.1, "y": 0.0}
    undock = {"primitive": "undock", "robots": [1, 2]}
    _, ref, _ = kinematic_run(spec([dock], [both], [undock]), st, fl)
    assert not ref.docked.any()
    assert np.linalg.norm(ref.pos[0] - ref.pos[1]) >= fl.min_spacing_m


# --- L3 기본 제어 ---

def test_nominal_is_feedforward_when_on_reference():
    p = torch.tensor([[[1.0, 1.0], [1.5, 1.0]]])
    v = torch.tensor([[[0.05, 0.0], [0.0, 0.02]]])
    u = nominal_velocity(p, p.clone(), v, torch.zeros((1, 2), dtype=torch.long), GAINS, LIMITS.v_max)
    assert u.numpy() == pytest.approx(v.numpy())


def test_nominal_saturates_and_keeps_direction():
    p = torch.zeros((1, 1, 2))
    ref = torch.tensor([[[3.0, 4.0]]])
    u = nominal_velocity(p, ref, torch.zeros((1, 1, 2)), torch.zeros((1, 1), dtype=torch.long), GAINS, LIMITS.v_max)
    assert float(u.norm()) == pytest.approx(LIMITS.v_max)
    assert float(u[0, 0, 0] / u[0, 0, 1]) == pytest.approx(0.75)


def test_consensus_only_within_group():
    p = torch.tensor([[[1.0, 1.0], [1.2, 1.0], [1.4, 1.0]]])
    ref = p.clone()
    ref[0, 1, 0] += 0.05  # 2번만 뒤처짐
    group = torch.tensor([[0, 0, 1]])
    u = nominal_velocity(p, ref, torch.zeros_like(p), group, GAINS, LIMITS.v_max)
    assert float(u[0, 2].norm()) == pytest.approx(0.0, abs=1e-6)  # 다른 무리는 영향 없음
    assert float(u[0, 0, 0]) < 0  # 같은 무리의 1번은 2번 쪽 오차를 맞추려 뒤로 물러난다


# --- L4 안전 필터 ---

def exact_qp(p, u_nom, D, gamma):
    n = len(p)
    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            d = p[i] - p[j]
            dist = np.linalg.norm(d)
            normal = d / dist
            b = -gamma * (dist**2 - D**2) / (2 * dist)
            def f(x, i=i, j=j, normal=normal, b=b):
                u = x.reshape(n, 2)
                return normal @ (u[i] - u[j]) - b
            cons.append({"type": "ineq", "fun": f})
    x0 = u_nom.ravel()
    res = minimize(lambda x: ((x - x0) ** 2).sum(), x0, constraints=cons, method="SLSQP", options={"maxiter": 500, "ftol": 1e-10})
    return res.x.reshape(n, 2)


@pytest.mark.parametrize("seed", range(5))
def test_filter_matches_exact_qp(seed):
    rng = np.random.default_rng(seed)
    n = 6
    p = rng.uniform(0, 0.6, (n, 2))
    while (np.linalg.norm(p[:, None] - p[None], axis=-1) + np.eye(n) * 9).min() < 0.16:
        p = rng.uniform(0, 0.7, (n, 2))
    u = rng.uniform(-0.05, 0.05, (n, 2))
    cfg = CBFConfig(iters=2000, yield_gain=0.0)
    got = safety_filter(torch.tensor(p, dtype=torch.float32)[None], torch.tensor(u, dtype=torch.float32)[None], D_SAFE, 1.0, cfg).u[0].numpy()
    want = exact_qp(p, u, D_SAFE, cfg.gamma)
    assert got == pytest.approx(want, abs=2e-3)


def test_filter_leaves_safe_commands_alone():
    p = torch.tensor([[[0.5, 0.5], [2.0, 1.5]]])
    u = torch.tensor([[[0.05, 0.0], [0.0, -0.05]]])
    assert safety_filter(p, u, D_SAFE, LIMITS.v_max).u.numpy() == pytest.approx(u.numpy(), abs=1e-6)


def test_filter_respects_speed_limit():
    p = torch.tensor([[[1.0, 1.0], [1.16, 1.0], [1.08, 1.14]]])
    u = clamp_speed(torch.tensor([[[0.5, 0.0], [-0.5, 0.0], [0.0, -0.5]]]), LIMITS.v_max)
    assert float(safety_filter(p, u, D_SAFE, LIMITS.v_max).u.norm(dim=-1).max()) <= LIMITS.v_max + 1e-6


def test_filter_handles_batch_independently():
    p = torch.tensor([[[1.0, 1.0], [1.2, 1.0]], [[1.0, 1.0], [2.0, 1.0]]])
    u = torch.tensor([[[0.1, 0.0], [-0.1, 0.0]], [[0.1, 0.0], [-0.1, 0.0]]])
    out = safety_filter(p, u, D_SAFE, LIMITS.v_max).u
    single = safety_filter(p[:1], u[:1], D_SAFE, LIMITS.v_max).u
    assert out[0].numpy() == pytest.approx(single[0].numpy(), abs=1e-6)
    assert out[1].numpy() == pytest.approx(u[1].numpy(), abs=1e-6)


def test_docked_pair_is_exempt():
    p = torch.tensor([[[1.0, 1.0], [1.12, 1.0]]])
    u = torch.tensor([[[0.02, 0.0], [0.02, 0.0]]])
    d = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    assert safety_filter(p, u, d, LIMITS.v_max).u.numpy() == pytest.approx(u.numpy(), abs=1e-6)


def single_integrator_run(start, goal, seconds=20.0):
    p = torch.tensor(start, dtype=torch.float32)[None]
    ref = torch.tensor(goal, dtype=torch.float32)[None]
    lam, closest = None, float("inf")
    n = p.shape[1]
    for _ in range(int(seconds / LIMITS.dt)):
        u = clamp_speed(1.2 * (ref - p), LIMITS.v_max)
        s = safety_filter(p, u, D_SAFE, LIMITS.v_max, CBFConfig(iters=100), lam)
        lam = s.lam
        p = p + s.u * LIMITS.dt
        d = torch.cdist(p, p) + torch.eye(n) * 9
        closest = min(closest, float(d.min()))
    return p, ref, closest


def test_head_on_robots_pass_each_other():
    p, ref, closest = single_integrator_run([[1.0, 1.0], [2.0, 1.0]], [[2.0, 1.0], [1.0, 1.0]])
    assert closest >= D_SAFE - 0.005
    assert float((p - ref).norm(dim=-1).max()) < 0.03


def test_four_way_crossing_clears():
    start = [[1.0, 1.0], [2.0, 1.0], [1.5, 0.5], [1.5, 1.5]]
    goal = [[2.0, 1.0], [1.0, 1.0], [1.5, 1.5], [1.5, 0.5]]
    p, ref, closest = single_integrator_run(start, goal, seconds=30.0)
    assert closest >= D_SAFE - 0.005
    assert float((p - ref).norm(dim=-1).max()) < 0.03


# --- 시뮬레이터 ---

def make_sim(n=1, rand=IDEAL, **kw):
    fl = fleet(n, **kw)
    return SwarmSim(1, n, fl, LIMITS, rand)


def test_sim_reaches_commanded_speed():
    sim = make_sim()
    sim.reset(torch.tensor([[[1.0, 1.0]]]), torch.zeros((1, 1)))
    for _ in range(60):
        sim.step(torch.tensor([[[0.1, 0.0]]]), torch.zeros((1, 1)))
    assert float(sim.vel[0, 0, 0]) == pytest.approx(0.1, abs=1e-3)


def test_sim_applies_axis_gains_in_body_frame():
    rand = DomainRand(gain_forward=(1.0, 1.0), gain_strafe=(0.5, 0.5), tau=(0.05, 0.05), deadband=(0, 0), cmd_delay=(0, 0), obs_delay=(0, 0), pos_noise=(0, 0), drop=(0, 0))
    sim = make_sim(rand=rand)
    sim.reset(torch.tensor([[[1.0, 1.0]]]), torch.full((1, 1), math.pi / 2))  # 왼쪽을 보고 있으면
    for _ in range(40):
        sim.step(torch.tensor([[[0.1, 0.0]]]), torch.zeros((1, 1)))  # 경기장 +x는 몸체 -y(횡이동)
    assert float(sim.vel[0, 0, 0]) == pytest.approx(0.05, abs=2e-3)


def test_sim_command_delay():
    rand = DomainRand(gain_forward=(1, 1), gain_strafe=(1, 1), tau=(0.05, 0.05), deadband=(0, 0), cmd_delay=(3, 3), obs_delay=(0, 0), pos_noise=(0, 0), drop=(0, 0))
    sim = make_sim(rand=rand)
    sim.reset(torch.tensor([[[1.0, 1.0]]]), torch.zeros((1, 1)))
    u, w = torch.tensor([[[0.1, 0.0]]]), torch.zeros((1, 1))
    for _ in range(3):
        sim.step(u, w)
    assert float(sim.vel.norm()) == 0.0
    sim.step(u, w)
    assert float(sim.vel.norm()) > 0.0


def test_sim_contact_stops_overlap():
    sim = make_sim(2)
    sim.reset(torch.tensor([[[1.0, 1.0], [1.3, 1.0]]]), torch.zeros((1, 2)))
    u = torch.tensor([[[0.1, 0.0], [-0.1, 0.0]]])
    worst, touched = 9.0, False
    for _ in range(80):
        sim.step(u, torch.zeros((1, 2)))
        worst = min(worst, float((sim.pos[0, 0] - sim.pos[0, 1]).norm()))
        touched |= bool(sim.contacts.sum() > 0)
    assert touched and worst > 0.11


# --- 정책 ---

def random_obs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand((2, n, 2), generator=g) * 2
    return dict(
        p=p, theta=torch.rand((2, n), generator=g) * 6, v_hat=torch.rand((2, n, 2), generator=g) * 0.1,
        p_ref=p + 0.1 * torch.rand((2, n, 2), generator=g), v_ref=torch.rand((2, n, 2), generator=g) * 0.1,
        u_nom=torch.rand((2, n, 2), generator=g) * 0.1, last_cmd=torch.rand((2, n, 2), generator=g) * 0.1,
        group=torch.zeros((2, n), dtype=torch.long),
    )


def obs_of(d):
    return build_obs(d["p"], d["theta"], d["v_hat"], d["p_ref"], d["v_ref"], d["u_nom"], d["last_cmd"], d["group"], D_SAFE, LIMITS.v_max, 10.0)


@pytest.mark.parametrize("n", [1, 2, 7, 15])
def test_policy_handles_any_robot_count(n):
    o = obs_of(random_obs(n))
    assert o.nbr.shape == (2, n, K_NEIGHBORS, 8)
    assert ResidualActor().mean(o).shape == (2, n, 2)


def test_policy_is_permutation_equivariant():
    n = 7
    d = random_obs(n)
    actor = ResidualActor()
    for layer in actor.head[-1:]:
        torch.nn.init.orthogonal_(layer.weight, gain=1.0)  # 차이가 보이도록 초기 축소를 푼다
    base = actor.mean(obs_of(d))
    perm = torch.randperm(n)
    shuffled = {k: v[:, perm] for k, v in d.items()}
    moved = actor.mean(obs_of(shuffled))
    assert moved.detach().numpy() == pytest.approx(base[:, perm].detach().numpy(), abs=1e-5)


def test_policy_starts_near_zero():
    assert float(ResidualActor().mean(obs_of(random_obs(8))).detach().abs().max()) < 0.05


# --- 학습 경로와 전체 실행 ---

def test_scenarios_respect_arena_and_timing():
    fl = fleet(8)
    s = sample_scenario(np.random.default_rng(3), 16, 8, fl, LIMITS)
    goal = s.goal
    assert float(goal.min()) >= 0.06 - 1e-6 and float(goal[..., 0].max()) <= 2.94 + 1e-6 and float(goal[..., 1].max()) <= 1.94 + 1e-6
    assert float(s.duration.max()) * 1 <= 20.0 + 1e-6


def test_train_and_evaluate_smoke(tmp_path):
    cfg = TrainConfig(stages=((3, 1),), batch=4, minibatch=256, epochs=1)
    out = tmp_path / "actor.pt"
    train(cfg, out, log_every=100)
    actor, limit = load_actor(out)
    result = evaluate(actor, limit, (3,), batch=4, episodes=1)
    assert 0 <= result[3].success <= 1 and result[3].min_dist > 0


def e2e(spec_, n, seed=0, rand=IDEAL):
    fl = fleet(n)
    return run_in_sim(spec_, fl, random_start(n, fl, seed), rand=rand, seed=seed)


@pytest.mark.parametrize("shape", ["line", "column", "circle", "v", "triangle", "square", "grid"])
def test_ideal_robots_form_each_shape(shape):
    r = e2e(spec([formation(range(1, 7), shape)]), 6)
    assert r.ok, r
    assert r.worst_error < LIMITS.pos_tolerance_m
    assert r.contacts == 0 and r.min_distance >= 0.12


def test_fifteen_robots_with_randomized_dynamics():
    r = e2e(spec([formation(range(1, 16), "circle")]), 15, rand=DomainRand())
    assert r.error is None
    assert r.contacts == 0
    assert r.worst_error < 0.06


def test_text_to_motion():
    fl = fleet(8)
    parsed = CommandParser(fl, use_llm=False).parse("1~3번은 원 만들고 나머지는 V자로")
    assert parsed.ok, parsed.errors
    r = run_in_sim(parsed.spec, fl, random_start(8, fl, 1), rand=IDEAL)
    assert r.ok, r
    assert r.worst_error < LIMITS.pos_tolerance_m


def test_dock_then_undock_in_sim():
    r = e2e(spec([{"primitive": "dock", "robots": [1, 2, 3]}], [{"primitive": "undock", "robots": [1, 2, 3]}]), 3)
    assert r.ok, r
