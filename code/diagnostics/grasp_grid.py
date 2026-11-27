import os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'replay_test.py')).read().split("env = UmiSimEnv(build_scene(")[0]
exec(compile(SRC, 'replay_test.py', 'exec'), globals())
import umi_sim.env as envmod
envmod.SQUEEZE_BELOW = 0.06
L = 4

env = UmiSimEnv(build_scene(os.path.expanduser('~/umi_mujoco/sim/mujoco_menagerie')))
ik = PoseIK(env)
m = env.model
g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'cup_geom')
b = env.cup_id
m0 = float(m.body_mass[b]); I0 = m.body_inertia[b].copy()
print('cup geom: condim %d, friction %s, mass %.3f' % (m.geom_condim[g], np.round(m.geom_friction[g], 4), m0))


def configure(condim, fr, mass):
    m.geom_condim[g] = condim
    m.geom_friction[g] = fr
    m.body_mass[b] = mass
    m.body_inertia[b] = I0 * (mass / m0)


def replay_ok(e):
    s0 = 0 if e == 0 else int(ends[e - 1]); e1 = int(ends[e])
    pos = D['robot0_eef_pos'][s0:e1].astype(np.float64)
    rot = D['robot0_eef_rot_axis_angle'][s0:e1].astype(np.float64)
    w = D['robot0_gripper_width'][s0:e1, 0].astype(np.float64)
    env.reset(seed=seeds[e])
    q = env.arm_q
    for t in range(1, len(pos)):
        j = min(t + L, len(pos) - 1)
        q = ik.solve(pos[j], rot[j], q)
        env.step(q, float(w[j]), render=False)
    return bool(ScriptedExpert(env).success()[0])


configs = [
    ('A baseline (condim 3)',            3, [1.5, 0.01, 0.001], 0.10),
    ('B torsional friction (condim 4)',  4, [1.5, 0.05, 0.001], 0.10),
    ('C torsional+rolling (condim 6)',   6, [1.5, 0.05, 0.010], 0.10),
    ('D friction 3.0 (condim 3)',        3, [3.0, 0.01, 0.001], 0.10),
    ('E half mass (condim 3)',           3, [1.5, 0.01, 0.001], 0.05),
    ('F all of C+D+E',                   6, [3.0, 0.05, 0.010], 0.05),
]
print('\nconfig                              | replays that succeed | scripted expert')
for name, condim, fr, mass in configs:
    configure(condim, fr, mass)
    rep = sum(replay_ok(e) for e in range(10))
    ex_ok = 0
    for s in range(10):
        env.reset(seed=s); ex = ScriptedExpert(env); ex.run(); ex_ok += int(ex.success()[0])
    print('%-35s |        %2d/10         |     %2d/10' % (name, rep, ex_ok))
env.close()
