import os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'replay_test.py')).read().split("env = UmiSimEnv(build_scene(")[0]
exec(compile(SRC, 'replay_test.py', 'exec'), globals())
import umi_sim.env as envmod


def replay_cfg(env, ik, e, L, trace=False):
    s0 = 0 if e == 0 else int(ends[e - 1]); e1 = int(ends[e])
    pos = D['robot0_eef_pos'][s0:e1].astype(np.float64)
    rot = D['robot0_eef_rot_axis_angle'][s0:e1].astype(np.float64)
    w = D['robot0_gripper_width'][s0:e1, 0].astype(np.float64)
    env.reset(seed=seeds[e])
    q, max_cup_z, max_xy, rows = env.arm_q, 0.0, 0.0, []
    for t in range(1, len(pos)):
        j = min(t + L, len(pos) - 1)
        q = ik.solve(pos[j], rot[j], q)
        env.step(q, float(w[j]), render=False)
        p = env.eef_pose()[0]
        cup = env.get_object_state()['cup_pos']
        max_cup_z = max(max_cup_z, float(cup[2]))
        max_xy = max(max_xy, float(np.linalg.norm(p[:2] - pos[t][:2])))
        if trace and t % 20 == 0:
            rows.append((t, pos[t][2], p[2], w[t], env.gripper_width, cup[2], np.linalg.norm(cup[:2] - p[:2])))
    ok, dxy, cz = ScriptedExpert(env).success()
    return ok, max_cup_z, max_xy, rows


env = UmiSimEnv(build_scene(os.path.expanduser('~/umi_mujoco/sim/mujoco_menagerie')))
ik = PoseIK(env)
print('squeeze below | lead | replays that succeed | mean max cup height (cm) | mean max xy tracking error (cm)')
for thr in (0.06, 0.075):
    envmod.SQUEEZE_BELOW = thr
    for L in (0, 4):
        res = [replay_cfg(env, ik, e, L) for e in range(10)]
        print('   %.3f     |  %d   |       %2d/10        |         %5.1f          |            %4.1f' % (
            thr, L, sum(r[0] for r in res), 100 * np.mean([r[1] for r in res]), 100 * np.mean([r[2] for r in res])))

envmod.SQUEEZE_BELOW = 0.06
ok, mz, mxy, rows = replay_cfg(env, ik, 0, 4, trace=True)
print('\ntrace of episode 0 (lead 4, squeeze below 0.06), success=%s' % ok)
print(' step | recorded tcp z | replay tcp z | recorded width | replay width | cup z | cup-tcp xy dist')
for r in rows:
    print(' %4d |     %.3f      |    %.3f     |     %.3f      |    %.3f     | %.3f |     %.3f' % r)
env.close()
