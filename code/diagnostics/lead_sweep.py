import os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'replay_test.py')).read().split("env = UmiSimEnv(build_scene(")[0]
exec(compile(SRC, 'replay_test.py', 'exec'), globals())


def replay_lead(env, ik, e, L):
    s0 = 0 if e == 0 else int(ends[e - 1]); e1 = int(ends[e])
    pos = D['robot0_eef_pos'][s0:e1].astype(np.float64)
    rot = D['robot0_eef_rot_axis_angle'][s0:e1].astype(np.float64)
    w = D['robot0_gripper_width'][s0:e1, 0].astype(np.float64)
    env.reset(seed=seeds[e])
    q, err = env.arm_q, []
    for t in range(1, len(pos)):
        j = min(t + L, len(pos) - 1)
        q = ik.solve(pos[j], rot[j], q)
        env.step(q, float(w[j]), render=False)
        err.append(abs(env.eef_pose()[0][2] - pos[t][2]))
    ok, dxy, cz = ScriptedExpert(env).success()
    return ok, np.max(err), np.mean(err)


env = UmiSimEnv(build_scene(os.path.expanduser('~/umi_mujoco/sim/mujoco_menagerie')))
ik = PoseIK(env)
m = env.model
print('arm position-actuator gains kp:', np.round(m.actuator_gainprm[:7, 0]), '| kv:', np.round(-m.actuator_biasprm[:7, 2]))
print('lead (control steps) | replays that succeed | mean of max height error (cm) | mean height error (cm)')
for L in (0, 2, 4, 6, 8, 10):
    res = [replay_lead(env, ik, e, L) for e in range(10)]
    print('  %2d | %2d/10 | %.1f | %.1f' % (L, sum(r[0] for r in res), 100 * np.mean([r[1] for r in res]), 100 * np.mean([r[2] for r in res])))
env.close()
