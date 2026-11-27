import os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'replay_test.py')).read().split("env = UmiSimEnv(build_scene(")[0]
exec(compile(SRC, 'replay_test.py', 'exec'), globals())
import umi_sim.env as envmod
envmod.SQUEEZE_BELOW = 0.06

env = UmiSimEnv(build_scene(os.path.expanduser('~/umi_mujoco/sim/mujoco_menagerie')))
ik = PoseIK(env)
E, L = 0, 4
seed = seeds[E]
n = int(ends[E])
pos = D['robot0_eef_pos'][0:n].astype(np.float64)
rot = D['robot0_eef_rot_axis_angle'][0:n].astype(np.float64)
w = D['robot0_gripper_width'][0:n, 0].astype(np.float64)


def snap(cmd):
    return (env.eef_pose()[0].copy(), env.gripper_width, env.get_object_state()['cup_pos'].copy(), cmd)


env.reset(seed=seed)
ex = ScriptedExpert(env)
log_e = []
ex.on_step = lambda obs, x_cmd, grip: log_e.append(snap(grip))
ex.run()
print('expert on seed %d: success=%s, %d steps' % (seed, ex.success()[0], len(log_e)))

env.reset(seed=seed)
q = env.arm_q
log_r = []
for t in range(1, n):
    j = min(t + L, n - 1)
    q = ik.solve(pos[j], rot[j], q)
    env.step(q, float(w[j]), render=False)
    log_r.append(snap(float(w[j])))
print('replay (lead %d): %d steps' % (L, len(log_r)))

dt = 10 * env.model.opt.timestep
print('\nstep | xy speed m/s exp/rep | tcp z exp/rep | finger width exp/rep | gripper cmd exp/rep | cup z exp/rep | cup-tcp xy cm exp/rep')
for t in range(160, 216, 3):
    i = t - 1
    if i >= len(log_e) or i >= len(log_r):
        break
    out = []
    for lg in (log_e, log_r):
        p, wd, cup, cmd = lg[i]
        sp = np.linalg.norm(p[:2] - lg[i - 3][0][:2]) / (3 * dt)
        out.append((sp, p[2], wd, cmd, cup[2], 100 * np.linalg.norm(cup[:2] - p[:2])))
    a, b = out
    print('%4d |    %.2f / %.2f      | %.3f / %.3f |    %.3f / %.3f     |   %.3f / %.3f    | %.3f / %.3f | %5.1f / %5.1f' % (
        t, a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3], a[4], b[4], a[5], b[5]))
env.close()
