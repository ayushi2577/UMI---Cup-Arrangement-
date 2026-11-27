import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'eval_sim.py')).read().split("device = torch.device('cuda:0')")[0]
ckpt = sys.argv[1]
nep = int(sys.argv[2]) if len(sys.argv) > 2 else 50
s0 = int(sys.argv[3]) if len(sys.argv) > 3 else 100
sys.argv = ['eval_sim.py', '--ckpt', ckpt, '--lead', '6']
exec(compile(SRC, 'eval_sim.py', 'exec'), globals())
device = torch.device('cuda:0')
policy, cfg = load_policy(os.path.expanduser(ckpt), device)
env = UmiSimEnv(build_scene(args.menagerie))
ik = PoseIK(env)
model = env.model
g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'cup_geom')
XB = 0.012


def run(seed):
    torch.manual_seed(seed)
    obs = env.reset(seed=seed)
    chk = ScriptedExpert(env)
    hist = collections.deque([obs] * 4, maxlen=4)
    start_pose = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_rot_axis_angle']]).astype(np.float64)
    q_cmd, steps, M = env.arm_q, 0, 3 * args.n_exec
    streak, held, maxz = 0, False, 0.0
    while steps < 400:
        act = predict(policy, cfg, build_env_obs([hist[0], hist[-1]]), start_pose, device)
        cur = hist[-1]
        pos, rot, grip = interp_plan(act, cur['robot0_eef_pos'], cur['robot0_eef_rot_axis_angle'],
                                     cur['robot0_gripper_width'][0], args.shift, M)
        xb = XB if cur['robot0_gripper_width'][0] > 0.06 else 0.0
        for j in range(M):
            p = pos[j].copy(); p[2] = max(p[2], 0.027); p[0] += xb
            q_cmd = ik.solve(p, rot[j], q_cmd)
            obs = env.step(q_cmd, float(np.clip(grip[j], 0.0, 0.08)), render=True)
            hist.append(obs); steps += 1
            maxz = max(maxz, float(env.get_object_state()['cup_pos'][2]))
            streak = streak + 1 if chk.success()[0] else 0
            if streak >= 30:
                held = True
    return bool(chk.success()[0]), held, maxz > 0.06


print('cup friction | end success | placed and stayed 30 steps at some point | lifted >6cm')
for mu in (1.5, 1.0, 0.7):
    model.geom_friction[g] = [mu, 0.05, 0.001]
    res = [run(s) for s in range(s0, s0 + nep)]
    print('     %.1f     |    %2d/%d    |                 %2d/%d                  |   %2d/%d' % (
        mu, sum(r[0] for r in res), nep, sum(r[1] for r in res), nep, sum(r[2] for r in res), nep), flush=True)
env.close()
