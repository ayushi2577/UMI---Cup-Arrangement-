import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'eval_sim.py')).read().split("device = torch.device('cuda:0')")[0]
ckpt = sys.argv[1]
nep = int(sys.argv[2]) if len(sys.argv) > 2 else 40
sys.argv = ['eval_sim.py', '--ckpt', ckpt, '--lead', '6']
exec(compile(SRC, 'eval_sim.py', 'exec'), globals())
device = torch.device('cuda:0')
policy, cfg = load_policy(os.path.expanduser(ckpt), device)
env = UmiSimEnv(build_scene(args.menagerie))
ik = PoseIK(env)


def run(seed, xbias):
    torch.manual_seed(seed)
    obs = env.reset(seed=seed)
    hist = collections.deque([obs] * 4, maxlen=4)
    start_pose = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_rot_axis_angle']]).astype(np.float64)
    q_cmd, steps, M = env.arm_q, 0, 3 * args.n_exec
    off, maxz = None, 0.0
    while steps < 400:
        act = predict(policy, cfg, build_env_obs([hist[0], hist[-1]]), start_pose, device)
        cur = hist[-1]
        pos, rot, grip = interp_plan(act, cur['robot0_eef_pos'], cur['robot0_eef_rot_axis_angle'],
                                     cur['robot0_gripper_width'][0], args.shift, M)
        xb = xbias if cur['robot0_gripper_width'][0] > 0.06 else 0.0
        for m in range(M):
            p = pos[m].copy(); p[2] = max(p[2], 0.027); p[0] += xb
            q_cmd = ik.solve(p, rot[m], q_cmd)
            obs = env.step(q_cmd, float(np.clip(grip[m], 0.0, 0.08)), render=True)
            hist.append(obs); steps += 1
            cup = env.get_object_state()['cup_pos']
            maxz = max(maxz, float(cup[2]))
            if off is None and obs['robot0_gripper_width'][0] < 0.06:
                Rm = env.data.site_xmat[env.tcp_id].reshape(3, 3)
                off = 100 * (Rm.T @ (cup - env.eef_pose()[0]))[:2]
    return bool(ScriptedExpert(env).success()[0]), maxz > 0.06, off


print('x bias (cm) | policy success | lifted >6cm | median grasp offset x (cm) | median |offset y| (cm) | episodes that closed')
for xbias in (0.0, 0.006, 0.012, 0.02):
    res = [run(s, xbias) for s in range(nep)]
    offs = [r[2] for r in res if r[2] is not None]
    mx = np.median([o[0] for o in offs]) if offs else float('nan')
    my = np.median([abs(o[1]) for o in offs]) if offs else float('nan')
    print('    %.1f    |     %2d/%d     |   %2d/%d   |           %5.2f            |         %4.2f          |   %d/%d' % (
        100 * xbias, sum(r[0] for r in res), nep, sum(r[1] for r in res), nep, mx, my, len(offs), nep))
env.close()
