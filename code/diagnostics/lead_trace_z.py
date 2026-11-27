import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'eval_sim.py')).read().split("device = torch.device('cuda:0')")[0]
ckpt = sys.argv[1]
sys.argv = ['eval_sim.py', '--ckpt', ckpt]
exec(compile(SRC, 'eval_sim.py', 'exec'), globals())
device = torch.device('cuda:0')
policy, cfg = load_policy(os.path.expanduser(ckpt), device)
env = UmiSimEnv(build_scene(args.menagerie))
ik = PoseIK(env)


def run(seed, lead):
    args.lead = lead
    torch.manual_seed(seed)
    obs = env.reset(seed=seed)
    cup0 = env.get_object_state()['cup_pos'].copy()
    hist = collections.deque([obs] * 4, maxlen=4)
    start_pose = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_rot_axis_angle']]).astype(np.float64)
    q_cmd, steps, M = env.arm_q, 0, 3 * args.n_exec
    close_info, max_cup, min_z = None, 0.0, 1.0
    while steps < 330:
        act = predict(policy, cfg, build_env_obs([hist[0], hist[-1]]), start_pose, device)
        cur = hist[-1]
        pos, rot, grip = interp_plan(act, cur['robot0_eef_pos'], cur['robot0_eef_rot_axis_angle'],
                                     cur['robot0_gripper_width'][0], args.shift, M)
        for m in range(M):
            p = pos[m].copy(); p[2] = max(p[2], 0.027)
            q_cmd = ik.solve(p, rot[m], q_cmd)
            obs = env.step(q_cmd, float(np.clip(grip[m], 0, 0.08)), render=True)
            hist.append(obs); steps += 1
            tcp, w = obs['robot0_eef_pos'], obs['robot0_gripper_width'][0]
            min_z = min(min_z, tcp[2])
            max_cup = max(max_cup, float(env.get_object_state()['cup_pos'][2]))
            if close_info is None and w < 0.07:
                close_info = (100 * np.linalg.norm(tcp[:2] - cup0[:2]), 100 * tcp[2], steps)
    ok = bool(ScriptedExpert(env).success()[0])
    return close_info, 100 * min_z, 100 * max_cup, ok


print('lead | seed | xy offset to cup when closing starts (cm) | tcp z then (cm) | step | lowest tcp z (cm) | max cup height (cm) | success')
for lead in (0, 4, 8):
    for seed in (1001, 1002, 1003, 0, 1):
        ci, mz, mc, ok = run(seed, lead)
        if ci is None:
            print(' %d | %4d | never closed | | | %.1f | %.1f | %s' % (lead, seed, mz, mc, ok))
        else:
            print(' %d | %4d | %.1f | %.1f | %d | %.1f | %.1f | %s' % (lead, seed, ci[0], ci[1], ci[2], mz, mc, ok))
env.close()
