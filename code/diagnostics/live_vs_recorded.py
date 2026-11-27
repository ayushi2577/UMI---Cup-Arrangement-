import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
ckpt = sys.argv[1]
sys.argv = ['eval_sim.py', '--ckpt', ckpt]
SRC = open(os.path.join(HERE, 'eval_sim.py')).read().split("device = torch.device('cuda:0')")[0]
exec(compile(SRC, 'eval_sim.py', 'exec'), globals())

device = torch.device('cuda:0')
policy, cfg = load_policy(os.path.expanduser(ckpt), device)
root = zarr.open_group(args.data, mode='r'); D = root['data']; ends = root['meta/episode_ends'][:]
seeds = [json.loads(l)['seed'] for l in open(args.data + '.attempts.jsonl') if l.strip() and json.loads(l)['ok']]
env = UmiSimEnv(build_scene(args.menagerie)); ik = PoseIK(env)
L, ks = 4, np.arange(8)
live_e, rec_e, img_d, live_g, rec_g = [], [], [], [], []
print('ep    t | live pos err cm | recorded pos err cm | live grip mm | rec grip mm | img diff | eef live-vs-rec cm')
for e in range(5):
    s0 = 0 if e == 0 else int(ends[e - 1]); e1 = int(ends[e])
    pos = D['robot0_eef_pos'][s0:e1].astype(np.float64)
    rot = D['robot0_eef_rot_axis_angle'][s0:e1].astype(np.float64)
    w = D['robot0_gripper_width'][s0:e1, 0].astype(np.float64)
    obs = env.reset(seed=seeds[e])
    start_pose = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_rot_axis_angle']]).astype(np.float64)
    hist = collections.deque([obs] * 4, maxlen=4)
    q = env.arm_q
    for t in range(1, len(pos)):
        j = min(t + L, len(pos) - 1)
        q = ik.solve(pos[j], rot[j], q)
        obs = env.step(q, float(w[j]), render=True)
        hist.append(obs)
        if t >= 3 and t % 24 == 0 and t + 21 < len(pos):
            act_l = predict(policy, cfg, build_env_obs([hist[0], hist[-1]]), start_pose, device)
            sl = slice(s0 + t - 3, s0 + t + 1, 3)
            rec_obs = {'camera0_rgb': D['camera0_rgb'][sl],
                       'robot0_eef_pos': D['robot0_eef_pos'][sl].astype(np.float64),
                       'robot0_eef_rot_axis_angle': D['robot0_eef_rot_axis_angle'][sl].astype(np.float64),
                       'robot0_gripper_width': D['robot0_gripper_width'][sl].astype(np.float64)}
            act_r = predict(policy, cfg, rec_obs, D['robot0_demo_start_pose'][s0 + t].astype(np.float64), device)
            tgt, tg = pos[t + 3 * ks], w[t + 3 * ks]
            le = 100 * np.linalg.norm(act_l[ks, :3] - tgt, axis=1).mean()
            re = 100 * np.linalg.norm(act_r[ks, :3] - tgt, axis=1).mean()
            lg = 1000 * np.abs(act_l[ks, 6] - tg).mean(); rg = 1000 * np.abs(act_r[ks, 6] - tg).mean()
            dimg = np.abs(hist[-1]['camera0_rgb'].astype(float) - D['camera0_rgb'][s0 + t].astype(float)).mean()
            dp = 100 * np.linalg.norm(hist[-1]['robot0_eef_pos'] - pos[t])
            live_e.append(le); rec_e.append(re); img_d.append(dimg); live_g.append(lg); rec_g.append(rg)
            print('%d  %4d |     %6.2f      |       %6.2f        |   %6.2f    |   %6.2f    |  %6.2f  |  %5.2f' % (e, t, le, re, lg, rg, dimg, dp))
print('\nMEAN: live pos err %.2f cm | recorded %.2f cm | live grip %.2f mm | recorded %.2f mm | img diff %.2f' % (
    np.mean(live_e), np.mean(rec_e), np.mean(live_g), np.mean(rec_g), np.mean(img_d)))
