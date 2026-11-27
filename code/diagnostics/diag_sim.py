import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'eval_sim.py')).read().split("device = torch.device('cuda:0')")[0]
ckpt = sys.argv[1]; seed = int(sys.argv[2])
sys.argv = ['eval_sim.py', '--ckpt', ckpt]
exec(compile(SRC, 'eval_sim.py', 'exec'), globals())
device = torch.device('cuda:0')
policy, cfg = load_policy(os.path.expanduser(ckpt), device)
env = UmiSimEnv(build_scene(args.menagerie)); ik = PoseIK(env)
torch.manual_seed(seed)
obs = env.reset(seed=seed)
hist = collections.deque([obs] * 4, maxlen=4)
start_pose = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_rot_axis_angle']]).astype(np.float64)
q_cmd = env.arm_q; steps = 0; M = 3 * args.n_exec; shots = []
s0 = env.get_object_state()
print('seed', seed, '| cup xy', np.round(s0['cup_pos'][:2], 3), '| saucer xy', np.round(s0['saucer_pos'][:2], 3))
print('step | tcp xyz | grip | cup xyz | pred first dxyz (cm) | pred last dxyz (cm) | pred grip first/last')
while steps < args.max_steps:
    act = predict(policy, cfg, build_env_obs([hist[0], hist[-1]]), start_pose, device)
    cur = hist[-1]; tcp = cur['robot0_eef_pos']; cup = env.get_object_state()['cup_pos']
    print('%4d | %s | %.3f | %s | %s | %s | %.3f / %.3f' % (
        steps, np.round(tcp, 3), cur['robot0_gripper_width'][0], np.round(cup, 3),
        np.round(100 * (act[0, :3] - tcp), 1), np.round(100 * (act[-1, :3] - tcp), 1), act[0, 6], act[-1, 6]))
    if len(shots) < 10:
        shots.append(np.concatenate([cur['camera0_rgb'], env.render('front', 224, 224)], axis=0))
    pos, rot, grip = interp_plan(act, cur['robot0_eef_pos'], cur['robot0_eef_rot_axis_angle'],
                                 cur['robot0_gripper_width'][0], args.shift, M)
    for m in range(M):
        p = pos[m].copy(); p[2] = max(p[2], 0.015)
        q_cmd = ik.solve(p, rot[m], q_cmd)
        hist.append(env.step(q_cmd, float(np.clip(grip[m], 0, 0.08)), render=True)); steps += 1
        if steps >= args.max_steps: break
out = os.path.expanduser('~/umi_mujoco/eval_results/diag_seed%03d.png' % seed)
os.makedirs(os.path.dirname(out), exist_ok=True)
imageio.imwrite(out, np.concatenate(shots, axis=1))
print('saved', out, '(top row: what the policy saw, bottom row: front view, one column per policy call)')
env.close()
