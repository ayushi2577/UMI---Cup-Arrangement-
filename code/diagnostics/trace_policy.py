import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'eval_sim.py')).read().split("device = torch.device('cuda:0')")[0]
ckpt = sys.argv[1]; seed = int(sys.argv[2])
sys.argv = ['eval_sim.py', '--ckpt', ckpt]
exec(compile(SRC, 'eval_sim.py', 'exec'), globals())
device = torch.device('cuda:0')
policy, cfg = load_policy(os.path.expanduser(ckpt), device)
env = UmiSimEnv(build_scene(args.menagerie))
ik = PoseIK(env)

exp = [env.reset(seed=seed)]
ex = ScriptedExpert(env, render=False)
ex.on_step = lambda obs, x, g: exp.append(obs)
ex.run()
print('expert on seed %d: success=%s, %d steps' % (seed, ex.success()[0], len(exp) - 1))

torch.manual_seed(seed)
obs = env.reset(seed=seed)
s0 = env.get_object_state()
print('cup xy %s | saucer xy %s' % (np.round(s0['cup_pos'][:2], 3), np.round(s0['saucer_pos'][:2], 3)))
hist = collections.deque([obs] * 4, maxlen=4)
start_pose = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_rot_axis_angle']]).astype(np.float64)
q_cmd, steps, M, shots = env.arm_q, 0, 3 * args.n_exec, []
print('\nstep | policy tcp xyz (m) | expert tcp xyz (m) | width pol/exp | cup xyz | pred dz cm (first/last) | pred width (first/last)')
while steps < 330:
    act = predict(policy, cfg, build_env_obs([hist[0], hist[-1]]), start_pose, device)
    cur = hist[-1]; tcp = cur['robot0_eef_pos']
    cup = env.get_object_state()['cup_pos']
    e = exp[min(steps, len(exp) - 1)]
    print('%4d | %.3f %.3f %.3f | %.3f %.3f %.3f | %.3f / %.3f | %.3f %.3f %.3f | %5.1f / %5.1f | %.3f / %.3f' % (
        steps, tcp[0], tcp[1], tcp[2], *e['robot0_eef_pos'], cur['robot0_gripper_width'][0], e['robot0_gripper_width'][0],
        *cup, 100 * (act[0, 2] - tcp[2]), 100 * (act[7, 2] - tcp[2]), act[0, 6], act[7, 6]))
    if len(shots) < 12:
        shots.append(cur['camera0_rgb'])
    pos, rot, grip = interp_plan(act, cur['robot0_eef_pos'], cur['robot0_eef_rot_axis_angle'],
                                 cur['robot0_gripper_width'][0], args.shift, M)
    for m in range(M):
        p = pos[m].copy(); p[2] = max(p[2], 0.015)
        q_cmd = ik.solve(p, rot[m], q_cmd)
        hist.append(env.step(q_cmd, float(np.clip(grip[m], 0, 0.08)), render=True)); steps += 1
out = os.path.expanduser('~/umi_mujoco/eval_results/trace_seed%d.png' % seed)
os.makedirs(os.path.dirname(out), exist_ok=True)
imageio.imwrite(out, np.concatenate(shots, axis=1))
print('\nsaved', out, '(wrist view the policy saw at each call)')
env.close()
