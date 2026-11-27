import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'eval_sim.py')).read().split("device = torch.device('cuda:0')")[0]
ckpt = sys.argv[1]
nep = int(sys.argv[2]) if len(sys.argv) > 2 else 60
s0 = int(sys.argv[3]) if len(sys.argv) > 3 else 200
sys.argv = ['eval_sim.py', '--ckpt', ckpt, '--lead', '6']
exec(compile(SRC, 'eval_sim.py', 'exec'), globals())
device = torch.device('cuda:0')
policy, cfg = load_policy(os.path.expanduser(ckpt), device)
env = UmiSimEnv(build_scene(args.menagerie))
ik = PoseIK(env)
XB, HOLD = 0.012, 30


def run(seed, retreat):
    torch.manual_seed(seed)
    obs = env.reset(seed=seed)
    chk = ScriptedExpert(env)
    hist = collections.deque([obs] * 4, maxlen=4)
    start_pose = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_rot_axis_angle']]).astype(np.float64)
    q_cmd, steps, M = env.arm_q, 0, 3 * args.n_exec
    streak, held, trig = 0, False, None
    while steps < 400 and trig is None:
        act = predict(policy, cfg, build_env_obs([hist[0], hist[-1]]), start_pose, device)
        cur = hist[-1]
        pos, rot, grip = interp_plan(act, cur['robot0_eef_pos'], cur['robot0_eef_rot_axis_angle'],
                                     cur['robot0_gripper_width'][0], args.shift, M)
        xb = XB if cur['robot0_gripper_width'][0] > 0.06 else 0.0
        for m in range(M):
            p = pos[m].copy(); p[2] = max(p[2], 0.027); p[0] += xb
            q_cmd = ik.solve(p, rot[m], q_cmd)
            obs = env.step(q_cmd, float(np.clip(grip[m], 0.0, 0.08)), render=True)
            hist.append(obs); steps += 1
            ok_now = bool(chk.success()[0])
            streak = streak + 1 if ok_now else 0
            if streak >= HOLD:
                held = True
            if retreat and streak >= 10:
                trig = (p.copy(), rot[m])
                break
    if trig is not None:
        p0, r0 = trig
        end, k = steps + 100, 0
        while steps < end:
            k += 1
            p = p0.copy()
            if k > 20:
                p[2] += min(0.10, 0.003 * (k - 20))
            q_cmd = ik.solve(p, r0, q_cmd)
            env.step(q_cmd, 0.08, render=True)
            steps += 1
            ok_now = bool(chk.success()[0])
            streak = streak + 1 if ok_now else 0
            if streak >= HOLD:
                held = True
    return bool(chk.success()[0]), held, trig is not None


tot = dict(plain_end=0, plain_held=0, v2_end=0, v2_held=0, v2_triggered=0)
print('seed | plain end | plain held | v2 end | v2 held | v2 triggered')
for seed in range(s0, s0 + nep):
    a, b, _ = run(seed, False)
    c, d, t = run(seed, True)
    tot['plain_end'] += a; tot['plain_held'] += b; tot['v2_end'] += c; tot['v2_held'] += d; tot['v2_triggered'] += t
    print('%4d | %d | %d | %d | %d | %d' % (seed, a, b, c, d, t))
    sys.stdout.flush()
for k, v in tot.items():
    pr = v / nep
    print('RESULT %-13s %2d/%d = %2.0f%% (+/- %2.0f%%)' % (k, v, nep, 100 * pr, 196 * (pr * (1 - pr) / nep) ** 0.5))
sys.stdout.flush()
os._exit(0)
