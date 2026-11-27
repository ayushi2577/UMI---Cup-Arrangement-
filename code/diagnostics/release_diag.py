import os, sys, inspect
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
XB = 0.012
try:
    print('SUCCESS CRITERION SOURCE:')
    print(inspect.getsource(ScriptedExpert.success))
except Exception as e:
    print('could not read success source:', e)


def run(seed):
    torch.manual_seed(seed)
    obs = env.reset(seed=seed)
    hist = collections.deque([obs] * 4, maxlen=4)
    start_pose = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_rot_axis_angle']]).astype(np.float64)
    q_cmd, steps, M = env.arm_q, 0, 3 * args.n_exec
    tc, rel, maxz, prev, traj = None, None, 0.0, None, []
    while steps < 400:
        act = predict(policy, cfg, build_env_obs([hist[0], hist[-1]]), start_pose, device)
        cur = hist[-1]
        pos, rot, grip = interp_plan(act, cur['robot0_eef_pos'], cur['robot0_eef_rot_axis_angle'],
                                     cur['robot0_gripper_width'][0], args.shift, M)
        xb = XB if cur['robot0_gripper_width'][0] > 0.06 else 0.0
        for m in range(M):
            p = pos[m].copy(); p[2] = max(p[2], 0.027); p[0] += xb
            q_cmd = ik.solve(p, rot[m], q_cmd)
            wc = float(np.clip(grip[m], 0.0, 0.08))
            obs = env.step(q_cmd, wc, render=True)
            hist.append(obs); steps += 1
            st = env.get_object_state()
            cup = np.asarray(st['cup_pos'], dtype=float)
            sau = np.asarray(st['saucer_pos'], dtype=float)
            tcp = np.asarray(env.eef_pose()[0], dtype=float)
            speed = 0.0 if prev is None else 1000 * float(np.linalg.norm(tcp - prev))
            prev = tcp
            maxz = max(maxz, float(cup[2]))
            traj.append((float(100 * np.linalg.norm(cup[:2] - sau[:2])), float(100 * (cup[2] - sau[2])), speed))
            if tc is None and obs['robot0_gripper_width'][0] < 0.06:
                tc = steps
            if tc is not None and rel is None and maxz > 0.06 and wc > 0.06:
                rel = len(traj) - 1
    ok = bool(ScriptedExpert(env).success()[0])
    return ok, rel, traj


rows = []
print('seed | ok | release step | cup-saucer dist at release (cm) | cup height rel. saucer at release (cm) | hand speed at release (mm/step) | dist +15 steps (cm) | final dist (cm) | final cup height (cm)')
for seed in range(nep):
    ok, rel, traj = run(seed)
    if rel is None:
        print('%4d | %d | never released | - | - | - | - | %.1f | %.1f' % (seed, ok, traj[-1][0], traj[-1][1]))
        continue
    r = traj[rel]; a = traj[min(rel + 15, len(traj) - 1)]; f = traj[-1]
    rows.append((ok, r[0], r[1], r[2], a[0], f[0], f[1]))
    print('%4d | %d | %d | %.1f | %.1f | %.2f | %.1f | %.1f | %.1f' % (seed, ok, rel, r[0], r[1], r[2], a[0], f[0], f[1]))
for label, flag in (('SUCCESS', True), ('FAIL', False)):
    g = [r for r in rows if r[0] == flag]
    if g:
        print('%s (%d released): median release dist %.1f cm | median release height %.1f cm | median hand speed %.2f mm/step | median dist +15 %.1f cm | median final dist %.1f cm' % (
            label, len(g), np.median([r[1] for r in g]), np.median([r[2] for r in g]),
            np.median([r[3] for r in g]), np.median([r[4] for r in g]), np.median([r[5] for r in g])))
sys.stdout.flush()
os._exit(0)
