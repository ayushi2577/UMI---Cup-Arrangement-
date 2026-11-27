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
XB = 0.012


def saucer_xy(st):
    for k, v in st.items():
        if 'saucer' in k.lower() and 'pos' in k.lower():
            return np.asarray(v, dtype=float)[:2]
    return None


def run(seed):
    torch.manual_seed(seed)
    obs = env.reset(seed=seed)
    hist = collections.deque([obs] * 4, maxlen=4)
    start_pose = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_rot_axis_angle']]).astype(np.float64)
    q_cmd, steps, M = env.arm_q, 0, 3 * args.n_exec
    tc, leave, maxz, wmax_hold = None, None, 0.0, 0.0
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
            tcp = np.asarray(env.eef_pose()[0], dtype=float)
            maxz = max(maxz, float(cup[2]))
            if tc is None and obs['robot0_gripper_width'][0] < 0.06:
                tc = steps
            if tc is not None and leave is None:
                if np.linalg.norm(cup - tcp) > 0.04 and maxz > 0.06:
                    sx = saucer_xy(st)
                    leave = dict(step=steps, wc=wc, z=float(cup[2]),
                                 d_saucer=None if sx is None else float(100 * np.linalg.norm(tcp[:2] - sx)))
    st = env.get_object_state()
    sx = saucer_xy(st)
    cup = np.asarray(st['cup_pos'], dtype=float)
    dend = None if sx is None else float(100 * np.linalg.norm(cup[:2] - sx))
    ok = bool(ScriptedExpert(env).success()[0])
    return ok, tc, leave, maxz, dend, st


counts = {}
print('state keys:', None)
print('seed | ok | closes | leaves hand at | cmd width then | hand-saucer dist then (cm) | cup-saucer dist end (cm) | outcome')
for seed in range(nep):
    ok, tc, leave, maxz, dend, st = run(seed)
    if seed == 0:
        print('state keys:', list(st.keys()))
    if ok: oc = 'success'
    elif tc is None: oc = 'never_closed'
    elif maxz < 0.06: oc = 'never_lifted'
    elif leave is None: oc = 'carried_not_placed'
    elif leave['wc'] < 0.06: oc = 'slipped_while_closed'
    elif leave['d_saucer'] is not None and leave['d_saucer'] < 6: oc = 'released_over_saucer_but_missed'
    else: oc = 'released_away_from_saucer'
    counts[oc] = counts.get(oc, 0) + 1
    print('%4d | %d | %s | %s | %s | %s | %s | %s' % (seed, ok, tc,
          None if leave is None else leave['step'],
          None if leave is None else '%.3f' % leave['wc'],
          None if leave is None or leave['d_saucer'] is None else '%.1f' % leave['d_saucer'],
          None if dend is None else '%.1f' % dend, oc))
print('OUTCOME COUNTS:', counts)
