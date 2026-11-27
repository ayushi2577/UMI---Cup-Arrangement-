import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'eval_sim.py')).read().split("device = torch.device('cuda:0')")[0]
ckpt = sys.argv[1]
nep = int(sys.argv[2]) if len(sys.argv) > 2 else 60
s0 = int(sys.argv[3]) if len(sys.argv) > 3 else 100
sys.argv = ['eval_sim.py', '--ckpt', ckpt, '--lead', '6']
exec(compile(SRC, 'eval_sim.py', 'exec'), globals())
device = torch.device('cuda:0')
policy, cfg = load_policy(os.path.expanduser(ckpt), device)
env = UmiSimEnv(build_scene(args.menagerie))
ik = PoseIK(env)
XB = 0.012


def run(seed):
    torch.manual_seed(seed)
    obs = env.reset(seed=seed)
    chk = ScriptedExpert(env)
    hist = collections.deque([obs] * 4, maxlen=4)
    start_pose = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_rot_axis_angle']]).astype(np.float64)
    q_cmd, steps, M = env.arm_q, 0, 3 * args.n_exec
    base, sep = None, None
    while steps < 480:
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
            s = env.get_object_state()
            cup, sau = s['cup_pos'], s['saucer_pos']
            eef = obs['robot0_eef_pos']
            w = float(obs['robot0_gripper_width'][0])
            if base is None and cup[2] > 0.06:
                base = cup - eef
            if base is not None and sep is None and np.linalg.norm((cup - eef) - base) > 0.03:
                sep = (steps, float(np.linalg.norm(cup[:2] - sau[:2])), w, float(cup[2]))
            if steps >= 480:
                break
    return bool(chk.success()[0]), base is not None, sep


def cat(ok, lifted, sep):
    if not lifted: return 'never lifted'
    if sep is None: return 'held to the end'
    near, opened = sep[1] <= 0.08, sep[2] >= 0.06
    if near and opened: return 'released at saucer, ok' if ok else 'released at saucer, failed after'
    if near: return 'slipped near saucer'
    if opened: return 'gripper opened en route'
    return 'slipped en route'


cnt = collections.Counter()
print('seed | ok | lifted | sep step | cup-saucer dist (m) | grip width | cup z')
for seed in range(s0, s0 + nep):
    ok, lifted, sep = run(seed)
    c = cat(ok, lifted, sep); cnt[c] += 1
    print(seed, int(ok), int(lifted), sep, c)
    sys.stdout.flush()
for k, v in cnt.most_common():
    print('RESULT %-34s %d' % (k, v))
