import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'eval_sim.py')).read().split("device = torch.device('cuda:0')")[0]
ckpt = sys.argv[1]
nep = int(sys.argv[2]) if len(sys.argv) > 2 else 30
sys.argv = ['eval_sim.py', '--ckpt', ckpt, '--lead', '6']
exec(compile(SRC, 'eval_sim.py', 'exec'), globals())
device = torch.device('cuda:0')
policy, cfg = load_policy(os.path.expanduser(ckpt), device)
env = UmiSimEnv(build_scene(args.menagerie))
ik = PoseIK(env)


def run(seed):
    torch.manual_seed(seed)
    obs = env.reset(seed=seed)
    hist = collections.deque([obs] * 4, maxlen=4)
    start_pose = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_rot_axis_angle']]).astype(np.float64)
    q_cmd, steps, M, log = env.arm_q, 0, 3 * args.n_exec, []
    while steps < 260:
        act = predict(policy, cfg, build_env_obs([hist[0], hist[-1]]), start_pose, device)
        cur = hist[-1]
        pos, rot, grip = interp_plan(act, cur['robot0_eef_pos'], cur['robot0_eef_rot_axis_angle'],
                                     cur['robot0_gripper_width'][0], args.shift, M)
        for m in range(M):
            p = pos[m].copy(); p[2] = max(p[2], 0.027)
            q_cmd = ik.solve(p, rot[m], q_cmd)
            w_cmd = float(np.clip(grip[m], 0.0, 0.08))
            obs = env.step(q_cmd, w_cmd, render=True)
            hist.append(obs); steps += 1
            Rm = env.data.site_xmat[env.tcp_id].reshape(3, 3)
            tcp = env.eef_pose()[0]; cup = env.get_object_state()['cup_pos']
            log.append(dict(w_cmd=w_cmd, w=float(obs['robot0_gripper_width'][0]), cup_z=float(cup[2]),
                            rel=Rm.T @ (cup - tcp), tilt=float(np.degrees(np.arccos(np.clip(-Rm[2, 2], -1, 1))))))
    return log


dropped, held = [], []
print('seed | outcome | closes at | drops at | max command width in hold (m) | max hand tilt (deg) | cup offset in hand frame at grasp x,y,z (cm)')
for seed in range(nep):
    log = run(seed)
    tc = next((i for i, r in enumerate(log) if r['w'] < 0.06), None)
    if tc is None:
        continue
    maxz, td = 0.0, None
    for i in range(tc, len(log)):
        maxz = max(maxz, log[i]['cup_z'])
        if maxz > 0.06 and log[i]['cup_z'] < 0.045:
            td = i
            break
    if maxz <= 0.06:
        continue
    end = td if td is not None else len(log)
    win = log[tc:end]
    rel = 100 * log[min(tc + 8, len(log) - 1)]['rel']
    rec = dict(seed=seed, tc=tc, td=td, cmd=max(r['w_cmd'] for r in win[5:]) if len(win) > 5 else 0.0,
               tilt=max(r['tilt'] for r in win), rel=rel)
    (dropped if td is not None else held).append(rec)
    print('%4d | %-8s | %4d | %5s | %.3f | %5.1f | %5.1f %5.1f %5.1f' % (
        seed, 'dropped' if td is not None else 'held', tc, td if td is not None else '-', rec['cmd'], rec['tilt'], *rel))
env.close()
for name, g in (('DROPPED', dropped), ('HELD to the end', held)):
    if not g:
        print('%s: none' % name)
        continue
    print('%s (%d episodes): median max command %.3f m | commands above 0.06 in %d | median max tilt %.1f deg | median |offset x| %.1f cm, |offset y| %.1f cm' % (
        name, len(g), np.median([r['cmd'] for r in g]), sum(r['cmd'] > 0.06 for r in g), np.median([r['tilt'] for r in g]),
        np.median([abs(r['rel'][0]) for r in g]), np.median([abs(r['rel'][1]) for r in g])))
