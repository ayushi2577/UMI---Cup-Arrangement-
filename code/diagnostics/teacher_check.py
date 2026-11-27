import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'eval_sim.py')).read().split("device = torch.device('cuda:0')")[0]
ckpt = sys.argv[1]
sys.argv = ['eval_sim.py', '--ckpt', ckpt]
exec(compile(SRC, 'eval_sim.py', 'exec'), globals())
device = torch.device('cuda:0')
policy, cfg = load_policy(os.path.expanduser(ckpt), device)
env = UmiSimEnv(build_scene(args.menagerie))


def collect(seed):
    frames = [env.reset(seed=seed)]
    ex = ScriptedExpert(env, render=True)
    ex.on_step = lambda obs, x, g: frames.append(obs)
    ex.run()
    return frames


def evaluate(seed_list):
    rows = []
    ks = np.arange(8)
    for sd in seed_list:
        frames = collect(sd)
        start = np.concatenate([frames[0]['robot0_eef_pos'], frames[0]['robot0_eef_rot_axis_angle']]).astype(np.float64)
        for t in range(3, len(frames) - 24, 9):
            act = predict(policy, cfg, build_env_obs([frames[t - 3], frames[t]]), start, device)
            tgt = np.stack([frames[t + 3 * k]['robot0_eef_pos'] for k in ks])
            wid = np.array([frames[t + 3 * k]['robot0_gripper_width'][0] for k in ks])
            cur = frames[t]['robot0_eef_pos']
            d = act[ks, :3] - tgt
            rows.append(dict(z=cur[2],
                             exy=np.linalg.norm(d[:, :2], axis=1).mean(), ez=np.abs(d[:, 2]).mean(),
                             eg=np.abs(act[ks, 6] - wid).mean(),
                             stay=np.linalg.norm(cur - tgt, axis=1).mean(),
                             gacc=((act[ks, 6] < 0.06) == (wid < 0.06)).mean()))
    return rows


def show(name, rows):
    near = [r for r in rows if r['z'] < 0.06]
    m = lambda rs, k, s=100.0: s * np.mean([r[k] for r in rs]) if rs else float('nan')
    print('%-22s | n=%3d (near table %2d) | all: xy %.2f z %.2f cm | near table: xy %.2f z %.2f cm | grip err %.1f mm | open/closed correct %3.0f%% (near table %3.0f%%) | stay-still %.2f cm' % (
        name, len(rows), len(near), m(rows, 'exy'), m(rows, 'ez'), m(near, 'exy'), m(near, 'ez'),
        m(rows, 'eg', 1000.0), m(rows, 'gacc'), m(near, 'gacc'), m(rows, 'stay')))


show('training layouts', evaluate([1001, 1002, 1003, 1004]))
show('unseen layouts', evaluate([0, 1, 2, 3]))
env.close()
