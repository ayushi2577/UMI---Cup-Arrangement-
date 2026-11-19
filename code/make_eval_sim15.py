"""Create eval_sim15.py = eval_sim12.py + a signed, per-axis, grasp-phase check (--check_grasp N).
eval_sim12.py is only read. Refuses to write if any anchor is missing or not unique, or if the output exists.
Run from ~/umi_mujoco/sim/umi_sim:  python make_eval_sim15.py
"""
import os, sys

SRC, DST = 'eval_sim12.py', 'eval_sim15.py'

A_FLAG = "ap.add_argument('--check', type=int, default=0, help='N recorded samples for the plumbing check')\n"
A_DEF = "def check(policy, cfg, device):\n"
A_END = r"    print('\nbest-matching shift: %d  (use --shift %d for rollouts)' % (best, best))" + "\n"

NEW_FLAG = "ap.add_argument('--check_grasp', type=int, default=0, help='N samples just before the first close in recorded demos: signed per-axis policy error (use with --check 1)')\n"

NEW_FUNC = r'''def check_grasp(policy, cfg, device):
    """Signed (policy minus demo) position error, mm, for samples just before the demo's first close command."""
    root = zarr.open_group(args.data, mode='r')
    D = root['data']
    ends = root['meta/episode_ends'][:]
    rng = np.random.default_rng(1)
    ks = np.arange(8)
    rows, tries = [], 0
    while len(rows) < args.check_grasp and tries < 30 * args.check_grasp:
        tries += 1
        e = int(rng.integers(len(ends)))
        s0 = 0 if e == 0 else int(ends[e - 1]); e1 = int(ends[e])
        if e1 - s0 < 120:
            continue
        w = D['robot0_gripper_width'][s0:e1, 0]
        hit = np.nonzero(w < 0.07)[0]
        if len(hit) == 0:
            continue
        c = s0 + int(hit[0])
        off = int(rng.integers(3, 61))
        t = c - off
        if t < s0 + 3 or t + 55 > e1:
            continue
        sl = slice(t - 3, t + 1, 3)
        env_obs = {
            'camera0_rgb': D['camera0_rgb'][sl],
            'robot0_eef_pos': D['robot0_eef_pos'][sl].astype(np.float64),
            'robot0_eef_rot_axis_angle': D['robot0_eef_rot_axis_angle'][sl].astype(np.float64),
            'robot0_gripper_width': D['robot0_gripper_width'][sl].astype(np.float64),
        }
        start_pose = D['robot0_demo_start_pose'][t].astype(np.float64)
        act = predict(policy, cfg, env_obs, start_pose, device)
        slab_p = D['robot0_eef_pos'][t:t + 55].astype(np.float64)
        err = (act[ks, :3] - slab_p[3 * ks]).mean(axis=0)
        grasp_z = float(D['robot0_eef_pos'][c][2])
        rows.append([off, err[0], err[1], err[2], grasp_z])
    A = np.array(rows)
    print('\nCHECK_GRASP on %d samples from %s (frames 3-60 before the first close; signed = policy minus demo, mm)' % (len(A), args.data))
    print('demo eef z at first close: mean %.1f mm' % (1000 * A[:, 4].mean()))
    for name, m in (('all 3-60 before close', A[:, 0] >= 0), ('3-20 before close (hand nearly still)', A[:, 0] <= 20), ('21-60 before close (final approach)', A[:, 0] > 20)):
        B = A[m]
        if len(B) < 2:
            continue
        mu, se = 1000 * B[:, 1:4].mean(axis=0), 1000 * B[:, 1:4].std(axis=0, ddof=1) / np.sqrt(len(B))
        print('%-40s n=%3d  x %+6.1f +/-%.1f   y %+6.1f +/-%.1f   z %+6.1f +/-%.1f' % (
            name, len(B), mu[0], se[0], mu[1], se[1], mu[2], se[2]))
    print('If the policy needed the +12 mm x/z offsets because it aims low/short, x and z above would be near -12 mm.')


'''

NEW_TAIL = "    if args.check_grasp > 0:\n        check_grasp(policy, cfg, device)\n"


def main():
    if not os.path.exists(SRC):
        sys.exit('missing ' + SRC + ' (run from ~/umi_mujoco/sim/umi_sim)')
    if os.path.exists(DST):
        sys.exit(DST + ' already exists; refusing to overwrite')
    s = open(SRC).read()
    for a in (A_FLAG, A_DEF, A_END):
        n = s.count(a)
        if n != 1:
            sys.exit('anchor found %d times (need exactly 1), nothing written:\n%s' % (n, a))
    s = s.replace(A_FLAG, A_FLAG + NEW_FLAG)
    s = s.replace(A_DEF, NEW_FUNC + A_DEF)
    s = s.replace(A_END, A_END + NEW_TAIL)
    open(DST, 'w').write(s)
    compile(s, DST, 'exec')
    print('wrote %s (%d lines); %s untouched' % (DST, s.count('\n'), SRC))


main()
