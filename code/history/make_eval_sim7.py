import os, sys
D = os.path.expanduser('~/umi_mujoco/sim/umi_sim')
src = open(os.path.join(D, 'eval_sim6.py')).read()

edits = [
    # 1. new flags (anchor = line 43, seen in grep)
    ("ap.add_argument('--xbias_mode', choices=['always', 'pregrasp'], default='always', help='pregrasp: drop the x bias once the grasp is detected')\n",
     "ap.add_argument('--xbias_mode', choices=['always', 'pregrasp'], default='always', help='pregrasp: drop the x bias once the grasp is detected')\n"
     "ap.add_argument('--rt_trig', type=float, default=0.066, help='width that triggers retreat after a grasp (default 0.066 = old)')\n"
     "ap.add_argument('--rt_full', type=float, default=0.07, help='width the open-wait needs before it may exit (default 0.07 = old)')\n"
     "ap.add_argument('--rt_back', type=float, default=0.0, help='metres to move in x after the rise (negative = toward robot base); 0 = off')\n"),
    # 2. earlier trigger
    ("and w_m > 0.066:", "and w_m > args.rt_trig:"),
    # 3. wait for wider opening
    ("if w > 0.07 and stable >= 5:", "if w > args.rt_full and stable >= 5:"),
    # 4. back-off after the straight rise
    ("        hold(q_cmd)\n    for _ in range(50):\n        hold(q_cmd)\n    cdump_print()\n",
     "        hold(q_cmd)\n"
     "    if args.rt_back != 0.0:\n"
     "        RL['phase'] = 'retreat_back'\n"
     "        pb, rvb = env.eef_pose()\n"
     "        for i in range(1, 61):\n"
     "            s = i / 60\n"
     "            s = s * s * (3 - 2 * s)\n"
     "            p = pb.copy()\n"
     "            p[0] = pb[0] + args.rt_back * s\n"
     "            q_cmd = ik.solve(p, rvb, q_cmd)\n"
     "            hold(q_cmd)\n"
     "    for _ in range(50):\n        hold(q_cmd)\n    cdump_print()\n"),
]
for old, new in edits:
    if src.count(old) != 1:
        sys.exit('ANCHOR MISMATCH (%d) for: %r -- nothing written' % (src.count(old), old[:60]))
    src = src.replace(old, new)
out = os.path.join(D, 'eval_sim7.py')
open(out, 'w').write(src)
print('wrote', out)
