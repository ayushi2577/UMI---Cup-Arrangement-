"""Builds umi_sim/eval_sim4.py from eval_sim3.py. Adds 3 retreat-tuning flags, all defaulting to the
current behaviour (so eval_sim4.py with no new flags == eval_sim3.py). Refuses to write if any anchor
does not match exactly once. eval_sim.py / eval_sim2.py / eval_sim3.py are not touched.
Run from ~/umi_mujoco/sim:  python ~/umi_mujoco/make_eval_sim4.py
"""
import os, sys

src = os.path.expanduser('~/umi_mujoco/sim/umi_sim/eval_sim3.py')
dst = os.path.expanduser('~/umi_mujoco/sim/umi_sim/eval_sim4.py')
s = open(src).read()

edits = [
    # 1. new flags, right after --retreat
    ("ap.add_argument('--retreat', action='store_true', help='robot-observable open / wait / vertical retreat after a detected release')\n",
     "ap.add_argument('--retreat', action='store_true', help='robot-observable open / wait / vertical retreat after a detected release')\n"
     "ap.add_argument('--rt_hold', type=int, default=0, help='extra sim steps to hold fully open (pose fixed) before rising')\n"
     "ap.add_argument('--rt_nup', type=int, default=60, help='number of sim steps for the 10 cm rise (more = slower)')\n"
     "ap.add_argument('--rt_rise', type=float, default=0.10, help='rise height in metres')\n"),
    # 2. extra open hold before the lift phase
    ("    RL['phase'] = 'retreat_lift'\n",
     "    for _ in range(args.rt_hold):\n"
     "        hold(q_cmd)\n"
     "    RL['phase'] = 'retreat_lift'\n"),
    # 3. rise duration
    ("    n_up = 60\n", "    n_up = args.rt_nup\n"),
    # 4. rise height
    ("p[2] = pos0[2] + 0.10 * s", "p[2] = pos0[2] + args.rt_rise * s"),
]

for old, new in edits:
    n = s.count(old)
    if n != 1:
        sys.exit(f"ANCHOR FAILED ({n} matches), nothing written:\n{old!r}")
    s = s.replace(old, new)

open(dst, 'w').write(s)
print(f"wrote {dst} ({s.count(chr(10))} lines). eval_sim3.py untouched.")
