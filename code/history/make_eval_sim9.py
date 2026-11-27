import os, sys
D = os.path.expanduser('~/umi_mujoco/sim/umi_sim')
src = open(os.path.join(D, 'eval_sim8.py')).read()

edits = [
    ("ap.add_argument('--zhold', type=float, default=0.027, help='minimum target z while the cup is held (default 0.027 = old clamp)')\n",
     "ap.add_argument('--zhold', type=float, default=0.027, help='minimum target z while the cup is held (default 0.027 = old clamp)')\n"
     "ap.add_argument('--zbias', type=float, default=0.0, help='metres added to every z target (raises the hand so the cup sits lower in the gripper); 0 = off')\n"),
    ("p[2] = max(p[2], args.zhold if st['grasped'] else 0.027); p[0] += xb",
     "p[2] = max(p[2], args.zhold if st['grasped'] else 0.027) + args.zbias; p[0] += xb"),
]
for old, new in edits:
    if src.count(old) != 1:
        sys.exit('ANCHOR MISMATCH (%d) for: %r -- nothing written' % (src.count(old), old[:60]))
    src = src.replace(old, new)
out = os.path.join(D, 'eval_sim9.py')
open(out, 'w').write(src)
print('wrote', out)
