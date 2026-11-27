import os, sys
D = os.path.expanduser('~/umi_mujoco/sim/umi_sim')
src = open(os.path.join(D, 'eval_sim5.py')).read()

old = "    RL['phase'] = 'retreat_open'\n    last_w, stable = -1.0, 0\n"
new = (
    "    RL['phase'] = 'retreat_open'\n"
    "    import os as _os\n"
    "    _u = _os.environ.get('RT_UNLOAD')\n"
    "    if _u:\n"
    "        _p, _rv = env.eef_pose()\n"
    "        _p = _p.copy()\n"
    "        _p[2] += float(_u)\n"
    "        for _ in range(10):\n"
    "            q_cmd = ik.solve(_p, _rv, q_cmd)\n"
    "    last_w, stable = -1.0, 0\n"
)
if src.count(old) != 1:
    sys.exit('ANCHOR MISMATCH, nothing written')
out = os.path.join(D, 'eval_sim6.py')
open(out, 'w').write(src.replace(old, new))
print('wrote', out)
