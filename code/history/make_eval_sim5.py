import os, sys
D = os.path.expanduser('~/umi_mujoco/sim/umi_sim')
src = open(os.path.join(D, 'eval_sim4.py')).read()

HELPER = '''CD = {}
CDLAB = {66: 'hand_mesh', 69: 'Lfinger_mesh', 77: 'Rfinger_mesh'}
for _i in range(70, 75):
    CDLAB[_i] = 'Ltip%d' % _i
for _i in range(78, 83):
    CDLAB[_i] = 'Rtip%d' % _i


def cdump_step(env):
    m = next(getattr(env, a) for a in ('model', 'm') if hasattr(env, a))
    d = next(getattr(env, a) for a in ('data', 'd') if hasattr(env, a))
    ph = RL['phase']
    f6 = [0.0] * 6
    import numpy as _np
    f6 = _np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if 83 not in (c.geom1, c.geom2):
            continue
        o = c.geom2 if c.geom1 == 83 else c.geom1
        mujoco.mj_contactForce(m, d, i, f6)
        e = CD.setdefault((ph, CDLAB.get(o, 'geom%d' % o)), [0, 0.0])
        e[0] += 1
        e[1] = max(e[1], abs(float(f6[0])))
    if ph == 'retreat_lift' and 'off' not in CD:
        cb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'cup')
        pos, _r = env.eef_pose()
        CD['off'] = (d.xpos[cb] - pos).round(3).tolist()


def cdump_print():
    items = ['%s/%s n=%d F=%.1f' % (k[0][8:], k[1], v[0], v[1])
             for k, v in sorted(CD.items(), key=str) if k != 'off']
    print('CDUMP cup-eef off=%s | %s' % (CD.get('off'), '; '.join(items) or 'no cup contact'))


'''

edits = [
    ("def safe_retreat(env, ik, q_cmd, tr=None):\n", HELPER + "def safe_retreat(env, ik, q_cmd, tr=None):\n"),
    ("        rl_log(env, 0.08)\n    RL['phase'] = 'retreat_open'\n",
     "        rl_log(env, 0.08)\n        cdump_step(env)\n    CD.clear()\n    RL['phase'] = 'retreat_open'\n"),
    ("    for _ in range(50):\n        hold(q_cmd)\n    return q_cmd\n",
     "    for _ in range(50):\n        hold(q_cmd)\n    cdump_print()\n    return q_cmd\n"),
]
for old, new in edits:
    if src.count(old) != 1:
        sys.exit('ANCHOR MISMATCH, nothing written: %r' % old[:50])
    src = src.replace(old, new)
out = os.path.join(D, 'eval_sim5.py')
open(out, 'w').write(src)
print('wrote', out, len(src.splitlines()), 'lines')
