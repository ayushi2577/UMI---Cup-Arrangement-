"""Builds umi_sim/eval_sim3.py from umi_sim/eval_sim2.py by adding a DIAGNOSTIC-ONLY per-step release trace.
eval_sim.py and eval_sim2.py are never modified. Refuses to write if any anchor does not match exactly once.
New flag: --reltrace   (log every sim step of every episode; ground truth, diagnosis only, never used for control)
Writes <out>/reltrace_seed<N>.csv and prints one 'RELTRACE' summary line per episode."""
from pathlib import Path
import sys

D = Path('~/umi_mujoco/sim/umi_sim').expanduser()
SRC, DST = D / 'eval_sim2.py', D / 'eval_sim3.py'
text = SRC.read_text()

HELPER = '''RL = dict(on=False, rows=[], prev_cup=None, prev_sau=None, prev_tcp=None, phase='policy', n=0)


def rl_log(env, gcmd):
    """DIAGNOSTIC ONLY (ground truth). One row per sim step."""
    if not RL['on']:
        return
    m, d = env.model, env.data
    cup = d.xpos[env.cup_id].copy()
    sau = d.xpos[env.saucer_id].copy()
    tcp = d.site_xpos[env.tcp_id].copy()
    if RL['prev_cup'] is None:
        RL['prev_cup'], RL['prev_sau'], RL['prev_tcp'] = cup, sau, tcp
    cg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'cup_geom')
    root = int(m.body_rootid[m.site_bodyid[env.tcp_id]])
    fn, nc = 0.0, 0
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 == cg or g2 == cg:
            o = g2 if g1 == cg else g1
            if int(m.body_rootid[m.geom_bodyid[o]]) == root:
                mujoco.mj_contactForce(m, d, i, f6)
                fn += abs(float(f6[0]))
                nc += 1
    RL['n'] += 1
    RL['rows'].append(dict(
        n=RL['n'], phase=RL['phase'], width=float(env.gripper_width), gcmd=float(gcmd),
        cup_v=float(np.linalg.norm(cup - RL['prev_cup']) / 0.02),
        sau_v=float(np.linalg.norm(sau - RL['prev_sau']) / 0.02),
        tcp_v=float(np.linalg.norm(tcp - RL['prev_tcp']) / 0.02),
        cup_up=float(d.xmat[env.cup_id].reshape(3, 3)[2, 2]), cup_z=float(cup[2]),
        dx=float(cup[0] - tcp[0]), dy=float(cup[1] - tcp[1]), dz=float(cup[2] - tcp[2]),
        fn=fn, nc=nc))
    RL['prev_cup'], RL['prev_sau'], RL['prev_tcp'] = cup, sau, tcp


def rl_finish(seed, ok, outdir):
    import os, csv
    rows = RL['rows']
    if not RL['on'] or not rows:
        return
    outdir = os.path.expanduser(str(outdir))
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'reltrace_seed%d.csv' % seed), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    pol = [r for r in rows if r['phase'] == 'policy']
    parts = []
    for name, rs in (('pol_tail', pol[-30:]), ('open', [r for r in rows if r['phase'] == 'retreat_open']),
                     ('lift', [r for r in rows if r['phase'] == 'retreat_lift'])):
        if rs:
            parts.append('%s[n=%d cup_v<=%.2f Fn<=%.1f wmax=%.3f tcp_v<=%.3f]' % (
                name, len(rs), max(r['cup_v'] for r in rs), max(r['fn'] for r in rs),
                max(r['width'] for r in rs), max(r['tcp_v'] for r in rs)))
    pk = max(rows, key=lambda r: r['cup_v'])
    print('RELTRACE seed %d %s | %s | peak cup_v %.2f at row %d phase %s width %.3f Fn %.1f tcp_v %.3f' % (
        seed, 'OK  ' if ok else 'FAIL', ' '.join(parts), pk['cup_v'], pk['n'], pk['phase'], pk['width'],
        pk['fn'], pk['tcp_v']))


'''

EDITS = [
    ("ap.add_argument('--retreat', action='store_true'",
     "ap.add_argument('--reltrace', action='store_true', help='DIAGNOSTIC: per-step release trace csv + summary')\n"
     "ap.add_argument('--retreat', action='store_true'"),
    ("def safe_retreat(env, ik, q_cmd, tr=None):", HELPER + "def safe_retreat(env, ik, q_cmd, tr=None):"),
    ("        env.step(q, 0.08, render=False)\n        if tr is not None:\n            tr.update(0.08)\n",
     "        env.step(q, 0.08, render=False)\n        if tr is not None:\n            tr.update(0.08)\n        rl_log(env, 0.08)\n"),
    ("    last_w, stable = -1.0, 0\n", "    RL['phase'] = 'retreat_open'\n    last_w, stable = -1.0, 0\n"),
    ("    pos0, rv0 = env.eef_pose()\n    n_up = 60\n", "    RL['phase'] = 'retreat_lift'\n    pos0, rv0 = env.eef_pose()\n    n_up = 60\n"),
    ("    tr = Tracer(env) if args.trace else None\n    st = dict(holding_n=0, grasped=False)\n",
     "    tr = Tracer(env) if args.trace else None\n    st = dict(holding_n=0, grasped=False)\n"
     "    RL.update(on=bool(args.reltrace), rows=[], prev_cup=None, prev_sau=None, prev_tcp=None, phase='policy', n=0)\n"),
    ("            if tr is not None:\n                tr.update(float(grip[m]))\n            if args.retreat and st['grasped'] and w_m > 0.066:\n",
     "            if tr is not None:\n                tr.update(float(grip[m]))\n            rl_log(env, float(grip[m]))\n            if args.retreat and st['grasped'] and w_m > 0.066:\n"),
    ("    r['retreated'] = bool(st.get('release'))\n",
     "    r['retreated'] = bool(st.get('release'))\n    rl_finish(seed, ok, args.out)\n"),
]

bad = [a[:60] for a, _ in EDITS if text.count(a) != 1]
if bad:
    sys.exit('REFUSING to write: anchor not found exactly once: %s' % bad)
for a, b in EDITS:
    text = text.replace(a, b)
compile(text, 'eval_sim3.py', 'exec')
DST.write_text(text)
print('wrote %s (%d lines). eval_sim.py and eval_sim2.py untouched.' % (DST, len(text.splitlines())))
