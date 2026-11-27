"""Builds umi_sim/eval_sim11.py from eval_sim10.py. Refuses to write on anchor mismatch.

New flags (defaults = old behaviour):
  --zcap C        metres. If >= 0: while the policy commands closing and no grasp has been
                  detected yet, and the TCP is more than C above the cup resting height
                  (0.03 m, a scene constant), keep the gripper open (0.08) and lower the hand
                  target until the TCP is within C. Never lowers the eef target below 0.027
                  (the old floor, i.e. at worst the pre-zbias behaviour). Uses only robot
                  proprioception (TCP height from forward kinematics) + a constant. No object
                  ground truth is used for control.
  --zcap_steps N  max sim steps the gate may hold the gripper open per episode (60).

Logging (diagnosis): GTX line per episode: tcp z at first closing command (c0), tcp z and eef z
at grasp detection, the target z, and number of gated steps.
"""
import os, sys

SRC = os.path.expanduser('~/umi_mujoco/sim/umi_sim/eval_sim10.py')
DST = os.path.expanduser('~/umi_mujoco/sim/umi_sim/eval_sim11.py')
s = open(SRC).read()


def once(text, anchor):
    n = text.count(anchor)
    if n != 1:
        sys.exit('ANCHOR MISMATCH (%d matches), nothing written:\n%s' % (n, anchor))


# 1) arguments
A1 = "help='pregrasp: drop the x bias once the grasp is detected')"
once(s, A1)
s = s.replace(A1, A1 + """
ap.add_argument('--zcap', type=float, default=-1.0, help='metres; >=0 enables the grasp-height gate (see make_eval_sim11.py)')
ap.add_argument('--zcap_steps', type=int, default=60, help='max gated steps per episode')""")

# 2) gate: after the target line, before the IK call
A2 = "p[2] = max(p[2], args.zhold if st['grasped'] else 0.027) + args.zbias; p[0] += xb\n"
once(s, A2)
G = """            gcmd = float(np.clip(grip[m], 0.0, 0.08))
            if gcmd < 0.06 and not st['grasped'] and 'c0' not in st:
                st['c0'] = float(env.data.site_xpos[env.tcp_id][2])
            if args.zcap >= 0 and not st['grasped'] and gcmd < 0.06 and st.get('gate_n', 0) < args.zcap_steps:
                tcp_z = float(env.data.site_xpos[env.tcp_id][2])
                excess = tcp_z - 0.03 - args.zcap
                if excess > 0.0 and tcp_z < 0.075:
                    eef_z = float(obs['robot0_eef_pos'][2])
                    p[2] = max(min(p[2], eef_z - excess), 0.027)
                    gcmd = 0.08
                    st['gate_n'] = st.get('gate_n', 0) + 1
"""
s = s.replace(A2, A2 + G)

# 3) use gcmd in the env step
A3 = "obs = env.step(q_cmd, float(np.clip(grip[m], 0.0, 0.08)), render=True)"
once(s, A3)
s = s.replace(A3, "obs = env.step(q_cmd, gcmd, render=True)")

# 4) extra snapshot at grasp detection
A4 = "                    st['g0'] = gt_snap(env, w_m)\n"
once(s, A4)
s = s.replace(A4, A4 + "                    st['g0x'] = dict(c0=st.get('c0'), tcp=float(env.data.site_xpos[env.tcp_id][2]), eef=float(obs['robot0_eef_pos'][2]), tgt=float(p[2]))\n")

# 5) store in result
A5 = "gt=dict(g0=st.get('g0'), g1=st.get('g1'), maxsep=st.get('maxsep'))"
once(s, A5)
s = s.replace(A5, "gt=dict(g0=st.get('g0'), g1=st.get('g1'), maxsep=st.get('maxsep'), g0x=st.get('g0x'), gate=st.get('gate_n', 0))")

# 6) GTX lines before the final summary print
A6 = "print('cup lifted above 6 cm in %d/%d episodes' % (sum(r['max_cup_z'] > 0.06 for r in results), len(results)))"
once(s, A6)
H6 = """for r in results:
    g = r.get('gt', {})
    x = g.get('g0x') or {}
    print('GTX %3d %-17s c0=%s tcp=%s eef=%s tgt=%s gate=%d' % (
        r['seed'], _grp(r),
        ('%.4f' % x['c0']) if x.get('c0') is not None else 'none',
        ('%.4f' % x['tcp']) if 'tcp' in x else 'none',
        ('%.4f' % x['eef']) if 'eef' in x else 'none',
        ('%.4f' % x['tgt']) if 'tgt' in x else 'none', g.get('gate', 0)))
_gated = [r for r in results if r.get('gt', {}).get('gate', 0) > 0]
print('GATED episodes: %d, of which success %d' % (len(_gated), sum(r['ok'] for r in _gated)))
"""
s = s.replace(A6, H6 + A6)

open(DST, 'w').write(s)
print('wrote', DST)
