"""Builds umi_sim/eval_sim13.py from eval_sim12.py. Refuses to write on anchor mismatch.

New flags (default 0 = eval_sim12 behaviour):
  --stop_v V      pre-grasp, while the policy commands closing: keep the gripper OPEN as long as the
                  TCP speed (m/s, from kinematics) is above V. Close only when the hand has nearly stopped.
  --stop_wait N   safety: stop holding open after the policy has commanded closing for N steps (no deadlock).
Proprioception only (TCP position differences). No ground truth.
"""
import os, sys

SRC = os.path.expanduser('~/umi_mujoco/sim/umi_sim/eval_sim12.py')
DST = os.path.expanduser('~/umi_mujoco/sim/umi_sim/eval_sim13.py')
s = open(SRC).read()


def once(text, anchor):
    n = text.count(anchor)
    if n != 1:
        sys.exit('ANCHOR MISMATCH (%d matches), nothing written:\n%s' % (n, anchor))


A1 = "ap.add_argument('--zcap_delay', type=int, default=0, help='gate acts only after this many closing steps (pre-grasp)')"
once(s, A1)
s = s.replace(A1, A1 +
              "\nap.add_argument('--stop_v', type=float, default=0.0, help='keep gripper open pre-grasp while TCP speed (m/s) > this; 0 = off')"
              "\nap.add_argument('--stop_wait', type=int, default=30, help='max policy-closing steps to hold open for --stop_v')")

A2 = "            if args.zcap >= 0 and not st['grasped'] and gcmd < 0.06 and st.get('gate_n', 0) < args.zcap_steps and st.get('cl_n', 0) > args.zcap_delay:\n"
once(s, A2)
B2 = ("            _tcp = env.data.site_xpos[env.tcp_id].copy()\n"
      "            _sp = float(np.linalg.norm(_tcp - st['ptcp']) / 0.02) if 'ptcp' in st else 0.0\n"
      "            st['ptcp'] = _tcp\n"
      "            if args.stop_v > 0 and not st['grasped'] and gcmd < 0.06 and st.get('cl_n', 0) <= args.stop_wait and _sp > args.stop_v:\n"
      "                gcmd = 0.08\n"
      "                st['stop_n'] = st.get('stop_n', 0) + 1\n")
s = s.replace(A2, B2 + A2)

open(DST, 'w').write(s)
print('wrote', DST)
