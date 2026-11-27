"""Builds umi_sim/eval_sim12.py from eval_sim11.py. Refuses to write on anchor mismatch.

New flag (default 0 = eval_sim11 behaviour):
  --zcap_delay K   the --zcap gate may only act after the policy has been commanding closing for
                   more than K sim steps (pre-grasp). In a normal grasp the hand is still
                   descending when closing starts and reaches the window within a few steps;
                   only a hand that is STILL too high after K steps is gated (gripper re-opened,
                   hand target lowered, then the policy closes again = a robot-observable regrasp).
Still proprioception only (TCP height from kinematics + the constant cup rest height).
"""
import os, sys

SRC = os.path.expanduser('~/umi_mujoco/sim/umi_sim/eval_sim11.py')
DST = os.path.expanduser('~/umi_mujoco/sim/umi_sim/eval_sim12.py')
s = open(SRC).read()


def once(text, anchor):
    n = text.count(anchor)
    if n != 1:
        sys.exit('ANCHOR MISMATCH (%d matches), nothing written:\n%s' % (n, anchor))


A1 = "ap.add_argument('--zcap_steps', type=int, default=60, help='max gated steps per episode')"
once(s, A1)
s = s.replace(A1, A1 + "\nap.add_argument('--zcap_delay', type=int, default=0, help='gate acts only after this many closing steps (pre-grasp)')")

A2 = "            if args.zcap >= 0 and not st['grasped'] and gcmd < 0.06 and st.get('gate_n', 0) < args.zcap_steps:\n"
once(s, A2)
B2 = ("            if gcmd < 0.06 and not st['grasped']:\n"
      "                st['cl_n'] = st.get('cl_n', 0) + 1\n"
      "            if args.zcap >= 0 and not st['grasped'] and gcmd < 0.06 and st.get('gate_n', 0) < args.zcap_steps and st.get('cl_n', 0) > args.zcap_delay:\n")
s = s.replace(A2, B2)

open(DST, 'w').write(s)
print('wrote', DST)
