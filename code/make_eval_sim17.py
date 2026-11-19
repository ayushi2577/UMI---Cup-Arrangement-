"""Create eval_sim17.py = eval_sim16.py + --regrasp (re-grasp when the cup is lost after a grasp) + --wlog (width/z log).
eval_sim16.py is only read. Refuses to write if an anchor is missing or not unique, or if the output exists.
Run from ~/umi_mujoco/sim/umi_sim:  python make_eval_sim17.py

--regrasp N     allow up to N re-grasps per episode (0 = off, default). A cup is "lost" when a grasp was detected, the
                release has not started, and the measured finger width stays below --rg_lost for 5 sim steps in a row
                (fingers closed on nothing). Then: open the gripper, rise --rg_rise m over 20 steps (hold for --rg_steps
                total), clear the grasp state (so the z floor and the grasp detector reset) and hand control back to the policy.
--rg_lost W     width below which the fingers count as empty (default 0.035; holding the cup measures about 0.049).
--rg_rise H     rise during recovery in metres (default 0.04).
--rg_steps S    sim steps of the recovery move (default 40).
--wlog          print 'W <seed> <step> <width> <tcp z> <grasped>' every 10 sim steps.
Prints 'RG <seed> lost at step ... (try k)' for each recovery.
"""
import os, sys

SRC, DST = 'eval_sim16.py', 'eval_sim17.py'

A_FLAG = "ap.add_argument('--trackfix_freeze', action='store_true', help='stop updating the trackfix correction once the grasp is detected')\n"
A_HOLD = "            st['holding_n'] = st['holding_n'] + 1 if 0.03 < w_m < 0.062 else 0\n"

NEW_FLAG = (
    "ap.add_argument('--regrasp', type=int, default=0, help='max re-grasps per episode when the cup is lost after a grasp; 0 = off')\n"
    "ap.add_argument('--rg_lost', type=float, default=0.035, help='finger width below which the fingers count as empty')\n"
    "ap.add_argument('--rg_rise', type=float, default=0.04, help='rise during a re-grasp recovery, metres')\n"
    "ap.add_argument('--rg_steps', type=int, default=40, help='sim steps of the re-grasp recovery move')\n"
    "ap.add_argument('--wlog', action='store_true', help='print width and tcp z every 10 sim steps')\n"
)

NEW_BLOCK = (
    "            if args.wlog and steps % 10 == 0:\n"
    "                print('W %d %d %.3f %.3f %d' % (seed, steps, w_m, float(obs['robot0_eef_pos'][2]), int(st['grasped'])))\n"
    "            if args.regrasp and st['grasped'] and not st.get('release') and st.get('rg_k', 0) < args.regrasp and w_m < args.rg_lost:\n"
    "                st['lost_n'] = st.get('lost_n', 0) + 1\n"
    "            else:\n"
    "                st['lost_n'] = 0\n"
    "            if st['lost_n'] >= 5:\n"
    "                st['rg_k'] = st.get('rg_k', 0) + 1\n"
    "                print('RG %d lost at step %d width %.3f z %.3f (try %d)' % (seed, steps, w_m, float(obs['robot0_eef_pos'][2]), st['rg_k']))\n"
    "                rg_base = np.array(obs['robot0_eef_pos'], dtype=np.float64)\n"
    "                rg_rot = np.array(obs['robot0_eef_rot_axis_angle'], dtype=np.float64)\n"
    "                for rg_j in range(args.rg_steps):\n"
    "                    rg_p = rg_base + np.array([0.0, 0.0, args.rg_rise * min(1.0, (rg_j + 1) / 20.0)])\n"
    "                    q_cmd = ik.solve(rg_p, rg_rot, q_cmd)\n"
    "                    obs = env.step(q_cmd, 0.08, render=True)\n"
    "                    hist.append(obs); steps += 1\n"
    "                st['grasped'] = False; st['holding_n'] = 0; st['lost_n'] = 0\n"
    "                for rg_key in ('c0', 'cl_n', 'gate_n', 'g0', 'g0x', 'tf_pr'):\n"
    "                    st.pop(rg_key, None)\n"
    "                break\n"
)


def main():
    if not os.path.exists(SRC):
        sys.exit('missing ' + SRC + ' (run from ~/umi_mujoco/sim/umi_sim)')
    if os.path.exists(DST):
        sys.exit(DST + ' already exists; refusing to overwrite')
    s = open(SRC).read()
    for a in (A_FLAG, A_HOLD):
        n = s.count(a)
        if n != 1:
            sys.exit('anchor found %d times (need exactly 1), nothing written:\n%s' % (n, a))
    s = s.replace(A_FLAG, A_FLAG + NEW_FLAG)
    s = s.replace(A_HOLD, A_HOLD + NEW_BLOCK)
    compile(s, DST, 'exec')
    open(DST, 'w').write(s)
    print('wrote %s (%d lines); %s untouched' % (DST, s.count('\n'), SRC))


main()
