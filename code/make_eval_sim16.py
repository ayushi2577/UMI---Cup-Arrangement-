"""Create eval_sim16.py = eval_sim15.py + --trackfix (robot-observable tracking-error correction).
eval_sim15.py and eval_sim12.py are only read. Refuses to write if an anchor is missing or not unique, or if the output exists.
Run from ~/umi_mujoco/sim/umi_sim:  python make_eval_sim16.py

--trackfix K       integral gain per sim step (0 = off, default). Each sim step: c += K * (desired_target - measured_tcp), clipped.
                   The target sent to IK is desired_target + c. Uses only the commanded target and the measured tcp pose.
--trackfix_max M   clip on |c| per axis in metres (default 0.02).
--trackfix_freeze  stop updating c once the grasp is detected.
Prints one line per episode when the grasp is detected:  TF <seed> x <mm> y <mm> z <mm>  (the correction c at that moment).
"""
import os, sys

SRC, DST = 'eval_sim15.py', 'eval_sim16.py'

A_FLAG = "ap.add_argument('--zbias', type=float, default=0.0, help='metres added to every z target (raises the hand so the cup sits lower in the gripper); 0 = off')\n"
A_ST = "    st = dict(holding_n=0, grasped=False)\n"
A_IK = "            q_cmd = ik.solve(p, rot[m], q_cmd)\n            obs = env.step(q_cmd, gcmd, render=True)\n"

NEW_FLAG = (
    "ap.add_argument('--trackfix', type=float, default=0.0, help='integral gain per sim step on (desired target - measured tcp), added to the IK target; 0 = off')\n"
    "ap.add_argument('--trackfix_max', type=float, default=0.02, help='clip on the trackfix correction per axis, metres')\n"
    "ap.add_argument('--trackfix_freeze', action='store_true', help='stop updating the trackfix correction once the grasp is detected')\n"
)

NEW_ST = "    tf_c = np.zeros(3)\n"

NEW_IK = (
    "            p_des = p.copy()\n"
    "            if args.trackfix > 0:\n"
    "                p = p + tf_c\n"
    "            q_cmd = ik.solve(p, rot[m], q_cmd)\n"
    "            obs = env.step(q_cmd, gcmd, render=True)\n"
    "            if args.trackfix > 0 and not (args.trackfix_freeze and st['grasped']):\n"
    "                tf_c = np.clip(tf_c + args.trackfix * (p_des - np.asarray(obs['robot0_eef_pos'], dtype=np.float64)), -args.trackfix_max, args.trackfix_max)\n"
    "            if args.trackfix > 0 and st['grasped'] and 'tf_pr' not in st:\n"
    "                st['tf_pr'] = 1\n"
    "                print('TF %d x %.1f y %.1f z %.1f' % (seed, 1000 * tf_c[0], 1000 * tf_c[1], 1000 * tf_c[2]))\n"
)


def main():
    if not os.path.exists(SRC):
        sys.exit('missing ' + SRC + ' (run make_eval_sim15.py first, from ~/umi_mujoco/sim/umi_sim)')
    if os.path.exists(DST):
        sys.exit(DST + ' already exists; refusing to overwrite')
    s = open(SRC).read()
    for a in (A_FLAG, A_ST, A_IK):
        n = s.count(a)
        if n != 1:
            sys.exit('anchor found %d times (need exactly 1), nothing written:\n%s' % (n, a))
    s = s.replace(A_FLAG, A_FLAG + NEW_FLAG)
    s = s.replace(A_ST, A_ST + NEW_ST)
    s = s.replace(A_IK, NEW_IK)
    compile(s, DST, 'exec')
    open(DST, 'w').write(s)
    print('wrote %s (%d lines); %s untouched' % (DST, s.count('\n'), SRC))


main()
