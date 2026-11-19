"""Create eval_sim18.py = eval_sim17.py + --zbias_grasp (a different z bias during the final descent and grasp).
eval_sim17.py is only read. Refuses to write if the anchor is missing or not unique, or if the output exists.
Run from ~/umi_mujoco/sim/umi_sim:  python make_eval_sim18.py

Why: in 120 episodes, grasps detected at TCP z of 38-55 mm succeeded 35-41% vs about 68% below 38 mm, and the hand can never be
commanded below 0.027 + --zbias (the z floor is applied before the bias is added).

--zbias_grasp Z   z bias (metres) used while the policy's target z is below --zb_zone with the gripper open (final descent),
                  and until the grasp is detected. After the grasp it ramps linearly to --zbias over 30 sim steps, so the
                  hand does not jump. Default None = off (behaves exactly like eval_sim17.py).
--zb_zone H       target z (metres) below which the final-descent bias applies (default 0.08).
"""
import os, sys

SRC, DST = 'eval_sim17.py', 'eval_sim18.py'

A_FLAG = "ap.add_argument('--wlog', action='store_true', help='print width and tcp z every 10 sim steps')\n"
A_LINE = "            p = pos[m].copy(); p[2] = max(p[2], args.zhold if st['grasped'] else 0.027) + args.zbias; p[0] += xb\n"

NEW_FLAG = (
    "ap.add_argument('--zbias_grasp', type=float, default=None, help='z bias during the final descent and grasp (metres); None = same as --zbias')\n"
    "ap.add_argument('--zb_zone', type=float, default=0.08, help='target z below which --zbias_grasp applies while the gripper is open')\n"
)

NEW_LINE = (
    "            zb_eff = args.zbias\n"
    "            if args.zbias_grasp is not None:\n"
    "                if (not st['grasped']) and cur['robot0_gripper_width'][0] > 0.06 and pos[m][2] < args.zb_zone:\n"
    "                    st['zg_on'] = True\n"
    "                if st.get('zg_on'):\n"
    "                    if not st['grasped']:\n"
    "                        zb_eff = args.zbias_grasp\n"
    "                    else:\n"
    "                        zg_t0 = st.setdefault('zg_t0', steps)\n"
    "                        zg_k = min(1.0, (steps - zg_t0) / 30.0)\n"
    "                        zb_eff = args.zbias_grasp + (args.zbias - args.zbias_grasp) * zg_k\n"
    "            p = pos[m].copy(); p[2] = max(p[2], args.zhold if st['grasped'] else 0.027) + zb_eff; p[0] += xb\n"
)


def main():
    if not os.path.exists(SRC):
        sys.exit('missing ' + SRC + ' (run make_eval_sim17.py first, from ~/umi_mujoco/sim/umi_sim)')
    if os.path.exists(DST):
        sys.exit(DST + ' already exists; refusing to overwrite')
    s = open(SRC).read()
    for a in (A_FLAG, A_LINE):
        n = s.count(a)
        if n != 1:
            sys.exit('anchor found %d times (need exactly 1), nothing written:\n%s' % (n, a))
    s = s.replace(A_FLAG, A_FLAG + NEW_FLAG)
    s = s.replace(A_LINE, NEW_LINE)
    compile(s, DST, 'exec')
    open(DST, 'w').write(s)
    print('wrote %s (%d lines); %s untouched' % (DST, s.count('\n'), SRC))


main()
