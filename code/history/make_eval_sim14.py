"""Builds umi_sim/eval_sim14.py from eval_sim12.py. Refuses to write on anchor mismatch.

New flag (default off = eval_sim12 behaviour):
  --video_full   for episodes recorded with --videos: capture a frame on every 2nd env.step for the
                 WHOLE episode, including the release, retreat and post-retreat steps that run inside
                 hold() (eval_sim12 stops capturing at release detection). Recording only; the
                 controller is unchanged.
"""
import os, sys

SRC = os.path.expanduser('~/umi_mujoco/sim/umi_sim/eval_sim12.py')
DST = os.path.expanduser('~/umi_mujoco/sim/umi_sim/eval_sim14.py')
s = open(SRC).read()


def once(text, anchor):
    n = text.count(anchor)
    if n != 1:
        sys.exit('ANCHOR MISMATCH (%d matches), nothing written:\n%s' % (n, anchor))


A1 = "ap.add_argument('--videos', type=int, default=0)\n"
once(s, A1)
s = s.replace(A1, A1 + "ap.add_argument('--video_full', action='store_true', help='record release/retreat frames too (wraps env.step)')\n")

A2 = "    obs = env.reset(seed=seed)\n"
once(s, A2)
B2 = ("    _vn = [0]\n"
      "    if record and args.video_full:\n"
      "        _vstep = env.step\n"
      "        def _wrapped(*a, **k):\n"
      "            r_ = _vstep(*a, **k)\n"
      "            _vn[0] += 1\n"
      "            if _vn[0] % 2 == 0:\n"
      "                frames.append(env.render('front', 360, 480))\n"
      "            return r_\n"
      "        env.step = _wrapped\n")
s = s.replace(A2, A2 + B2)

A3 = "            if record and steps % 2 == 0:\n"
once(s, A3)
s = s.replace(A3, "            if record and not args.video_full and steps % 2 == 0:\n")

A4 = "    return r, frames\n"
once(s, A4)
s = s.replace(A4, "    if record and args.video_full:\n        del env.step\n" + A4)

open(DST, 'w').write(s)
print('wrote', DST)
