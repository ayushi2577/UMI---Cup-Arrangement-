"""Makes eval_sim19.py from eval_sim18.py.

Adds ONE option: --video_full (off by default). When off, eval_sim19.py is the same
program as eval_sim18.py. The video code is copied from eval_sim14.py (wraps env.step so
frames are also recorded during the release and the retreat).

Safety: eval_sim18.py is only read. If eval_sim19.py already exists, or any anchor text
is not found exactly once, nothing is written.
"""
import os
import sys

d = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(d, 'eval_sim18.py')
dst = os.path.join(d, 'eval_sim19.py')

if not os.path.exists(src):
    sys.exit('eval_sim18.py not found next to this script: ' + src)
if os.path.exists(dst):
    sys.exit('eval_sim19.py already exists, refusing to overwrite it')

s = open(src).read()

for name in ('_vn', '_vstep', '_wrapped', 'video_full'):
    if name in s:
        sys.exit('name already present in eval_sim18.py: ' + name + ' (nothing written)')

edits = []

# 1. new command-line option
edits.append((
    "ap.add_argument('--videos', type=int, default=0)",
    "ap.add_argument('--videos', type=int, default=0)\n"
    "ap.add_argument('--video_full', action='store_true', help='record release/retreat frames too (wraps env.step)')",
))

# 2. wrap env.step at the start of rollout (same code as eval_sim14.py)
edits.append((
    "obs = env.reset(seed=seed)\n"
    "    hist = collections.deque([obs] * 4, maxlen=4)",
    "obs = env.reset(seed=seed)\n"
    "    _vn = [0]\n"
    "    if record and args.video_full:\n"
    "        _vstep = env.step\n"
    "        def _wrapped(*a, **k):\n"
    "            r_ = _vstep(*a, **k)\n"
    "            _vn[0] += 1\n"
    "            if _vn[0] % 2 == 0:\n"
    "                frames.append(env.render('front', 360, 480))\n"
    "            return r_\n"
    "        env.step = _wrapped\n"
    "    hist = collections.deque([obs] * 4, maxlen=4)",
))

# 3. do not record the old, shorter way when video_full is on (avoids doubled frames)
edits.append((
    "if record and steps % 2 == 0:",
    "if record and not args.video_full and steps % 2 == 0:",
))

# 4. put env.step back to normal before returning
edits.append((
    "r['diag'] = tr.finish(r)\n"
    "    return r, frames",
    "r['diag'] = tr.finish(r)\n"
    "    if record and args.video_full:\n"
    "        del env.step\n"
    "    return r, frames",
))

# check every anchor first
for i, (old, new) in enumerate(edits, 1):
    n = s.count(old)
    if n != 1:
        sys.exit('anchor %d found %d times (need exactly 1). Nothing written.\n--- anchor ---\n%s' % (i, n, old))

for old, new in edits:
    s = s.replace(old, new)

# the result must be valid Python before we save it
try:
    compile(s, dst, 'exec')
except SyntaxError as e:
    sys.exit('patched file would have a syntax error, nothing written: %s' % e)

open(dst, 'w').write(s)
print('wrote', dst, '(%d lines)' % (s.count('\n') + 1))
print('eval_sim18.py was not changed.')
