"""Builds umi_sim/eval_sim2.py from umi_sim/eval_sim.py by exact-text patching.

eval_sim.py is NOT modified. If any anchor text is missing or ambiguous the script stops
without writing anything, so a wrong/edited eval_sim.py can never produce a half-patched file.

New options in eval_sim2.py (all default to the old behaviour):
  --trace              diagnostics only: classifies every episode (never lifted / slipped from the closed
                       gripper / released far from saucer / released on saucer then disturbed ...).
                       Uses simulator ground truth for EXPLAINING failures, never for control.
  --xbias_mode pregrasp  apply --xbias only until the grasp is detected (gripper width held in 3-6.2 cm
                       for 15 sim steps). 'always' (default) = old behaviour, bias also during release/retreat.
  --glead L            lead (sim steps) for the GRIPPER channel only; default = --lead (old behaviour).
  --deterministic      force deterministic torch kernels so identical runs give identical results
  --retreat            robot-observable release handling: when the measured gripper width rises above
                       6.6 cm after a detected grasp, stop the policy, open fully, wait for the width to
                       settle, rise straight up 10 cm, hold. Uses only gripper width and end-effector pose.
"""
import os, sys

SRC = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/umi_mujoco/sim/umi_sim/eval_sim.py')
DST = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else '~/umi_mujoco/sim/umi_sim/eval_sim2.py')
src = open(SRC).read()


def sub(old, new):
    global src
    n = src.count(old)
    if n != 1:
        sys.exit('PATCH FAILED (nothing written): expected exactly 1 match, found %d for:\n%s' % (n, old))
    src = src.replace(old, new)


# 1. new command-line options ------------------------------------------------------------
sub("args = ap.parse_args()\n", """ap.add_argument('--trace', action='store_true', help='diagnostics: classify each episode (ground truth, never used for control)')
ap.add_argument('--xbias_mode', choices=['always', 'pregrasp'], default='always', help='pregrasp: drop the x bias once the grasp is detected')
ap.add_argument('--glead', type=float, default=None, help='gripper-channel lead in sim steps (default: same as --lead)')
ap.add_argument('--deterministic', action='store_true', help='force deterministic torch/CUDA kernels (slower) so repeated runs match')
ap.add_argument('--retreat', action='store_true', help='robot-observable open / wait / vertical retreat after a detected release')
args = ap.parse_args()
if args.glead is None:
    args.glead = args.lead
""")

# 2. separate lead for the gripper channel -----------------------------------------------------
sub("    return pos, sl(ms).as_rotvec(), np.interp(ms, times, G)\n",
    "    return pos, sl(ms).as_rotvec(), np.interp(ms, times + (args.lead - args.glead), G)\n")

# 2b. optional deterministic kernels ---------------------------------------------------------
sub("device = torch.device('cuda:0')\n", """if args.deterministic:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
device = torch.device('cuda:0')
""")

# 3. tracer + safe retreat, defined just before rollout() ----------------------------------------
HELPERS = '''class Tracer:
    """DIAGNOSTIC ONLY. Reads simulator ground truth to explain failures; never feeds back into control."""
    def __init__(self, env):
        self.env = env
        m = env.model
        self.robot_root = int(m.body_rootid[m.site_bodyid[env.tcp_id]])
        self.cup_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'cup_geom')
        self.sau_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'saucer_geom')
        self.sau_xy0 = env.data.xpos[env.saucer_id][:2].copy()
        self.sau_touch_before = 0
        self.sau_touch_after = 0
        self.n = 0
        self.lifted = False
        self.was_held = False
        self.not_held_n = 0
        self.cand = None
        self.sep = None
        self.n_sep = 0
        self.touch_after_open = 0
        self.xy_settle = None
        self.prev_cup = env.data.xpos[env.cup_id].copy()

    def robot_touch(self, gid):
        d, m = self.env.data, self.env.model
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == gid or g2 == gid:
                o = g2 if g1 == gid else g1
                if int(m.body_rootid[m.geom_bodyid[o]]) == self.robot_root:
                    return True
        return False

    def update(self, gcmd):
        env, d = self.env, self.env.data
        self.n += 1
        cup = d.xpos[env.cup_id].copy()
        sau = d.xpos[env.saucer_id]
        tcp = d.site_xpos[env.tcp_id]
        w = env.gripper_width
        touch = self.robot_touch(self.cup_g)
        if self.robot_touch(self.sau_g):      # a finger / the hand touches the saucer
            if self.sep is None:
                self.sau_touch_before += 1
            else:
                self.sau_touch_after += 1
        held = bool(touch and w < 0.06 and np.linalg.norm(cup - tcp) < 0.06)
        speed = float(np.linalg.norm(cup - self.prev_cup) / 0.02)
        self.prev_cup = cup
        if cup[2] > 0.06:
            self.lifted = True
        if self.lifted:
            if held:
                self.was_held, self.not_held_n, self.cand = True, 0, None
            elif self.was_held:
                self.not_held_n += 1
                if self.not_held_n == 1:
                    self.cand = dict(step=self.n, dxy=float(np.linalg.norm(cup[:2] - sau[:2])), cup_z=float(cup[2]),
                                     tcp_z=float(tcp[2]), width=float(w), gcmd=float(gcmd),
                                     cmd_open=bool(gcmd >= 0.06), speed=speed)
                if self.not_held_n == 5:      # debounce: really left the gripper
                    self.n_sep += 1
                    self.was_held = False
                    if self.sep is None:
                        self.sep = self.cand
        if self.sep is not None:
            if touch and w > 0.07:
                self.touch_after_open += 1     # fingers/hand touching the cup although the gripper is open
            if self.n == self.sep['step'] + 40:
                self.xy_settle = cup[:2].copy()

    def finish(self, r):
        cup = self.env.data.xpos[self.env.cup_id]
        drift = float(np.linalg.norm(cup[:2] - self.xy_settle)) if self.xy_settle is not None else 0.0
        s = self.sep
        sau_moved = float(np.linalg.norm(self.env.data.xpos[self.env.saucer_id][:2] - self.sau_xy0))
        if r['ok']:
            cat = 'success'
        elif r['max_cup_z'] < 0.06:
            cat = 'never_lifted'
        elif s is None:
            cat = 'held_at_end_or_never_left_gripper'
        elif not s['cmd_open']:
            cat = 'slipped_from_closed_grip_' + ('near_saucer' if s['dxy'] < 0.05 else 'far_from_saucer')
        elif s['dxy'] > 0.05:
            cat = 'released_far_from_saucer'
        elif sau_moved > 0.03:
            cat = 'released_on_saucer_then_SAUCER_PUSHED_AWAY'
        elif self.touch_after_open > 0 or drift > 0.01:
            cat = 'released_on_saucer_then_disturbed'
        else:
            cat = 'released_on_saucer_undisturbed_but_failed'
        return dict(cat=cat, sep=s, n_sep=int(self.n_sep), touch_after_open=int(self.touch_after_open),
                    drift_after_release=drift, saucer_moved=sau_moved,
                    sau_touch_before=int(self.sau_touch_before), sau_touch_after=int(self.sau_touch_after))


def safe_retreat(env, ik, q_cmd, tr=None):
    """Uses only gripper width and end-effector pose. Stop the policy, open fully while holding the pose,
    wait until the finger width stops changing, rise straight up 10 cm (smooth), then hold."""
    def hold(q):
        env.step(q, 0.08, render=False)
        if tr is not None:
            tr.update(0.08)
    last_w, stable = -1.0, 0
    for _ in range(75):
        hold(q_cmd)
        w = env.gripper_width
        stable = stable + 1 if abs(w - last_w) < 3e-4 else 0
        last_w = w
        if w > 0.07 and stable >= 5:
            break
    pos0, rv0 = env.eef_pose()
    n_up = 60
    for i in range(1, n_up + 1):
        s = i / n_up
        s = s * s * (3 - 2 * s)
        p = pos0.copy()
        p[2] = pos0[2] + 0.10 * s
        q_cmd = ik.solve(p, rv0, q_cmd)
        hold(q_cmd)
    for _ in range(50):
        hold(q_cmd)
    return q_cmd


def rollout(env, policy, cfg, ik, seed, device, record=False):
'''
sub("def rollout(env, policy, cfg, ik, seed, device, record=False):\n", HELPERS)

# 4. rollout state ---------------------------------------------------------------------------
sub("    q_cmd, steps, max_cup_z, frames = env.arm_q, 0, 0.0, []\n",
    "    q_cmd, steps, max_cup_z, frames = env.arm_q, 0, 0.0, []\n"
    "    tr = Tracer(env) if args.trace else None\n"
    "    st = dict(holding_n=0, grasped=False)\n")

# 5. x-bias mode -------------------------------------------------------------------------------
sub("        xb = args.xbias if cur['robot0_gripper_width'][0] > 0.06 else 0.0\n",
    "        xb = args.xbias if cur['robot0_gripper_width'][0] > 0.06 else 0.0\n"
    "        if args.xbias_mode == 'pregrasp' and st['grasped']:\n"
    "            xb = 0.0\n")

# 6. per-step bookkeeping, grasp / release detection ---------------------------------------------
sub("            max_cup_z = max(max_cup_z, float(env.get_object_state()['cup_pos'][2]))\n",
    "            max_cup_z = max(max_cup_z, float(env.get_object_state()['cup_pos'][2]))\n"
    "            w_m = float(obs['robot0_gripper_width'][0])\n"
    "            st['holding_n'] = st['holding_n'] + 1 if 0.03 < w_m < 0.062 else 0\n"
    "            if st['holding_n'] >= 15:\n"
    "                st['grasped'] = True\n"
    "            if tr is not None:\n"
    "                tr.update(float(grip[m]))\n"
    "            if args.retreat and st['grasped'] and w_m > 0.066:\n"
    "                st['release'] = True\n"
    "                break\n")

# 7. leave the policy loop on release, run the retreat, then the usual success check ---------------
sub("            if steps >= args.max_steps:\n                break\n    ok, dxy, cz = ScriptedExpert(env).success()\n",
    "            if steps >= args.max_steps:\n                break\n"
    "        if st.get('release'):\n            break\n"
    "    if st.get('release'):\n        q_cmd = safe_retreat(env, ik, q_cmd, tr)\n"
    "    ok, dxy, cz = ScriptedExpert(env).success()\n")

# 8. richer result record -----------------------------------------------------------------------
sub("    return dict(seed=seed, ok=bool(ok), dxy=float(dxy), cup_z=float(cz), up=up, max_cup_z=max_cup_z, steps=steps), frames\n",
    "    cup_p, tcp_p = env.data.xpos[env.cup_id], env.data.site_xpos[env.tcp_id]\n"
    "    r = dict(seed=seed, ok=bool(ok), dxy=float(dxy), cup_z=float(cz), up=up, max_cup_z=max_cup_z, steps=steps)\n"
    "    r['held_end'] = bool(cup_p[2] > 0.06 and np.linalg.norm(cup_p - tcp_p) < 0.06)\n"
    "    r['retreated'] = bool(st.get('release'))\n"
    "    if tr is not None:\n"
    "        r['diag'] = tr.finish(r)\n"
    "    return r, frames\n")

# 9. per-seed print + summary ---------------------------------------------------------------------
sub("        seed, 'OK  ' if r['ok'] else 'FAIL', r['dxy'], r['cup_z'], r['max_cup_z'], r['up'], time.time() - t0))\n",
    "        seed, 'OK  ' if r['ok'] else 'FAIL', r['dxy'], r['cup_z'], r['max_cup_z'], r['up'], time.time() - t0))\n"
    "    if 'diag' in r:\n"
    "        s = r['diag']['sep']\n"
    "        print('      -> %s%s | finger/hand touches after open: %d, cup drift after release: %.3f m | saucer moved %.3f m, finger-saucer touch steps before/after release: %d/%d' % (r['diag']['cat'], '' if s is None else ' | left gripper at step %d, %.3f m from saucer, z %.3f, cmd %.3f' % (\n"
    "            s['step'], s['dxy'], s['cup_z'], s['gcmd']), r['diag']['touch_after_open'], r['diag']['drift_after_release'], r['diag']['saucer_moved'], r['diag']['sau_touch_before'], r['diag']['sau_touch_after']))\n")

sub("print('cup lifted above 6 cm in %d/%d episodes' % (sum(r['max_cup_z'] > 0.06 for r in results), len(results)))\n",
    "print('cup lifted above 6 cm in %d/%d episodes' % (sum(r['max_cup_z'] > 0.06 for r in results), len(results)))\n"
    "print('strict success (cup not still held at the end): %d/%d' % (sum(r['ok'] and not r['held_end'] for r in results), len(results)))\n"
    "if args.trace:\n"
    "    print('outcome breakdown:')\n"
    "    for k, v in collections.Counter(r['diag']['cat'] for r in results).most_common():\n"
    "        print('  %-45s %d' % (k, v))\n")

if os.path.exists(DST):
    bak = DST + '.bak'
    os.replace(DST, bak)
    print('existing %s moved to %s' % (DST, bak))
open(DST, 'w').write(src)
print('wrote', DST, '(%d lines). Original %s untouched.' % (src.count('\n'), SRC))
