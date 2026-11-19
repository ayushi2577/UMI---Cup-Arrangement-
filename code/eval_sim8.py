"""Evaluate a trained UMI policy in the MuJoCo sim.
  python eval_sim.py --ckpt CKPT --check 30         # plumbing check on recorded demos, no rollouts
  python eval_sim.py --ckpt CKPT --n 100 --expert   # closed-loop rollouts on unseen seeds 0..n-1
"""
import os, sys, json, time, argparse, collections
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.expanduser('~/umi_mujoco/universal_manipulation_interface')
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, REPO)

import numpy as np, torch, dill, hydra, mujoco, zarr, imageio
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R, Slerp

OmegaConf.register_new_resolver('eval', eval, replace=True)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
register_codecs()
from umi.real_world.real_inference_util import get_real_umi_obs_dict, get_real_umi_action
from umi_sim.scene import build_scene
from umi_sim.env import UmiSimEnv
from umi_sim.expert import ScriptedExpert

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--menagerie', default=os.path.expanduser('~/umi_mujoco/sim/mujoco_menagerie'))
ap.add_argument('--data', default=os.path.expanduser('~/umi_mujoco/sim_demos/cup_sim_200.zarr'))
ap.add_argument('--check', type=int, default=0, help='N recorded samples for the plumbing check')
ap.add_argument('--n', type=int, default=100)
ap.add_argument('--start_seed', type=int, default=0)
ap.add_argument('--n_exec', type=int, default=8, help='action steps executed per inference')
ap.add_argument('--shift', type=int, default=0, help='action k is reached at sim step 3*(k+shift)')
ap.add_argument('--max_steps', type=int, default=480)
ap.add_argument('--videos', type=int, default=0)
ap.add_argument('--expert', action='store_true', help='also run the scripted expert on the same seeds')
ap.add_argument('--out', default=os.path.expanduser('~/umi_mujoco/eval_results'))
ap.add_argument('--squeeze', type=float, default=0.06, help='commanded widths below this squeeze fully')
ap.add_argument('--lead', type=float, default=0.0, help='sim steps to command targets earlier, to offset arm lag')
ap.add_argument('--xbias', type=float, default=0.0, help='metres added to x targets while the gripper is open')
ap.add_argument('--trace', action='store_true', help='diagnostics: classify each episode (ground truth, never used for control)')
ap.add_argument('--xbias_mode', choices=['always', 'pregrasp'], default='always', help='pregrasp: drop the x bias once the grasp is detected')
ap.add_argument('--rt_trig', type=float, default=0.066, help='width that triggers retreat after a grasp (default 0.066 = old)')
ap.add_argument('--rt_full', type=float, default=0.07, help='width the open-wait needs before it may exit (default 0.07 = old)')
ap.add_argument('--rt_back', type=float, default=0.0, help='metres to move in x after the rise (negative = toward robot base); 0 = off')
ap.add_argument('--zhold', type=float, default=0.027, help='minimum target z while the cup is held (default 0.027 = old clamp)')
ap.add_argument('--glead', type=float, default=None, help='gripper-channel lead in sim steps (default: same as --lead)')
ap.add_argument('--deterministic', action='store_true', help='force deterministic torch/CUDA kernels (slower) so repeated runs match')
ap.add_argument('--reltrace', action='store_true', help='DIAGNOSTIC: per-step release trace csv + summary')
ap.add_argument('--retreat', action='store_true', help='robot-observable open / wait / vertical retreat after a detected release')
ap.add_argument('--rt_hold', type=int, default=0, help='extra sim steps to hold fully open (pose fixed) before rising')
ap.add_argument('--rt_nup', type=int, default=60, help='number of sim steps for the 10 cm rise (more = slower)')
ap.add_argument('--rt_rise', type=float, default=0.10, help='rise height in metres')
args = ap.parse_args()
if args.glead is None:
    args.glead = args.lead
import umi_sim.env as _envmod
_envmod.SQUEEZE_BELOW = args.squeeze


def load_policy(ckpt, device):
    payload = torch.load(open(ckpt, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    ws = cls(cfg)
    try:
        ws.load_payload(payload, exclude_keys=None, include_keys=None)
    except Exception as e:
        print('load_payload failed (%s: %s); loading state dicts directly' % (type(e).__name__, e))
        ws.model.load_state_dict(payload['state_dicts']['model'])
        if cfg.training.use_ema:
            ws.ema_model.load_state_dict(payload['state_dicts']['ema_model'])
    policy = ws.ema_model if cfg.training.use_ema else ws.model
    policy.num_inference_steps = 16
    policy.eval().to(device)
    return policy, cfg


@torch.no_grad()
def predict(policy, cfg, env_obs, start_pose, device):
    """env_obs: dict of 2-frame histories (oldest first). Returns (16,7): world-frame [pos, rotvec, gripper]."""
    policy.reset()
    obs_np = get_real_umi_obs_dict(
        env_obs=env_obs, shape_meta=cfg.task.shape_meta,
        obs_pose_repr=cfg.task.pose_repr.obs_pose_repr,
        tx_robot1_robot0=None, episode_start_pose=[start_pose])
    obs_t = dict_apply(obs_np, lambda x: torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(device))
    res = policy.predict_action(obs_t)
    a = res['action_pred'][0].detach().cpu().numpy()
    return get_real_umi_action(a, env_obs, cfg.task.pose_repr.action_pose_repr)


def build_env_obs(frames):
    return {
        'camera0_rgb': np.stack([f['camera0_rgb'] for f in frames]),
        'robot0_eef_pos': np.stack([f['robot0_eef_pos'] for f in frames]).astype(np.float64),
        'robot0_eef_rot_axis_angle': np.stack([f['robot0_eef_rot_axis_angle'] for f in frames]).astype(np.float64),
        'robot0_gripper_width': np.stack([f['robot0_gripper_width'] for f in frames]).astype(np.float64),
    }


class PoseIK:
    """Damped least-squares IK for the TCP site, position + orientation."""
    def __init__(self, env):
        m = env.model
        self.env, self.m, self.d = env, m, mujoco.MjData(m)
        hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
        hj = [j for j in range(m.njnt) if int(m.jnt_type[j]) == hinge]
        self.qadr, self.dofadr = m.jnt_qposadr[hj], m.jnt_dofadr[hj]
        self.lo, self.hi = m.jnt_range[hj, 0], m.jnt_range[hj, 1]

    def solve(self, pos, rotvec, q_init, iters=30):
        m, d, tcp = self.m, self.d, self.env.tcp_id
        Rt = R.from_rotvec(rotvec).as_matrix()
        d.qpos[:] = self.env.data.qpos
        d.qpos[self.qadr] = q_init
        for _ in range(iters):
            mujoco.mj_forward(m, d)
            p = d.site_xpos[tcp].copy()
            M = d.site_xmat[tcp].reshape(3, 3).copy()
            e_p = pos - p
            e_r = R.from_matrix(Rt @ M.T).as_rotvec()
            if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_r) < 1e-3:
                break
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, tcp)
            J = np.vstack([jp[:, self.dofadr], jr[:, self.dofadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.concatenate([e_p, e_r]))
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        return d.qpos[self.qadr].copy()


def interp_plan(act, cur_pos, cur_rot, cur_grip, shift, M):
    """Per-sim-step targets for steps 1..M. Action k is reached at sim step 3*(k+shift)."""
    times, P, RV, G = [], [], [], []
    if shift > 0:
        times.append(0.0); P.append(cur_pos); RV.append(cur_rot); G.append(float(cur_grip))
    for k in range(len(act)):
        times.append(3.0 * (k + shift) - args.lead); P.append(act[k, :3]); RV.append(act[k, 3:6]); G.append(act[k, 6])
    times, P, G = np.array(times), np.array(P), np.array(G)
    sl = Slerp(times, R.from_rotvec(np.array(RV)))
    ms = np.arange(1, M + 1).astype(float)
    pos = np.stack([np.interp(ms, times, P[:, i]) for i in range(3)], axis=1)
    return pos, sl(ms).as_rotvec(), np.interp(ms, times + (args.lead - args.glead), G)


class Tracer:
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


RL = dict(on=False, rows=[], prev_cup=None, prev_sau=None, prev_tcp=None, phase='policy', n=0)


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


CD = {}
CDLAB = {66: 'hand_mesh', 69: 'Lfinger_mesh', 77: 'Rfinger_mesh'}
for _i in range(70, 75):
    CDLAB[_i] = 'Ltip%d' % _i
for _i in range(78, 83):
    CDLAB[_i] = 'Rtip%d' % _i


def cdump_step(env):
    m = next(getattr(env, a) for a in ('model', 'm') if hasattr(env, a))
    d = next(getattr(env, a) for a in ('data', 'd') if hasattr(env, a))
    ph = RL['phase']
    f6 = [0.0] * 6
    import numpy as _np
    f6 = _np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if 83 not in (c.geom1, c.geom2):
            continue
        o = c.geom2 if c.geom1 == 83 else c.geom1
        mujoco.mj_contactForce(m, d, i, f6)
        e = CD.setdefault((ph, CDLAB.get(o, 'geom%d' % o)), [0, 0.0])
        e[0] += 1
        e[1] = max(e[1], abs(float(f6[0])))
    if ph == 'retreat_lift' and 'off' not in CD:
        cb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'cup')
        pos, _r = env.eef_pose()
        CD['off'] = (d.xpos[cb] - pos).round(3).tolist()


def cdump_print():
    items = ['%s/%s n=%d F=%.1f' % (k[0][8:], k[1], v[0], v[1])
             for k, v in sorted(CD.items(), key=str) if k != 'off']
    print('CDUMP cup-eef off=%s | %s' % (CD.get('off'), '; '.join(items) or 'no cup contact'))


def safe_retreat(env, ik, q_cmd, tr=None):
    """Uses only gripper width and end-effector pose. Stop the policy, open fully while holding the pose,
    wait until the finger width stops changing, rise straight up 10 cm (smooth), then hold."""
    def hold(q):
        env.step(q, 0.08, render=False)
        if tr is not None:
            tr.update(0.08)
        rl_log(env, 0.08)
        cdump_step(env)
    CD.clear()
    RL['phase'] = 'retreat_open'
    import os as _os
    _u = _os.environ.get('RT_UNLOAD')
    if _u:
        _p, _rv = env.eef_pose()
        _p = _p.copy()
        _p[2] += float(_u)
        for _ in range(10):
            q_cmd = ik.solve(_p, _rv, q_cmd)
    last_w, stable = -1.0, 0
    for _ in range(75):
        hold(q_cmd)
        w = env.gripper_width
        stable = stable + 1 if abs(w - last_w) < 3e-4 else 0
        last_w = w
        if w > args.rt_full and stable >= 5:
            break
    for _ in range(args.rt_hold):
        hold(q_cmd)
    RL['phase'] = 'retreat_lift'
    pos0, rv0 = env.eef_pose()
    n_up = args.rt_nup
    for i in range(1, n_up + 1):
        s = i / n_up
        s = s * s * (3 - 2 * s)
        p = pos0.copy()
        p[2] = pos0[2] + args.rt_rise * s
        q_cmd = ik.solve(p, rv0, q_cmd)
        hold(q_cmd)
    if args.rt_back != 0.0:
        RL['phase'] = 'retreat_back'
        pb, rvb = env.eef_pose()
        for i in range(1, 61):
            s = i / 60
            s = s * s * (3 - 2 * s)
            p = pb.copy()
            p[0] = pb[0] + args.rt_back * s
            q_cmd = ik.solve(p, rvb, q_cmd)
            hold(q_cmd)
    for _ in range(50):
        hold(q_cmd)
    cdump_print()
    return q_cmd


def rollout(env, policy, cfg, ik, seed, device, record=False):
    torch.manual_seed(seed)
    obs = env.reset(seed=seed)
    hist = collections.deque([obs] * 4, maxlen=4)       # sim frames t-3 .. t
    start_pose = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_rot_axis_angle']]).astype(np.float64)
    q_cmd, steps, max_cup_z, frames = env.arm_q, 0, 0.0, []
    tr = Tracer(env) if args.trace else None
    st = dict(holding_n=0, grasped=False)
    RL.update(on=bool(args.reltrace), rows=[], prev_cup=None, prev_sau=None, prev_tcp=None, phase='policy', n=0)
    M = 3 * args.n_exec
    while steps < args.max_steps:
        env_obs = build_env_obs([hist[0], hist[-1]])     # frames t-3 and t (3 sim steps apart, as in training)
        act = predict(policy, cfg, env_obs, start_pose, device)
        cur = hist[-1]
        pos, rot, grip = interp_plan(act, cur['robot0_eef_pos'], cur['robot0_eef_rot_axis_angle'],
                                     cur['robot0_gripper_width'][0], args.shift, M)
        xb = args.xbias if cur['robot0_gripper_width'][0] > 0.06 else 0.0
        if args.xbias_mode == 'pregrasp' and st['grasped']:
            xb = 0.0
        for m in range(M):
            p = pos[m].copy(); p[2] = max(p[2], args.zhold if st['grasped'] else 0.027); p[0] += xb
            q_cmd = ik.solve(p, rot[m], q_cmd)
            obs = env.step(q_cmd, float(np.clip(grip[m], 0.0, 0.08)), render=True)
            hist.append(obs); steps += 1
            max_cup_z = max(max_cup_z, float(env.get_object_state()['cup_pos'][2]))
            w_m = float(obs['robot0_gripper_width'][0])
            st['holding_n'] = st['holding_n'] + 1 if 0.03 < w_m < 0.062 else 0
            if st['holding_n'] >= 15:
                st['grasped'] = True
            if tr is not None:
                tr.update(float(grip[m]))
            rl_log(env, float(grip[m]))
            if args.retreat and st['grasped'] and w_m > args.rt_trig:
                st['release'] = True
                break
            if record and steps % 2 == 0:
                frames.append(env.render('front', 360, 480))
            if steps >= args.max_steps:
                break
        if st.get('release'):
            break
    if st.get('release'):
        q_cmd = safe_retreat(env, ik, q_cmd, tr)
    ok, dxy, cz = ScriptedExpert(env).success()
    up = float(env.data.xmat[env.cup_id].reshape(3, 3)[2, 2])
    cup_p, tcp_p = env.data.xpos[env.cup_id], env.data.site_xpos[env.tcp_id]
    r = dict(seed=seed, ok=bool(ok), dxy=float(dxy), cup_z=float(cz), up=up, max_cup_z=max_cup_z, steps=steps)
    r['held_end'] = bool(cup_p[2] > 0.06 and np.linalg.norm(cup_p - tcp_p) < 0.06)
    r['retreated'] = bool(st.get('release'))
    rl_finish(seed, ok, args.out)
    if tr is not None:
        r['diag'] = tr.finish(r)
    return r, frames


def check(policy, cfg, device):
    """Compare the policy's predicted world-frame poses with what the recorded demo really did next."""
    root = zarr.open_group(args.data, mode='r')
    D = root['data']
    ends = root['meta/episode_ends'][:]
    rng = np.random.default_rng(0)
    shifts = [0, 1, 2]
    pe = {s: [] for s in shifts}; re_ = {s: [] for s in shifts}; ge = {s: [] for s in shifts}; stay = {s: [] for s in shifts}
    first_step = []
    ks = np.arange(8)
    for _ in range(args.check):
        e = int(rng.integers(len(ends)))
        s0 = 0 if e == 0 else int(ends[e - 1]); e1 = int(ends[e])
        if e1 - s0 < 70:
            continue
        t = int(rng.integers(s0 + 3, e1 - 54))
        sl = slice(t - 3, t + 1, 3)
        env_obs = {
            'camera0_rgb': D['camera0_rgb'][sl],
            'robot0_eef_pos': D['robot0_eef_pos'][sl].astype(np.float64),
            'robot0_eef_rot_axis_angle': D['robot0_eef_rot_axis_angle'][sl].astype(np.float64),
            'robot0_gripper_width': D['robot0_gripper_width'][sl].astype(np.float64),
        }
        start_pose = D['robot0_demo_start_pose'][t].astype(np.float64)
        act = predict(policy, cfg, env_obs, start_pose, device)
        slab_p = D['robot0_eef_pos'][t:t + 55].astype(np.float64)
        slab_r = D['robot0_eef_rot_axis_angle'][t:t + 55].astype(np.float64)
        slab_g = D['robot0_gripper_width'][t:t + 55, 0].astype(np.float64)
        cur_p = env_obs['robot0_eef_pos'][-1]
        first_step.append(np.linalg.norm(act[0, :3] - cur_p))
        for s in shifts:
            idx = 3 * (ks + s)
            pe[s].append(np.linalg.norm(act[ks, :3] - slab_p[idx], axis=1).mean())
            re_[s].append(np.degrees((R.from_rotvec(act[ks, 3:6]).inv() * R.from_rotvec(slab_r[idx])).magnitude()).mean())
            ge[s].append(np.abs(act[ks, 6] - slab_g[idx]).mean())
            stay[s].append(np.linalg.norm(cur_p - slab_p[idx], axis=1).mean())
    n = len(first_step)
    print('\nCHECK on %d recorded samples (mostly training episodes: this tests the plumbing, not generalisation)' % n)
    print('mean |action[0] - current pose| = %.2f cm' % (100 * np.mean(first_step)))
    print('shift | policy pos err (cm) | rot err (deg) | grip err (mm) | "stay still" pos err (cm)')
    for s in shifts:
        print('  %d   |      %6.2f        |    %6.2f     |    %6.2f     |        %6.2f' % (
            s, 100 * np.mean(pe[s]), np.mean(re_[s]), 1000 * np.mean(ge[s]), 100 * np.mean(stay[s])))
    best = min(shifts, key=lambda s: np.mean(pe[s]))
    print('\nbest-matching shift: %d  (use --shift %d for rollouts)' % (best, best))


if args.deterministic:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
device = torch.device('cuda:0')
policy, cfg = load_policy(os.path.expanduser(args.ckpt), device)
print('policy loaded | obs pose repr: %s | action pose repr: %s' % (cfg.task.pose_repr.obs_pose_repr, cfg.task.pose_repr.action_pose_repr))
if args.check:
    check(policy, cfg, device)
    sys.exit(0)

os.makedirs(args.out, exist_ok=True)
env = UmiSimEnv(build_scene(args.menagerie))
ik = PoseIK(env)
results, t0 = [], time.time()
for i in range(args.n):
    seed = args.start_seed + i
    r, frames = rollout(env, policy, cfg, ik, seed, device, record=(i < args.videos))
    results.append(r)
    if frames:
        imageio.mimsave(os.path.join(args.out, 'seed%03d.mp4' % seed), frames, fps=25, macro_block_size=1)
    print('seed %3d | %s | cup-saucer xy %.3f m | cup z now %.3f, max %.3f, upright %.2f | %.0fs elapsed' % (
        seed, 'OK  ' if r['ok'] else 'FAIL', r['dxy'], r['cup_z'], r['max_cup_z'], r['up'], time.time() - t0))
    if 'diag' in r:
        s = r['diag']['sep']
        print('      -> %s%s | finger/hand touches after open: %d, cup drift after release: %.3f m | saucer moved %.3f m, finger-saucer touch steps before/after release: %d/%d' % (r['diag']['cat'], '' if s is None else ' | left gripper at step %d, %.3f m from saucer, z %.3f, cmd %.3f' % (
            s['step'], s['dxy'], s['cup_z'], s['gcmd']), r['diag']['touch_after_open'], r['diag']['drift_after_release'], r['diag']['saucer_moved'], r['diag']['sau_touch_before'], r['diag']['sau_touch_after']))

n_ok = sum(r['ok'] for r in results)
p = n_ok / len(results)
print('\nPOLICY success: %d/%d = %.0f%% (+/- %.0f%%)' % (n_ok, len(results), 100 * p, 196 * np.sqrt(p * (1 - p) / len(results))))
print('cup lifted above 6 cm in %d/%d episodes' % (sum(r['max_cup_z'] > 0.06 for r in results), len(results)))
print('strict success (cup not still held at the end): %d/%d' % (sum(r['ok'] and not r['held_end'] for r in results), len(results)))
if args.trace:
    print('outcome breakdown:')
    for k, v in collections.Counter(r['diag']['cat'] for r in results).most_common():
        print('  %-45s %d' % (k, v))
summary = dict(policy=results, n_exec=args.n_exec, shift=args.shift)

if args.expert:
    ex_ok = 0
    for i in range(args.n):
        env.reset(seed=args.start_seed + i)
        ex = ScriptedExpert(env, render=False)
        ex.run()
        ex_ok += int(ex.success()[0])
    print('EXPERT success on the same seeds: %d/%d = %.0f%%' % (ex_ok, args.n, 100 * ex_ok / args.n))
    summary['expert_ok'] = ex_ok

json.dump(summary, open(os.path.join(args.out, 'results.json'), 'w'), indent=1)
print('saved', os.path.join(args.out, 'results.json'))
env.close()
