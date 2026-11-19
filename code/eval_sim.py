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
args = ap.parse_args()
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
    return pos, sl(ms).as_rotvec(), np.interp(ms, times, G)


def rollout(env, policy, cfg, ik, seed, device, record=False):
    torch.manual_seed(seed)
    obs = env.reset(seed=seed)
    hist = collections.deque([obs] * 4, maxlen=4)       # sim frames t-3 .. t
    start_pose = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_rot_axis_angle']]).astype(np.float64)
    q_cmd, steps, max_cup_z, frames = env.arm_q, 0, 0.0, []
    M = 3 * args.n_exec
    while steps < args.max_steps:
        env_obs = build_env_obs([hist[0], hist[-1]])     # frames t-3 and t (3 sim steps apart, as in training)
        act = predict(policy, cfg, env_obs, start_pose, device)
        cur = hist[-1]
        pos, rot, grip = interp_plan(act, cur['robot0_eef_pos'], cur['robot0_eef_rot_axis_angle'],
                                     cur['robot0_gripper_width'][0], args.shift, M)
        xb = args.xbias if cur['robot0_gripper_width'][0] > 0.06 else 0.0
        for m in range(M):
            p = pos[m].copy(); p[2] = max(p[2], 0.027); p[0] += xb
            q_cmd = ik.solve(p, rot[m], q_cmd)
            obs = env.step(q_cmd, float(np.clip(grip[m], 0.0, 0.08)), render=True)
            hist.append(obs); steps += 1
            max_cup_z = max(max_cup_z, float(env.get_object_state()['cup_pos'][2]))
            if record and steps % 2 == 0:
                frames.append(env.render('front', 360, 480))
            if steps >= args.max_steps:
                break
    ok, dxy, cz = ScriptedExpert(env).success()
    up = float(env.data.xmat[env.cup_id].reshape(3, 3)[2, 2])
    return dict(seed=seed, ok=bool(ok), dxy=float(dxy), cup_z=float(cz), up=up, max_cup_z=max_cup_z, steps=steps), frames


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

n_ok = sum(r['ok'] for r in results)
p = n_ok / len(results)
print('\nPOLICY success: %d/%d = %.0f%% (+/- %.0f%%)' % (n_ok, len(results), 100 * p, 196 * np.sqrt(p * (1 - p) / len(results))))
print('cup lifted above 6 cm in %d/%d episodes' % (sum(r['max_cup_z'] > 0.06 for r in results), len(results)))
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
