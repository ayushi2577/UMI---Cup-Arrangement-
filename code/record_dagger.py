"""Handoff-DAgger recorder. The final policy (section 6 controller) drives from reset until the gripper is
open and the TCP is below --handoff_z. The scripted expert then takes over from that exact pose, goes to the cup,
grasps, carries, places, releases. Only the expert segment is stored (UMI zarr format, no action field needed).
Ground truth is used only to label the expert and to drop episodes the policy already ruined.
  python umi_sim/record_dagger.py --ckpt CK --out ~/umi_mujoco/sim_demos/dagger_a.zarr --n 5
Resumable: re-run the same command; it continues until --n successful episodes are stored."""
import os, sys, json, time, argparse, collections
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.expanduser('~/umi_mujoco/universal_manipulation_interface')
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, REPO)

import numpy as np, torch, dill, hydra, mujoco, zarr, numcodecs
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R, Slerp

OmegaConf.register_new_resolver('eval', eval, replace=True)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
register_codecs()
from umi.real_world.real_inference_util import get_real_umi_obs_dict, get_real_umi_action
from umi_sim.scene import build_scene, CUP_HALF_H, SAUCER_HALF_H
from umi_sim.env import UmiSimEnv
from umi_sim.expert import ScriptedExpert, GRASP_Z, OPEN, CLOSED

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out', required=True, help='directory ending in .zarr')
ap.add_argument('--n', type=int, default=5, help='successful handoff episodes wanted')
ap.add_argument('--start_seed', type=int, default=10000, help='must not overlap eval seeds 0-279 or demo seeds 5000+')
ap.add_argument('--handoff_z', type=float, default=0.12, help='hand over when TCP z is below this with the gripper open')
ap.add_argument('--max_steps', type=int, default=480)
ap.add_argument('--max_attempts', type=int, default=100000)
ap.add_argument('--menagerie', default=os.path.expanduser('~/umi_mujoco/sim/mujoco_menagerie'))
ap.add_argument('--ref', default=os.path.expanduser('~/umi_mujoco/data/cup_in_the_wild.zarr.zip'))
args = ap.parse_args()
args.out = os.path.expanduser(args.out)

# final controller settings (section 6), applied during the policy phase only
N_EXEC, LEAD, GLEAD, XBIAS, ZBIAS = 8, 6.0, 0.0, 0.012, 0.012


def load_policy(ckpt, device):
    payload = torch.load(open(ckpt, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    ws = hydra.utils.get_class(cfg._target_)(cfg)
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
    policy.reset()
    obs_np = get_real_umi_obs_dict(
        env_obs=env_obs, shape_meta=cfg.task.shape_meta,
        obs_pose_repr=cfg.task.pose_repr.obs_pose_repr,
        tx_robot1_robot0=None, episode_start_pose=[start_pose])
    obs_t = dict_apply(obs_np, lambda x: torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(device))
    a = policy.predict_action(obs_t)['action_pred'][0].detach().cpu().numpy()
    return get_real_umi_action(a, env_obs, cfg.task.pose_repr.action_pose_repr)


def build_env_obs(frames):
    return {
        'camera0_rgb': np.stack([f['camera0_rgb'] for f in frames]),
        'robot0_eef_pos': np.stack([f['robot0_eef_pos'] for f in frames]).astype(np.float64),
        'robot0_eef_rot_axis_angle': np.stack([f['robot0_eef_rot_axis_angle'] for f in frames]).astype(np.float64),
        'robot0_gripper_width': np.stack([f['robot0_gripper_width'] for f in frames]).astype(np.float64),
    }


class PoseIK:
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


def interp_plan(act, M):
    times = np.array([3.0 * k - LEAD for k in range(len(act))])
    sl = Slerp(times, R.from_rotvec(act[:, 3:6]))
    ms = np.arange(1, M + 1).astype(float)
    pos = np.stack([np.interp(ms, times, act[:, i]) for i in range(3)], axis=1)
    grip = np.interp(ms, times + (LEAD - GLEAD), act[:, 6])
    return pos, sl(ms).as_rotvec(), grip


def policy_to_handoff(env, policy, cfg, ik, device, seed):
    """Returns (R_ready, reason). reason == 'ok' means the expert may take over now."""
    torch.manual_seed(seed)
    obs = env.reset(seed=seed)
    R_ready = env.data.site_xmat[env.tcp_id].reshape(3, 3).copy()
    cup0 = env.get_object_state()['cup_pos'].copy()
    hist = collections.deque([obs] * 4, maxlen=4)
    start_pose = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_rot_axis_angle']]).astype(np.float64)
    q_cmd, steps, M = env.arm_q, 0, 3 * N_EXEC
    while steps < args.max_steps:
        act = predict(policy, cfg, build_env_obs([hist[0], hist[-1]]), start_pose, device)
        pos, rot, grip = interp_plan(act, M)
        for m in range(M):
            w = float(obs['robot0_gripper_width'][0])
            p = pos[m].copy()
            p[2] = max(p[2], 0.027) + ZBIAS
            if w > 0.06:
                p[0] += XBIAS
            q_cmd = ik.solve(p, rot[m], q_cmd)
            obs = env.step(q_cmd, float(np.clip(grip[m], 0.0, 0.08)), render=True)
            hist.append(obs)
            steps += 1
            w = float(obs['robot0_gripper_width'][0])
            cup = env.get_object_state()['cup_pos']
            if np.linalg.norm(cup[:2] - cup0[:2]) > 0.01 or cup[2] > 0.04:
                return None, 'cup_disturbed'
            if w < 0.06:
                return None, 'closed_before_handoff'
            if float(env.data.site_xpos[env.tcp_id][2]) < args.handoff_z:
                return R_ready, 'ok'
    return None, 'never_reached_handoff'


class HandoffExpert(ScriptedExpert):
    def run_from_here(self, R_ready):
        env = self.env
        self.max_ik_err = 0.0
        self.x_cmd = env.eef_pose()[0]
        self.q_cmd = env.arm_q
        self.R_t = R_ready
        s = env.get_object_state()
        cup, sau = s['cup_pos'], s['saucer_pos']
        place_z = sau[2] + SAUCER_HALF_H + CUP_HALF_H + 0.01
        if self.x_cmd[2] > 0.10:
            self.move_to([cup[0], cup[1], 0.10], OPEN)
        self.move_to([cup[0], cup[1], GRASP_Z], OPEN)
        self.hold(CLOSED, 40)
        self.move_to([cup[0], cup[1], 0.18], CLOSED)
        self.move_to([sau[0], sau[1], 0.18], CLOSED)
        self.move_to([sau[0], sau[1], 0.12], CLOSED)
        self.move_to([sau[0], sau[1], place_z], CLOSED)
        for w in np.linspace(0.05, OPEN, 20):
            self._step(w)
        self.hold(OPEN, 40)
        old = self.speed
        self.speed = 0.08
        self.move_to([sau[0], sau[1], place_z + 0.05], OPEN)
        self.speed = old
        self.move_to([sau[0], sau[1], 0.18], OPEN)
        self.hold(OPEN, 10)


LOWDIM = {
    'robot0_eef_pos': (np.float32, (3,)),
    'robot0_eef_rot_axis_angle': (np.float32, (3,)),
    'robot0_gripper_width': (np.float32, (1,)),
    'robot0_demo_start_pose': (np.float64, (6,)),
    'robot0_demo_end_pose': (np.float64, (6,)),
}


def img_compressor(ref):
    st = zarr.ZipStore(ref, mode='r')
    cfg = zarr.group(store=st)['data/camera0_rgb'].compressor.get_config()
    st.close()
    return numcodecs.get_codec(cfg)


def open_out(path, comp):
    root = zarr.open_group(zarr.DirectoryStore(path), mode='a')
    data, meta = root.require_group('data'), root.require_group('meta')
    if 'camera0_rgb' not in data:
        data.create_dataset('camera0_rgb', shape=(0, 224, 224, 3), chunks=(1, 224, 224, 3), dtype=np.uint8, compressor=comp)
        for k, (dt, tail) in LOWDIM.items():
            data.create_dataset(k, shape=(0,) + tail, chunks=(400,) + tail, dtype=dt, compressor=None)
        meta.create_dataset('episode_ends', shape=(0,), chunks=(1596,), dtype=np.int64, compressor=None)
    ends = meta['episode_ends']
    total = int(ends[-1]) if ends.shape[0] else 0
    for k in data.array_keys():
        a = data[k]
        if a.shape[0] != total:
            print('repairing %s: %d -> %d frames' % (k, a.shape[0], total))
            a.resize((total,) + a.shape[1:])
    return root


def unwrap_rotvec(rv):
    out = rv.astype(np.float64).copy()
    def alt(r):
        th = np.linalg.norm(r)
        return r if th < 1e-6 else r - 2 * np.pi * r / th
    if out[0][0] > 0:
        out[0] = alt(out[0])
    for i in range(1, len(out)):
        a2 = alt(out[i])
        if np.linalg.norm(a2 - out[i - 1]) < np.linalg.norm(out[i] - out[i - 1]):
            out[i] = a2
    return out


def to_arrays(frames):
    n = len(frames)
    pos = np.stack([f['robot0_eef_pos'] for f in frames]).astype(np.float64)
    rot = unwrap_rotvec(np.stack([f['robot0_eef_rot_axis_angle'] for f in frames]))
    return {
        'camera0_rgb': np.stack([f['camera0_rgb'] for f in frames]).astype(np.uint8),
        'robot0_eef_pos': pos.astype(np.float32),
        'robot0_eef_rot_axis_angle': rot.astype(np.float32),
        'robot0_gripper_width': np.stack([f['robot0_gripper_width'] for f in frames]).astype(np.float32),
        'robot0_demo_start_pose': np.tile(np.concatenate([pos[0], rot[0]]), (n, 1)),
        'robot0_demo_end_pose': np.tile(np.concatenate([pos[-1], rot[-1]]), (n, 1)),
    }


def append_episode(root, arrs):
    n = len(arrs['camera0_rgb'])
    for k, v in arrs.items():
        root['data'][k].append(v)
    ends = root['meta/episode_ends']
    prev = int(ends[-1]) if ends.shape[0] else 0
    ends.append(np.array([prev + n], dtype=np.int64))


device = torch.device('cuda:0')
policy, cfg = load_policy(os.path.expanduser(args.ckpt), device)
os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
root = open_out(args.out, img_compressor(args.ref))
log_path = args.out + '.attempts.jsonl'
attempted = [json.loads(l)['seed'] for l in open(log_path) if l.strip()] if os.path.exists(log_path) else []
seed = max(attempted) + 1 if attempted else args.start_seed
n_ok = int(root['meta/episode_ends'].shape[0])
print('starting with %d episodes stored, want %d, next seed %d' % (n_ok, args.n, seed))

env = UmiSimEnv(build_scene(args.menagerie))
ik = PoseIK(env)
why = collections.Counter()
tries, t_all = 0, time.time()
while n_ok < args.n and tries < args.max_attempts:
    t0 = time.time()
    reason, ok, nfr, dxy = 'crash', False, 0, float('nan')
    try:
        R_ready, reason = policy_to_handoff(env, policy, cfg, ik, device, seed)
        if reason == 'ok':
            frames = [env.get_obs()]
            ex = HandoffExpert(env, render=True)
            ex.on_step = lambda obs, x, g: frames.append(obs)
            ex.run_from_here(R_ready)
            ok, dxy, _ = ex.success()
            nfr = len(frames)
            reason = 'expert_ok' if ok else 'expert_failed'
            if ok:
                append_episode(root, to_arrays(frames))
                n_ok += 1
    except Exception as e:
        reason = 'crash_%s' % type(e).__name__
        print('seed %d crashed: %s: %s' % (seed, type(e).__name__, e))
    why[reason] += 1
    with open(log_path, 'a') as f:
        f.write(json.dumps({'seed': seed, 'reason': reason, 'ok': bool(ok), 'frames': nfr}) + '\n')
    print('seed %5d | %-24s | %3d frames | %.1fs | stored %d/%d' % (seed, reason, nfr, time.time() - t0, n_ok, args.n))
    seed += 1
    tries += 1
env.close()
print('\nthis run: %d attempts, %.1f min' % (tries, (time.time() - t_all) / 60))
for k, v in why.most_common():
    print('  %-26s %d' % (k, v))
ends = root['meta/episode_ends'][:]
if len(ends):
    lens = np.diff(np.concatenate([[0], ends]))
    print('stored episodes %d | frames %d | length min/mean/max %d %.0f %d' % (len(ends), int(ends[-1]), lens.min(), lens.mean(), lens.max()))
