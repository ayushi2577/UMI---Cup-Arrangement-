import os, sys, json
os.environ.setdefault('MUJOCO_GL', 'egl'); os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, mujoco, zarr
from scipy.spatial.transform import Rotation as R
from umi_sim.scene import build_scene
from umi_sim.env import UmiSimEnv
from umi_sim.expert import ScriptedExpert

DATA = os.path.expanduser('~/umi_mujoco/sim_demos/cup_sim_200.zarr')
root = zarr.open_group(DATA, mode='r'); D = root['data']; ends = root['meta/episode_ends'][:]
seeds = [json.loads(l)['seed'] for l in open(DATA + '.attempts.jsonl') if l.strip() and json.loads(l)['ok']]
assert len(seeds) == len(ends), (len(seeds), len(ends))


class PoseIK:
    def __init__(self, env):
        m = env.model
        self.env, self.m, self.d = env, m, mujoco.MjData(m)
        hj = [j for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)]
        self.qadr, self.dofadr = m.jnt_qposadr[hj], m.jnt_dofadr[hj]
        self.lo, self.hi = m.jnt_range[hj, 0], m.jnt_range[hj, 1]

    def solve(self, pos, rotvec, q_init, iters=30):
        m, d, tcp = self.m, self.d, self.env.tcp_id
        Rt = R.from_rotvec(rotvec).as_matrix()
        d.qpos[:] = self.env.data.qpos
        d.qpos[self.qadr] = q_init
        for _ in range(iters):
            mujoco.mj_forward(m, d)
            p = d.site_xpos[tcp].copy(); M = d.site_xmat[tcp].reshape(3, 3).copy()
            e_p = pos - p; e_r = R.from_matrix(Rt @ M.T).as_rotvec()
            if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_r) < 1e-3:
                break
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, tcp)
            J = np.vstack([jp[:, self.dofadr], jr[:, self.dofadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.concatenate([e_p, e_r]))
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        return d.qpos[self.qadr].copy()


def set_gravcomp(env, on):
    m = env.model
    for b in range(1, m.nbody):
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) in ('cup', 'saucer'):
            continue
        m.body_gravcomp[b] = 1.0 if on else 0.0


def replay(env, ik, e):
    s0 = 0 if e == 0 else int(ends[e - 1]); e1 = int(ends[e])
    pos = D['robot0_eef_pos'][s0:e1].astype(np.float64)
    rot = D['robot0_eef_rot_axis_angle'][s0:e1].astype(np.float64)
    w = D['robot0_gripper_width'][s0:e1, 0].astype(np.float64)
    env.reset(seed=seeds[e])
    q, drift = env.arm_q, 0.0
    for t in range(1, len(pos)):
        q = ik.solve(pos[t], rot[t], q)
        env.step(q, float(w[t]), render=False)
        drift = max(drift, abs(env.eef_pose()[0][2] - pos[t][2]))
    ok, dxy, cz = ScriptedExpert(env).success()
    return ok, dxy, drift


env = UmiSimEnv(build_scene(os.path.expanduser('~/umi_mujoco/sim/mujoco_menagerie')))
ik = PoseIK(env)
for label, gc in (('CURRENT physics', False), ('GRAVITY-COMPENSATED arm', True)):
    set_gravcomp(env, gc)
    res = [replay(env, ik, e) for e in range(10)]
    print('\n=== %s ===' % label)
    print('replay of 10 recorded demos through the policy pipeline: %d/10 succeed' % sum(r[0] for r in res))
    print('max height drift from the recorded demo (cm), per episode:', np.round([100 * r[2] for r in res], 1))
    ok = 0
    for s in range(10):
        env.reset(seed=s); ex = ScriptedExpert(env); ex.run(); ok += int(ex.success()[0])
    print('scripted expert under this physics: %d/10' % ok)
env.close()
