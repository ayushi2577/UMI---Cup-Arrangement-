"""Scripted expert: pick up the cup, place it on the saucer. Uses privileged object poses."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from umi_sim.scene import CUP_HALF_H, SAUCER_HALF_H

HOVER_Z = 0.18
GRASP_Z = 0.035      # TCP height when grasping (cup centre is at 0.03)
OPEN, CLOSED = 0.08, 0.0


class ScriptedExpert:
    def __init__(self, env, speed=0.3, render=False):
        self.env = env
        self.m = env.model
        self.speed = speed                    # m/s
        self.render = render
        self.dt = env.substeps * self.m.opt.timestep
        self.ik = mujoco.MjData(self.m)
        hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
        hj = [j for j in range(self.m.njnt) if int(self.m.jnt_type[j]) == hinge]
        self.qadr = self.m.jnt_qposadr[hj]
        self.dofadr = self.m.jnt_dofadr[hj]
        self.qlo = self.m.jnt_range[hj, 0]
        self.qhi = self.m.jnt_range[hj, 1]
        self.on_step = None                   # hook(obs, tcp_target_pos, gripper_cmd) for recording later
        self.max_ik_err = 0.0

    def solve_ik(self, target, q_init, iters=30):
        m, d, tcp = self.m, self.ik, self.env.tcp_id
        d.qpos[:] = self.env.data.qpos
        d.qpos[self.qadr] = q_init
        for _ in range(iters):
            mujoco.mj_forward(m, d)
            pos = d.site_xpos[tcp].copy()
            mat = d.site_xmat[tcp].reshape(3, 3).copy()
            e_p = target - pos
            e_r = R.from_matrix(self.R_t @ mat.T).as_rotvec()
            if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_r) < 1e-3:
                break
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, tcp)
            J = np.vstack([jp[:, self.dofadr], jr[:, self.dofadr]])
            err = np.concatenate([e_p, e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + np.clip(dq, -0.2, 0.2), self.qlo, self.qhi)
        mujoco.mj_forward(m, d)
        self.max_ik_err = max(self.max_ik_err, float(np.linalg.norm(target - d.site_xpos[tcp])))
        return d.qpos[self.qadr].copy()

    def _step(self, grip):
        obs = self.env.step(self.q_cmd, grip, render=self.render)
        if self.on_step:
            self.on_step(obs, self.x_cmd.copy(), grip)

    def move_to(self, target, grip):
        target = np.asarray(target, dtype=float)
        x0 = self.x_cmd.copy()
        n = max(1, int(np.ceil(np.linalg.norm(target - x0) / (self.speed * self.dt))))
        for i in range(1, n + 1):
            self.x_cmd = x0 + (target - x0) * i / n
            self.q_cmd = self.solve_ik(self.x_cmd, self.q_cmd)
            self._step(grip)

    def hold(self, grip, n):
        for _ in range(n):
            self._step(grip)

    def run(self):
        env = self.env
        self.max_ik_err = 0.0
        self.x_cmd = env.eef_pose()[0]
        self.q_cmd = env.arm_q
        self.R_t = env.data.site_xmat[env.tcp_id].reshape(3, 3).copy()   # keep the ready orientation (pointing down)

        s = env.get_object_state()
        cup, sau = s["cup_pos"], s["saucer_pos"]
        place_z = sau[2] + SAUCER_HALF_H + CUP_HALF_H + 0.01

        self.move_to([cup[0], cup[1], HOVER_Z], OPEN)
        self.move_to([cup[0], cup[1], GRASP_Z], OPEN)
        self.hold(CLOSED, 40)
        self.move_to([cup[0], cup[1], HOVER_Z], CLOSED)
        self.move_to([sau[0], sau[1], HOVER_Z], CLOSED)
        self.move_to([sau[0], sau[1], place_z], CLOSED)
        for w in np.linspace(0.05, OPEN, 20):          # open the gripper gradually
            self._step(w)
        self.hold(OPEN, 40)                            # let the cup settle on the saucer
        old_speed = self.speed
        self.speed = 0.08                              # lift away slowly so the fingers don't drag the cup
        self.move_to([sau[0], sau[1], place_z + 0.05], OPEN)
        self.speed = old_speed
        self.move_to([sau[0], sau[1], HOVER_Z], OPEN)
        self.hold(OPEN, 10)

    def success(self):
        s = self.env.get_object_state()
        cup, sau = s["cup_pos"], s["saucer_pos"]
        up = self.env.data.xmat[self.env.cup_id].reshape(3, 3)[2, 2]
        dxy = np.linalg.norm(cup[:2] - sau[:2])
        on_top = cup[2] > sau[2] + SAUCER_HALF_H + CUP_HALF_H - 0.01
        return bool(dxy < 0.03 and on_top and up > 0.95), float(dxy), float(cup[2])
