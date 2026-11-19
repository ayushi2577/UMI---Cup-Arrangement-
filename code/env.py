"""MuJoCo environment that exposes the same kinds of observations UMI's real-robot setup records."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

READY_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
GRIPPER_MAX_WIDTH = 0.08
SQUEEZE_BELOW = 0.06

CUP_Z = 0.03
SAUCER_Z = 0.01


class UmiSimEnv:
    def __init__(self, xml_path, img_size=224, substeps=10):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.img_size = img_size
        self.substeps = substeps
        self.rng = np.random.default_rng(0)
        self._renderers = {}

        m = self.model
        hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
        slide = int(mujoco.mjtJoint.mjJNT_SLIDE)
        hinges = [j for j in range(m.njnt) if int(m.jnt_type[j]) == hinge]
        slides = [j for j in range(m.njnt) if int(m.jnt_type[j]) == slide]
        if len(hinges) != 7 or len(slides) != 2 or m.nu != 8:
            raise RuntimeError(
                "unexpected robot layout: %d hinge joints, %d slide joints, %d actuators (expected 7, 2, 8)"
                % (len(hinges), len(slides), m.nu))
        self.arm_qadr = np.array([m.jnt_qposadr[j] for j in hinges])
        self.finger_qadr = np.array([m.jnt_qposadr[j] for j in slides])

        def _id(kind, name):
            i = mujoco.mj_name2id(m, kind, name)
            if i < 0:
                raise RuntimeError("scene has no %s named '%s'" % (kind, name))
            return i

        self.tcp_id = _id(mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.cup_id = _id(mujoco.mjtObj.mjOBJ_BODY, "cup")
        self.saucer_id = _id(mujoco.mjtObj.mjOBJ_BODY, "saucer")
        self.cup_qadr = m.jnt_qposadr[_id(mujoco.mjtObj.mjOBJ_JOINT, "cup_joint")]
        self.saucer_qadr = m.jnt_qposadr[_id(mujoco.mjtObj.mjOBJ_JOINT, "saucer_joint")]

    def _sample_layout(self):
        cup_xy = np.array([self.rng.uniform(0.38, 0.62), self.rng.uniform(-0.27, -0.07)])
        saucer_xy = np.array([self.rng.uniform(0.38, 0.62), self.rng.uniform(0.07, 0.27)])
        return cup_xy, saucer_xy

    def _set_free_body(self, qadr, xyz, yaw):
        q = self.data.qpos
        q[qadr:qadr + 3] = xyz
        q[qadr + 3:qadr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.arm_qadr] = READY_QPOS
        d.qpos[self.finger_qadr] = 0.04
        d.ctrl[:7] = READY_QPOS
        d.ctrl[7] = 255.0
        cup_xy, saucer_xy = self._sample_layout()
        self._set_free_body(self.cup_qadr, (cup_xy[0], cup_xy[1], CUP_Z), self.rng.uniform(-np.pi, np.pi))
        self._set_free_body(self.saucer_qadr, (saucer_xy[0], saucer_xy[1], SAUCER_Z), self.rng.uniform(-np.pi, np.pi))
        mujoco.mj_forward(m, d)
        for _ in range(200):
            mujoco.mj_step(m, d)
        return self.get_obs()

    def render(self, camera, height=None, width=None):
        h = height or self.img_size
        w = width or self.img_size
        if (h, w) not in self._renderers:
            self._renderers[(h, w)] = mujoco.Renderer(self.model, h, w)
        r = self._renderers[(h, w)]
        r.update_scene(self.data, camera=camera)
        return r.render().copy()

    @property
    def arm_q(self):
        return self.data.qpos[self.arm_qadr].copy()

    @property
    def gripper_width(self):
        return float(self.data.qpos[self.finger_qadr].sum())

    def eef_pose(self):
        pos = self.data.site_xpos[self.tcp_id].copy()
        mat = self.data.site_xmat[self.tcp_id].reshape(3, 3).copy()
        return pos, R.from_matrix(mat).as_rotvec()

    def get_obs(self, render=True):
        pos, rotvec = self.eef_pose()
        obs = {
            "robot0_eef_pos": pos,
            "robot0_eef_rot_axis_angle": rotvec,
            "robot0_gripper_width": np.array([self.gripper_width]),
        }
        if render:
            obs["camera0_rgb"] = self.render("wrist")
        return obs

    def get_object_state(self):
        return {
            "cup_pos": self.data.xpos[self.cup_id].copy(),
            "saucer_pos": self.data.xpos[self.saucer_id].copy(),
        }

    def step(self, arm_q_target, gripper_width, render=True):
        d = self.data
        d.ctrl[:7] = arm_q_target
        if gripper_width < SQUEEZE_BELOW:   # commanded width below this means squeeze; the data stores the measured cup width, not the command
            gripper_width = 0.0
        d.ctrl[7] = float(np.clip(gripper_width / GRIPPER_MAX_WIDTH, 0.0, 1.0) * 255.0)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, d)
        return self.get_obs(render=render)

    def close(self):
        for r in self._renderers.values():
            r.close()
        self._renderers = {}
