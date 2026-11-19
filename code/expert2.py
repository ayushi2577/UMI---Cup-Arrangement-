"""Scripted expert with randomised approach offsets and a random start pose (for more varied demos)."""
import numpy as np
from umi_sim.expert import ScriptedExpert, HOVER_Z, GRASP_Z, OPEN, CLOSED
from umi_sim.scene import CUP_HALF_H, SAUCER_HALF_H


class ScriptedExpertV2(ScriptedExpert):
    jitter = 0.0
    rng = None

    def prepare_start(self, target):
        env = self.env
        self.x_cmd = env.eef_pose()[0]
        self.q_cmd = env.arm_q
        self.R_t = env.data.site_xmat[env.tcp_id].reshape(3, 3).copy()
        self.move_to(target, OPEN)
        self.hold(OPEN, 15)

    def run(self):
        env = self.env
        self.max_ik_err = 0.0
        self.x_cmd = env.eef_pose()[0]
        self.q_cmd = env.arm_q
        self.R_t = env.data.site_xmat[env.tcp_id].reshape(3, 3).copy()
        rng = self.rng if self.rng is not None else np.random.default_rng(0)
        j1 = rng.uniform(-self.jitter, self.jitter, 2)
        j2 = rng.uniform(-self.jitter, self.jitter, 2)
        s = env.get_object_state()
        cup, sau = s["cup_pos"], s["saucer_pos"]
        place_z = sau[2] + SAUCER_HALF_H + CUP_HALF_H + 0.01
        self.move_to([cup[0] + j1[0], cup[1] + j1[1], HOVER_Z], OPEN)
        self.move_to([cup[0], cup[1], 0.10], OPEN)
        self.move_to([cup[0], cup[1], GRASP_Z], OPEN)
        self.hold(CLOSED, 40)
        self.move_to([cup[0], cup[1], HOVER_Z], CLOSED)
        self.move_to([sau[0] + j2[0], sau[1] + j2[1], HOVER_Z], CLOSED)
        self.move_to([sau[0], sau[1], 0.12], CLOSED)
        self.move_to([sau[0], sau[1], place_z], CLOSED)
        for w in np.linspace(0.05, OPEN, 20):
            self._step(w)
        self.hold(OPEN, 40)
        old_speed = self.speed
        self.speed = 0.08
        self.move_to([sau[0], sau[1], place_z + 0.05], OPEN)
        self.speed = old_speed
        self.move_to([sau[0], sau[1], HOVER_Z], OPEN)
        self.hold(OPEN, 10)
