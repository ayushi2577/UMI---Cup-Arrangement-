import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from umi_sim.scene import build_scene
from umi_sim.env import UmiSimEnv
from umi_sim.expert import ScriptedExpert

env = UmiSimEnv(build_scene(sys.argv[1]))
for seed in (0, 5):
    env.reset(seed=seed)
    s = env.get_object_state()
    print("\n=== seed %d | cup xy %s | saucer xy %s" % (seed, np.round(s["cup_pos"][:2], 3), np.round(s["saucer_pos"][:2], 3)))
    ex = ScriptedExpert(env)
    rows = []
    def hook(obs, x_cmd, grip):
        o = env.get_object_state()
        up = env.data.xmat[env.cup_id].reshape(3, 3)[2, 2]
        tcp = env.eef_pose()[0]
        rows.append((grip, env.gripper_width, tcp[2], o["cup_pos"][2], up, np.linalg.norm(o["cup_pos"][:2] - tcp[:2])))
    ex.on_step = hook
    ex.run()
    print("step | grip cmd | grip width | tcp z | cup z | cup up | cup-tcp xy")
    for i in range(0, len(rows), 25):
        g, w, tz, cz, up, d = rows[i]
        print("%4d | %.2f | %.3f | %.3f | %.3f | %.2f | %.3f" % (i, g, w, tz, cz, up, d))
env.close()
