"""Diagnosis only, prints the robot hand/finger collision geoms. Run from ~/umi_mujoco/sim:
  python umi_sim/inspect_fingers.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import mujoco
from umi_sim.scene import build_scene
from umi_sim.env import UmiSimEnv

env = UmiSimEnv(build_scene(os.path.expanduser('~/umi_mujoco/sim/mujoco_menagerie')))
env.reset(seed=160)
m, d = env.model, env.data
root = int(m.body_rootid[m.site_bodyid[env.tcp_id]])
tcp = d.site_xpos[env.tcp_id].copy()
R = d.site_xmat[env.tcp_id].reshape(3, 3)
print('TCP site body:', mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.site_bodyid[env.tcp_id])), 'gripper width now %.4f' % env.gripper_width)
print('geom_id name | body | type | size (m) | friction | contype/conaff | pos in TCP frame (mm: x y z)')
for g in range(m.ngeom):
    if int(m.body_rootid[m.geom_bodyid[g]]) != root:
        continue
    if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
        continue
    loc = R.T @ (d.geom_xpos[g] - tcp) * 1000.0
    print('%3d %-28s | %-14s | %d | %s | %s | %d/%d | %+6.1f %+6.1f %+6.1f' % (
        g, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])),
        int(m.geom_type[g]), np.round(m.geom_size[g], 4), np.round(m.geom_friction[g], 3), int(m.geom_contype[g]), int(m.geom_conaffinity[g]),
        loc[0], loc[1], loc[2]))
