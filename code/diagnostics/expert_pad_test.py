"""Diagnosis only. Makes the 8 small fingertip contact boxes taller (in memory only, no file is changed)
and measures scripted-expert success at several forced grasp heights.
Run from ~/umi_mujoco/sim:  python umi_sim/expert_pad_test.py 30 160"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import mujoco
import umi_sim.expert as e1
import umi_sim.expert2 as e2
from umi_sim.scene import build_scene
from umi_sim.env import UmiSimEnv
from umi_sim.expert2 import ScriptedExpertV2

n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
s0 = int(sys.argv[2]) if len(sys.argv) > 2 else 160
env = UmiSimEnv(build_scene(os.path.expanduser('~/umi_mujoco/sim/mujoco_menagerie')))
m = env.model
pads = []
for g in range(m.ngeom):
    b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
    if b in ('left_finger', 'right_finger') and int(m.geom_type[g]) == 6 and m.geom_size[g][1] < 0.0025:
        pads.append(g)
print('tip pad geoms:', pads)
orig = {g: m.geom_size[g].copy() for g in pads}


def set_pad_half_height(h):
    for g in pads:
        s = orig[g].copy()
        if h is not None:
            s[2] = h
        m.geom_size[g] = s
        m.geom_rbound[g] = float(np.linalg.norm(s))
        m.geom_aabb[g] = np.concatenate([[0.0, 0.0, 0.0], s])


for h in (None, 0.012, 0.020, 0.030):
    set_pad_half_height(h)
    for z in (0.035, 0.045, 0.050, 0.055):
        e1.GRASP_Z = z
        e2.GRASP_Z = z
        ok_n, t0 = 0, time.time()
        for i in range(n):
            env.reset(seed=s0 + i)
            ex = ScriptedExpertV2(env, render=False)
            ex.run()
            ok_n += int(bool(ex.success()[0]))
        print('EXPERT pad_half_h %s grasp_z %.3f: %d/%d = %.0f%% | %.0fs' % ('orig' if h is None else '%.3f' % h, z, ok_n, n, 100 * ok_n / n, time.time() - t0))
