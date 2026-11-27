"""Diagnosis only. Runs the scripted expert (no policy) with the grasp height forced to several values.
If success drops at higher grasp heights, height alone breaks the grasp (physics).
If success stays ~100%, height alone is harmless and the policy's behaviour is the cause.
Put this file in ~/umi_mujoco/sim/umi_sim/ and run from ~/umi_mujoco/sim:
  python umi_sim/expert_height_test.py 30 160"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import umi_sim.expert as e1
import umi_sim.expert2 as e2
from umi_sim.scene import build_scene
from umi_sim.env import UmiSimEnv
from umi_sim.expert2 import ScriptedExpertV2

n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
s0 = int(sys.argv[2]) if len(sys.argv) > 2 else 160
menagerie = os.path.expanduser('~/umi_mujoco/sim/mujoco_menagerie')
env = UmiSimEnv(build_scene(menagerie))
print('original GRASP_Z', e2.GRASP_Z)
for z in (0.028, 0.032, 0.035, 0.038, 0.040, 0.042):
    e1.GRASP_Z = z
    e2.GRASP_Z = z
    ok_n, t0, bad = 0, time.time(), []
    for i in range(n):
        env.reset(seed=s0 + i)
        ex = ScriptedExpertV2(env, render=False)
        ex.run()
        ok = bool(ex.success()[0])
        ok_n += ok
        if not ok:
            bad.append(s0 + i)
    print('EXPERT grasp_z %.3f: %d/%d = %.0f%% | %.0fs | failed seeds %s' % (z, ok_n, n, 100 * ok_n / n, time.time() - t0, bad[:10]))
