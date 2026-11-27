"""Diagnosis only (ground truth). Scripted expert with forced grasp height; classifies each failure.
Run from ~/umi_mujoco/sim:  python umi_sim/expert_fail_mode.py 30 160"""
import os, sys, collections
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
env = UmiSimEnv(build_scene(os.path.expanduser('~/umi_mujoco/sim/mujoco_menagerie')))
S = {}
_orig = env.step


def step(*a, **k):
    out = _orig(*a, **k)
    d = env.data
    cup, tcp = d.xpos[env.cup_id], d.site_xpos[env.tcp_id]
    S['maxz'] = max(S['maxz'], float(cup[2]))
    sep = float(np.linalg.norm(cup - tcp))
    if cup[2] > 0.06:
        if 'lift' not in S:
            S['lift'] = float((d.site_xmat[env.tcp_id].reshape(3, 3).T @ (cup - tcp))[2] * 1000.0)
        if sep > 0.06 and env.gripper_width < 0.06:
            S['slip'] = True
        S['tilt'] = min(S['tilt'], float(d.xmat[env.cup_id].reshape(3, 3)[2, 2]))
    return out


env.step = step
for z in (0.035, 0.045, 0.050, 0.055):
    e1.GRASP_Z = z
    e2.GRASP_Z = z
    cls, lift = collections.Counter(), collections.defaultdict(list)
    for i in range(n):
        env.reset(seed=s0 + i)
        S.clear()
        S.update(maxz=0.0, tilt=1.0)
        ex = ScriptedExpertV2(env, render=False)
        ex.run()
        ok, dxy, cz = ex.success()
        if ok:
            c = 'success'
        elif S['maxz'] < 0.06:
            c = 'never_lifted'
        elif S.get('slip'):
            c = 'slipped_while_closed'
        else:
            c = 'failed_at_place_or_release'
        cls[c] += 1
        if 'lift' in S:
            lift[c].append(S['lift'])
    print('EXPERT grasp_z %.3f | %s | cup-in-hand z at lift (mm): %s' % (
        z, dict(cls), {k: round(float(np.mean(v)), 1) for k, v in lift.items()}))
