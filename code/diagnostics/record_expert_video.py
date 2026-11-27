"""Usage: python record_expert_video.py [seed ...]
Records the scripted expert on each seed to ~/umi_mujoco/exp/vid_expert/expert_seedNNN.mp4"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imageio
from umi_sim.scene import build_scene
from umi_sim.env import UmiSimEnv
from umi_sim.expert import ScriptedExpert

men = os.path.expanduser('~/umi_mujoco/sim/mujoco_menagerie')
out = os.path.expanduser('~/umi_mujoco/exp/vid_expert')
os.makedirs(out, exist_ok=True)
seeds = [int(s) for s in sys.argv[1:]] or [160, 161, 162]
env = UmiSimEnv(build_scene(men))
orig_step = env.step
for seed in seeds:
    frames, n = [], [0]
    def step(*a, **k):
        r = orig_step(*a, **k)
        n[0] += 1
        if n[0] % 2 == 0:
            frames.append(env.render('front', 360, 480))
        return r
    env.step = step
    env.reset(seed=seed)
    ex = ScriptedExpert(env)
    ex.run()
    good, dxy, cz = ex.success()
    path = os.path.join(out, 'expert_seed%03d.mp4' % seed)
    imageio.mimsave(path, frames, fps=25, macro_block_size=1)
    print('seed %d | %s | xy dist %.3f | %d frames | %s' % (seed, 'OK' if good else 'FAIL', dxy, len(frames), path))
env.step = orig_step
env.close()
