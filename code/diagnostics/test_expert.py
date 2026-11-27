"""Usage: python test_expert.py <menagerie_dir> [n_seeds]"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from umi_sim.scene import build_scene
from umi_sim.env import UmiSimEnv
from umi_sim.expert import ScriptedExpert

men = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
env = UmiSimEnv(build_scene(men))
ok = 0
for seed in range(n):
    env.reset(seed=seed)
    ex = ScriptedExpert(env)
    t0 = time.time()
    ex.run()
    good, dxy, cz = ex.success()
    ok += good
    print("seed %2d | %s | cup-saucer xy dist %.3f m | cup z %.3f | max IK err %.4f m | %.1fs"
          % (seed, "OK  " if good else "FAIL", dxy, cz, ex.max_ik_err, time.time() - t0))
print("\nsuccess: %d/%d" % (ok, n))
env.close()
