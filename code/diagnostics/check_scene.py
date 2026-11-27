"""Usage: python check_scene.py <menagerie_dir> <output_prefix>"""
import os, sys, time, traceback
import numpy as np
import imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from umi_sim.scene import build_scene
from umi_sim.env import UmiSimEnv, READY_QPOS, CUP_Z, SAUCER_Z

men_dir, prefix = sys.argv[1], sys.argv[2]

try:
    xml_path = build_scene(men_dir)
    env = UmiSimEnv(xml_path)
    env.reset(seed=0)
except Exception:
    traceback.print_exc()
    print("\nSCENE LOAD FAILED - paste the error above to Claude.")
    sys.exit(1)

m = env.model
print("model loaded: nq=%d nv=%d nu=%d nbody=%d ngeom=%d ncam=%d" % (m.nq, m.nv, m.nu, m.nbody, m.ngeom, m.ncam))
print("cameras:", [m.camera(i).name for i in range(m.ncam)])
print()

results = []


def run(name, fn):
    try:
        results.append((name, True, fn() or ""))
    except Exception as e:
        results.append((name, False, "%s: %s" % (type(e).__name__, e)))


def t_objects_settle():
    env.reset(seed=1)
    s = env.get_object_state()
    cup_up = env.data.xmat[env.cup_id].reshape(3, 3)[2, 2]
    assert abs(s["cup_pos"][2] - CUP_Z) < 0.005, "cup z=%.3f (expected %.3f)" % (s["cup_pos"][2], CUP_Z)
    assert abs(s["saucer_pos"][2] - SAUCER_Z) < 0.005, "saucer z=%.3f" % s["saucer_pos"][2]
    assert cup_up > 0.98, "cup tipped over (up-axis %.2f)" % cup_up
    return "cup z=%.3f saucer z=%.3f cup upright=%.3f" % (s["cup_pos"][2], s["saucer_pos"][2], cup_up)


def t_tcp_pose():
    env.reset(seed=1)
    pos, _ = env.eef_pose()
    zaxis = env.data.site_xmat[env.tcp_id].reshape(3, 3)[:, 2]
    assert zaxis[2] < -0.95, "gripper not pointing down (tcp z-axis = %s)" % np.round(zaxis, 2)
    assert 0.2 < pos[0] < 0.5 and abs(pos[1]) < 0.1, "unexpected TCP position %s" % np.round(pos, 3)
    return "TCP at %s, pointing down" % np.round(pos, 3)


def t_gripper():
    env.reset(seed=1)
    out = []
    for target in (0.0, 0.08, 0.03):
        for _ in range(60):
            env.step(READY_QPOS, target, render=False)
        out.append((target, env.gripper_width))
    assert out[0][1] < 0.01, "closing failed: width %.3f" % out[0][1]
    assert out[1][1] > 0.07, "opening failed: width %.3f" % out[1][1]
    assert abs(out[2][1] - 0.03) < 0.01, "width control inaccurate: asked 0.03, got %.3f" % out[2][1]
    return "commanded->measured " + ", ".join("%.2f->%.3f" % o for o in out)


def t_arm_tracking():
    env.reset(seed=1)
    target = READY_QPOS + np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(150):
        env.step(target, 0.08, render=False)
    err = np.abs(env.arm_q - target).max()
    assert err < 0.05, "joint tracking error %.3f rad" % err
    return "max joint error %.4f rad after 3 s" % err


def t_speed():
    env.reset(seed=1)
    t0 = time.time()
    for _ in range(20):
        env.step(READY_QPOS, 0.08, render=True)
    dt = (time.time() - t0) / 20
    return "%.1f ms per step incl. wrist render (%.0f steps/s)" % (dt * 1000, 1 / dt)


run("objects rest on table", t_objects_settle)
run("TCP pose at ready position", t_tcp_pose)
run("gripper width control", t_gripper)
run("arm joint tracking", t_arm_tracking)
run("step speed", t_speed)

env.reset(seed=0)
front = env.render("front", 360, 480)
over = env.render("overhead", 360, 480)
imageio.imwrite(prefix + "_overview.png", np.concatenate([front, over], axis=1))
top, bottom = [], []
for s in range(4):
    env.reset(seed=s)
    top.append(env.render("overhead", 224, 224))
    bottom.append(env.render("wrist", 224, 224))
imageio.imwrite(prefix + "_resets.png", np.concatenate([np.concatenate(top, 1), np.concatenate(bottom, 1)], 0))

print("RESULTS")
for name, ok, msg in results:
    print("  [%s] %-28s %s" % ("PASS" if ok else "FAIL", name, msg))
n_fail = sum(1 for r in results if not r[1])
print("\n%d/%d checks passed. images: %s_overview.png, %s_resets.png" % (len(results) - n_fail, len(results), prefix, prefix))
env.close()
