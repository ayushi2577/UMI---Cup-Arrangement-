import os
import sys
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))

SRC = open(
    os.path.join(HERE, "eval_sim.py")
).read().split("device = torch.device('cuda:0')")[0]

ckpt = sys.argv[1]

sys.argv = [
    "eval_sim.py",
    "--ckpt", ckpt,
    "--lead", "6"
]

exec(compile(SRC, "eval_sim.py", "exec"), globals())

model = env.model

print("\n========== ALL GRIPPER-RELATED GEOMETRIES ==========")

for i in range(model.ngeom):
    name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_GEOM, i
    ) or f"unnamed_geom_{i}"

    body_id = model.geom_bodyid[i]
    body_name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, body_id
    ) or "unnamed_body"

    key = (name + " " + body_name).lower()

    if any(k in key for k in [
        "finger", "hand", "gripper", "panda"
    ]):
        print(
            f"\nID: {i}"
            f"\nGeom: {name}"
            f"\nBody: {body_name}"
            f"\nType: {model.geom_type[i]}"
            f"\nSize: {model.geom_size[i]}"
            f"\nFriction: {model.geom_friction[i]}"
            f"\nContype: {model.geom_contype[i]}"
            f"\nConaffinity: {model.geom_conaffinity[i]}"
        )

print("\n========== END ==========")
