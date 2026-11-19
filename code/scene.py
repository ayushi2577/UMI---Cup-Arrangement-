"""Builds the MuJoCo scene: Franka Panda (mujoco_menagerie) + table + cup + saucer + wrist camera."""
import os
import numpy as np
import xml.etree.ElementTree as ET

WRIST_CAM_POS = "0.06 0 0.03"
WRIST_CAM_XYAXES = "0 1 0 1 0 0"
WRIST_CAM_FOVY = "100"
TCP_OFFSET = "0 0 0.1034"

CUP_RADIUS, CUP_HALF_H = 0.025, 0.03
SAUCER_RADIUS, SAUCER_HALF_H = 0.065, 0.01


def lookat_xyaxes(pos, target, up=(0.0, 0.0, 1.0)):
    pos, target, up = (np.asarray(v, dtype=float) for v in (pos, target, up))
    z = pos - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return " ".join("%.5f" % v for v in np.concatenate([x, y]))


def patch_panda(panda_dir):
    src = os.path.join(panda_dir, "panda.xml")
    dst = os.path.join(panda_dir, "panda_umi.xml")
    tree = ET.parse(src)
    root = tree.getroot()
    hand = None
    for body in root.iter("body"):
        if body.get("name") == "hand":
            hand = body
            break
    if hand is None:
        raise RuntimeError("could not find <body name='hand'> in panda.xml - menagerie layout changed")
    ET.SubElement(hand, "camera", name="wrist", pos=WRIST_CAM_POS, xyaxes=WRIST_CAM_XYAXES, fovy=WRIST_CAM_FOVY)
    ET.SubElement(hand, "site", name="tcp", pos=TCP_OFFSET, size="0.004", rgba="1 0 0 1", group="4")
    for kf in root.findall("keyframe"):
        root.remove(kf)
    k = float(os.environ.get('GRIPK', '1'))
    for g in root.iter('general'):
        if g.get('name') == 'actuator8':
            g.set('gainprm', '%g 0 0' % (0.01568627451 * k))
            g.set('biasprm', '0 %g %g' % (-100 * k, -10 * k))
    tree.write(dst)
    return dst


def build_scene(menagerie_dir, out_name="umi_scene.xml"):
    panda_dir = os.path.join(menagerie_dir, "franka_emika_panda")
    patch_panda(panda_dir)
    front = lookat_xyaxes((1.25, -0.85, 0.75), (0.45, 0.0, 0.05))
    overhead = lookat_xyaxes((0.45, 0.0, 1.3), (0.45, 0.0, 0.0), up=(1.0, 0.0, 0.0))
    xml = f"""<mujoco model="umi_sim_scene">
  <include file="panda_umi.xml"/>
  <statistic center="0.45 0 0.2" extent="1.2"/>
  <visual>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.25 0.25 0.25" specular="0 0 0"/>
    <global offwidth="960" offheight="720"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <material name="table_mat" rgba="0.76 0.60 0.42 1"/>
    <material name="cup_mat" rgba="0.95 0.95 0.90 1"/>
    <material name="saucer_mat" rgba="0.20 0.45 0.80 1"/>
  </asset>
  <worldbody>
    <light pos="0.5 0 2" dir="0 0 -1" diffuse="0.5 0.5 0.5" castshadow="false"/>
    <geom name="floor" type="plane" pos="0 0 -0.75" size="3 3 0.1" rgba="0.55 0.55 0.58 1"/>
    <geom name="table" type="box" pos="0.4 0 -0.02" size="0.55 0.7 0.02" material="table_mat" solref="0.005 1" friction="1 0.005 0.0001"/>
    <body name="cup" pos="0.5 -0.15 {CUP_HALF_H}">
      <freejoint name="cup_joint"/>
      <geom name="cup_geom" type="cylinder" size="{CUP_RADIUS} {CUP_HALF_H}" mass="0.1" material="cup_mat" solref="0.005 1" condim="4" friction="1.5 0.05 0.001"/>
    </body>
    <body name="saucer" pos="0.5 0.15 {SAUCER_HALF_H}">
      <freejoint name="saucer_joint"/>
      <geom name="saucer_geom" type="cylinder" size="{SAUCER_RADIUS} {SAUCER_HALF_H}" mass="0.08" material="saucer_mat" solref="0.005 1" friction="1 0.01 0.001"/>
    </body>
    <camera name="front" pos="1.25 -0.85 0.75" xyaxes="{front}"/>
    <camera name="overhead" pos="0.45 0 1.3" xyaxes="{overhead}"/>
  </worldbody>
</mujoco>
"""
    out = os.path.join(panda_dir, out_name)
    with open(out, "w") as f:
        f.write(xml)
    return out
