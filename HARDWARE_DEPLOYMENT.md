# Hardware deployment guide

How to take this project from the MuJoCo simulation to a real robot, what we know, what we do not know, and what to do first.

[← Back to the main README](README.md)

> [!CAUTION]
> **Nothing in this repository has ever been run on a real robot.** This guide is a map made by people who only worked in simulation. Where we know something from our own measurements it says **measured**. Where it comes from general knowledge of UMI or robotics it says **general knowledge, verify**. Where we simply do not know it says **unknown** or **TODO**.

---

## 1. Read this first (5 minutes)

1. **Our trained policy was trained on simulated images from a scripted program, not on real demonstrations.** Putting it on a real robot will most likely fail, because real camera images look very different from our simple simulated ones (*likely*, never tested).
2. **The simulation result (about 82%) depends on two hand-tuned offsets** (+12 mm sideways, +12 mm up) whose cause we never found. On hardware they may be wrong, unnecessary, or replaced by a different problem. Treat them as something to re-measure, not something to copy.
3. **What does carry over:** the task definition, the success check, the evaluation method, the list of mistakes, the safety-minded process, and the model interface (what goes in, what comes out).
4. **The realistic route** is to use UMI's own real-world pipeline for hardware, and use this repo as a reference and test bench. See section 3.

---

## 2. What transfers and what does not

| Thing | Transfers to hardware? | Notes |
|---|---|---|
| Task and success definition (cup within 3 cm of saucer centre, upright, released) | ✅ Yes | Use the same definition so numbers are comparable |
| Evaluation method (fixed seed blocks, count successes, judge by rollouts not loss) | ✅ Yes | On hardware, "seeds" become repeatable cup and saucer layouts you mark on the table |
| Lessons in the README (mistakes, dead ends) | ✅ Yes | Especially: confirm on layouts you never tuned on |
| Model structure and interface (section 4) | ✅ Yes | Same inputs and outputs |
| `final_policy.ckpt` | ⚠️ Unlikely | Trained only on simulated images |
| The two offsets and the grasp-phase fix | ⚠️ Idea only | The idea (control the hand target near the grasp) may help, the numbers will not carry over |
| Simulator scripts (`scene.py`, `env.py`, expert) | ❌ No | They only describe the simulated world |
| Robot, gripper and camera drivers | ❌ Not in this repo | See section 8 |

---

## 3. Choose your path

| Path | What it means | Effort | Risk | Our view |
|---|---|---|---|---|
| **A. Use UMI's official real-world pipeline** | Follow UMI's own hardware and software setup. Collect real demonstrations (or start from the official dataset and checkpoint), train, deploy | High | Medium | **Recommended** |
| **B. Try our simulated policy on the real robot** | Connect robot and camera, feed real images to `final_policy.ckpt` | Medium | Very high (likely 0% success) | Only as a quick curiosity, with the dry-run steps in section 9 |
| **C. Reduce the simulation gap first** | Make simulated images and physics look more like the real ones (textures, lighting, camera model, friction), retrain, then try B | High | High | Only if real demonstrations are impossible |

Notes on path A:
- The project name ("Cup Arrangement") and our cup-on-saucer task come from UMI's own cup arrangement task. The official dataset (`cup_in_the_wild.zarr.zip`, 17 GB) was downloaded on the lab PC at some point. Whether it was used for anything is not recorded here. UMI's repository README says how to get its data and pretrained checkpoints (*general knowledge, verify in your clone*).
- The UMI paper and site: https://umi-gripper.github.io/

---

## 4. The policy's interface (what goes in, what comes out)

Everything below is **measured** from `docs/final_policy_training_config.yaml` and the checkpoint.

**Inputs** (the last 2 time steps of each):
- `camera0_rgb`: one wrist camera image, 224 px
- `robot0_eef_pos`: hand position
- `robot0_eef_rot_axis_angle`: hand rotation
- `robot0_eef_rot_axis_angle_wrt_start`: hand rotation relative to where the episode started
- `robot0_gripper_width`: how open the gripper is

**Outputs:** a plan of 16 future steps, of which **8 are carried out** before the policy looks again.

**Size:** 171,273,610 parameters, about 1 GB of GPU memory while running, 16 denoising steps per decision.

To print the exact shapes of every input and of the action (do this before wiring anything):

```bash
cd ~/umi_mujoco/umi_release
python - << 'EOF'
import yaml
c = yaml.safe_load(open('docs/final_policy_training_config.yaml'))
print(yaml.dump(c['task']['shape_meta']))
EOF
```

**Time per decision: unknown (never measured).** This matters a lot for hardware, because the robot keeps moving while the policy thinks. To measure it on the computer you plan to use (this runs the policy 50 times on random input of the right shape and averages; it has not been run successfully yet, so fix any error it shows):

```bash
cd ~/umi_mujoco/universal_manipulation_interface
export PYTHONPATH=$PWD
python - << 'EOF'
import os, time, torch, dill, hydra
ck = os.path.expanduser('~/umi_mujoco/final_results/final_policy.ckpt')
payload = torch.load(open(ck,'rb'), map_location='cpu', pickle_module=dill)
cfg = payload['cfg']
ws = hydra.utils.get_class(cfg._target_)(cfg)
ws.load_payload(payload, exclude_keys=None, include_keys=None)
policy = ws.ema_model if cfg.training.get('use_ema', True) else ws.model
policy.eval().cuda()
obs = {}
for k, a in cfg.task.shape_meta.obs.items():
    obs[k] = torch.rand(1, a.get('horizon', 2), *a['shape'], device='cuda')
with torch.no_grad():
    for _ in range(5): policy.predict_action(obs)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(50): policy.predict_action(obs)
    torch.cuda.synchronize()
print(f"{(time.time()-t0)/50*1000:.0f} ms per decision (average of 50)")
EOF
```
Write the result here: `______ ms` (TODO).

**Time spacing of the actions: unknown.** In simulation the controller ran at 50 Hz, but how many simulation steps one policy step covers is not written down here. TODO: read `code/record_demos_v2.py` and `code/eval_sim19.py` (search for `steps`, `dt`, `hz`). On hardware the spacing of the 16 planned steps must match what the policy was trained on.

---

## 5. The simulated setup, as a specification to match

Use this to see how far a real setup is from the simulated one (**measured**, from `code/scene.py` and the README).

| | Simulation |
|---|---|
| Arm and gripper | Franka Panda with the Franka hand (from MuJoCo Menagerie) |
| Hand reference point (TCP) | 0.1034 m beyond the hand frame |
| Wrist camera | on the hand, 100° vertical field of view, simple pinhole (no fisheye), position (0.06, 0, 0.03) m from the hand frame |
| Policy image size | 224 px |
| Control rate | 50 Hz |
| Cup | radius 2.5 cm, height 6 cm, mass 0.1 kg |
| Saucer | radius 6.5 cm |
| Gripper width when holding the cup | about 4.9 cm |
| Table | flat, plain colour, heights measured above the table surface |
| Scene look | plain flat colours, simple lighting, no clutter |

Differences you should expect on hardware (*likely*): image appearance (textures, glare, shadows, auto-exposure), lens distortion, camera delay, gripper softness and friction, arm tracking error and calibration error, object variety.

---

## 6. The two offsets and the grasp-phase fix

What the simulated controller does to the policy's target for the hand (all numbers in simulation coordinates):

```text
sideways target   = policy x  + 12 mm
height target     = max(policy z, 27 mm) + 12 mm      # floor first, then the offset
```

- Because the floor is applied before the offset, the hand could never be commanded below 39 mm. The cup centre is at 30 mm.
- **The fix:** while the hand is below 8 cm with the gripper open, and until a grasp is detected, the height offset is 0 instead of 12 mm. After the grasp it ramps back to 12 mm over 30 steps. This raised success from 60% to 78% on the same seeds.
- **Without both offsets the simulated policy scored about 1 in 60**, even though it copies demonstrations accurately (position error 0.58 cm against 3.53 cm for "stay still"). Why is **unknown**. Several explanations were tested and ruled out (README, "Dead ends").

**What this means for hardware:** assume the same kind of mismatch (policy output vs where the hand really ends up) may appear, but do not assume the same numbers. Measure first (section 9, step 5).

---

## 7. Hardware you need to decide on

Fill in this table before ordering or borrowing anything. TODO(next person).

| Decision | Options | Our simulation used | Decided |
|---|---|---|---|
| Robot arm | the lab's available arm; UMI's public code describes Franka and UR-type arms (*general knowledge, verify*) | Franka Panda | ____ |
| Gripper | must report its opening width and match what the policy expects (about 4.9 cm closed on the cup) | Franka hand | ____ |
| Camera | wrist-mounted, same viewpoint as the demonstrations. UMI uses a wide-angle (fisheye) wrist camera (*general knowledge, verify*) | pinhole 100° | ____ |
| Computer | a GPU with room for the policy (about 1 GB to run, 8 GB to train at batch 8) | RTX 3050 8 GB | ____ |
| Robot control link | whatever real-time interface the arm offers | none (simulated) | ____ |
| Cup and saucer | real objects of a size close to ours (cup 5 cm wide, 6 cm tall; saucer 13 cm wide) | simulated | ____ |
| Emergency stop | **mandatory** | none | ____ |

---

## 8. What UMI's own repository already provides

*General knowledge, verify on the lab PC before relying on it.* To our knowledge UMI's public repository contains real-world robot, gripper and camera code and evaluation scripts for real runs. Find out what is there:

```bash
cd ~/umi_mujoco/universal_manipulation_interface
cat ~/umi_mujoco/umi_release/env/umi_repo_commit.txt
ls umi/real_world scripts_real 2>/dev/null
grep -ril "franka\|wsg\|ur5\|rtde" --include=*.py . 2>/dev/null | head -30
```

TODO(next person): list here which robots and grippers the code supports, and which hardware setup instructions UMI gives:
```
Supported robots:   ______
Supported grippers: ______
Camera setup:       ______
Latency settings:   ______
```

Our repo adds nothing in this area. All our code talks to the simulator.

---

## 9. Step-by-step plan (suggested, in this order)

Do not skip steps. Do not move to the next one until the previous works.

1. **Read** the UMI paper and the real-world part of UMI's repo (section 8). Decide your path (section 3).
2. **Get the arm and gripper moving without any AI.** Command a hand position, read it back. Check the gripper width reading is correct.
3. **Get camera images.** Show the live image at the policy's size (224 px). Compare it with a frame from the demonstration data. Mount the camera at the same viewpoint.
4. **Dry run, no motion.** Feed live images and hand readings to the policy, print the planned actions, and send nothing to the robot. Check the numbers look sensible (right direction, right size). Measure the time per decision here.
5. **Check policy against reality on recorded data.** Take real demonstration data if you have it and measure how far the policy output is from the demonstrated action. In simulation this check is `--check_grasp` in `code/eval_sim15.py`.
6. **First movement: very slow, small, with someone on the emergency stop.** No cup. Limit speed and range.
7. **First pick attempts** with an empty hand near the cup, then with the gripper closing on the cup.
8. **Evaluate properly.** Mark cup and saucer layouts on the table (your "seeds"), at least 20 per setting, same success rule as the simulation (cup within 3 cm of saucer centre, upright, released). Use layouts you did not tune on for the final number.
9. **Tune only one thing at a time** and compare on identical layouts. Expect noise: with 20 trials, a gap of fewer than 4-5 cases means nothing.

---

## 10. Safety checklist (general good practice)

- ☐ A working emergency stop within reach, tested before every session
- ☐ Maximum speed and acceleration limits set low for first runs
- ☐ Hard limits on the workspace (the hand cannot go below the table or far outside the layout)
- ☐ Clip every action from the policy to a safe range before sending it
- ☐ Gripper force limited to what the cup needs
- ☐ Dry run (print only) before every new policy or setting
- ☐ Nobody inside the robot's reach during runs
- ☐ Stop immediately on any unexpected motion and look at the logged actions before retrying
- ☐ A written log of every run: policy file (use its fingerprint), settings, layout, result

---

## 11. Open questions for the hardware work

1. Does the real policy also need hand-target offsets? Measure policy output vs achieved hand pose on the real arm before adding any.
2. How long does one policy decision take on the real computer, and does the robot need latency compensation? (UMI describes a way to handle latency, *general knowledge, verify in the paper*.)
3. Which camera and mount reproduce the viewpoint of the training data?
4. Can real demonstrations be collected (UMI handheld gripper, or teleoperation), or must path C be used?
5. What cup and saucer will be used, and do their sizes match the simulation?

---

## 12. Facts we could not verify (so you know)

- Which robots and grippers UMI's current code supports (section 8)
- The time per decision and the time between planned actions (section 4)
- Whether the official cup dataset or checkpoint was ever tested with our setup
- Whether any part of our simulated policy transfers to real images

If you find any of these out, please write the answer into this file.
