# UMI in MuJoCo: a robot arm picks up a cup and places it on a saucer

A diffusion-policy robot controller ([UMI](https://umi-gripper.github.io/), Universal Manipulation Interface) trained and tested entirely inside a MuJoCo simulation. A simulated Franka Panda arm with a wrist camera picks up a cup and places it on a saucer.

**Status (19 Nov 2026):** about **81% success** (389 of 480 episodes on fresh random layouts). Simulation only, never run on a real robot. Work is paused on purpose so the project can be handed over. This README is written so a newcomer can understand everything in 10-15 minutes.

**Evidence labels used below:** *measured* = a number a real run printed. *likely* = reasoning we believe but never proved. *unknown* = we do not know.

---

## 1. TL;DR

1. The policy learns to copy demonstrations very accurately, but when it controls the robot itself, it **almost always fails (about 1 in 60)** unless we add two hand-tuned offsets: a 12 mm sideways shift and a 12 mm upward shift of the hand target. *Why* the offsets are needed is **unknown**. This is the biggest open question.
2. With the offsets, success was about 60%. Removing the upward shift **only during the final descent and grasp** raised it to about **81%**. This is the one change that made a real difference.
3. Most of what we tried (retraining, extra data, retries, gating, finger changes) gave **no gain**. Section 8 lists them so you do not repeat them.
4. The remaining 19% of failures are mostly a cup that slips out of the gripper while being carried. A promising lead is a grip that closes off-centre on the cup (not confirmed).
5. Everything is documented, every number below has a log in `results/logs/`, and the final policy checkpoint is shared separately (link in section 5).

---

## 2. The project in plain words

- **Task:** pick up a cup, place it on a saucer, let go without disturbing it, and move away without pushing it.
- **Success means:** cup within 3 cm of the saucer centre, resting on top of it, upright, and no longer held.
- **Simulator:** MuJoCo. Cup radius 2.5 cm, mass 0.1 kg. Cup and saucer positions are random each episode (cup on one side of the table, saucer on the other, random rotation). Control runs at 50 Hz.
- **Data:** 1,800 demonstrations made by a scripted "expert" program (about 98-100% success). The expert has random jitter and random start poses.
- **Policy:** UMI's diffusion U-Net policy (`DiffusionUnetTimmPolicy`, DDIM sampling), 224 px images, no image augmentation, batch size 8, 10 epochs x 2,500 steps. The image encoder is trained, not frozen.
- **Evaluation:** run the trained policy on unseen random layouts ("seeds") and count successes. The final controller uses only the gripper width and hand pose. It does **not** use simulator ground truth (ground truth is used only for diagnosis and for scoring).
- **Hardware:** one Ubuntu PC with an RTX 3050 (8 GB). That is why batch size is only 8.

---

## 3. Results (all measured)

Noise is about +/- 5 episodes out of 120. A gap of 10 or more is meaningful, and it should repeat on other seed sets.

### Final result: 389 / 480 = 81%

| Fresh seeds | Success | Log |
|---|---|---|
| 1000-1119 | 94/120 (78%) | `results/logs/fresh_zg0.txt` |
| 2000-2119 | 87/120 (72.5%) | `results/logs/xb_0.012.txt` |
| 3000-3119 | 100/120 (83%) | `results/logs/xc_0.012_3000.txt` |
| 4000-4119 | 108/120 (90%) | `results/logs/xc_0.012_4000.txt` |

Sets vary a lot (72% to 90%), so quote "about 80%" with the spread, not one number. The fix value and offset were chosen on other seeds (160-279), so these four sets were not used for tuning.

### How we got there

| Setting | Success |
|---|---|
| Policy alone, early (seeds 100-159) | about 25-35% (scripted expert about 100%) |
| All offsets 0, no fix | 1/60 |
| Offsets, old behaviour (12 mm upward shift all the time) | 72/120 = 60% (seeds 1000-1119) |
| Upward shift halved during grasp (`--zbias_grasp 0.006`) | 91/120 = 76% (same seeds) |
| **Upward shift removed during grasp (`--zbias_grasp 0.0`, final)** | **94/120 = 78% (same seeds)** |

Same order on the tuning seeds 160-279: 68/120 (57%), 85/120 (71%), 100/120 (83%).

### The sideways offset has a narrow sweet spot (with the fix, out of 120)

| `--xbias` | Seeds 2000 | Seeds 3000 | Seeds 4000 |
|---|---|---|---|
| 0.004 | 51 | | |
| 0.006 | 83 | | |
| 0.008 | 96 | 101 | 96 |
| **0.012 (kept)** | 87 | 100 | 108 |
| 0.016 | 54 | | |

0.008 and 0.012 tie overall (293 vs 295 of 360). Both ends of the range collapse.

### Why the remaining runs fail (seeds 1000-1119, 26 failures)

Slipped out of a closed grip: 11. Released on the saucer, then knocked it off: 7. Never lifted: 5. Released far from the saucer: 2. Saucer pushed away: 1. The cup was lifted above 6 cm in 115 of 120 episodes, so grasping works most of the time.

Failures depend on the layout, not luck: re-running the same seeds fails on nearly the same ones.

---

## 4. Repository map

| You want... | Look at |
|---|---|
| **Evaluation (final)** | `code/eval_sim19.py` |
| Evaluation history (each file was made by patching the previous one) | `code/eval_sim.py` ... `eval_sim19.py`, patch scripts `code/make_eval_sim15.py` ... `make_eval_sim19.py` |
| **Training** | Not our own code. We use UMI's own `train.py`. Command and settings: section 5.5 and `docs/final_policy_training_config.yaml` |
| The one change we made inside UMI | `code/umi_repo_patch.diff` (learning-rate scheduler horizon) |
| Simulation scene and environment | `code/scene.py`, `code/env.py` (`UmiSimEnv`) |
| Scripted expert and demo recording | `code/expert.py`, `code/expert2.py` (used for demos), `code/record_demos_v2.py` |
| DAgger experiments (extra data from the policy's own mistakes) | `code/record_dagger.py`, `code/merge_dagger.py`, `docs/training_commands_dagger.txt` |
| Analysis helper | `code/rt_summary.py` (per-seed table from the per-step trace) |
| Logs behind every number | `results/logs/` |
| Saved training settings | `docs/final_policy_training_config.yaml`, `docs/training_configs/` |
| Software versions | `env/requirements.txt`, `env/python_version.txt`, `env/umi_repo_commit.txt` |
| Trained model | Not in git (1.3 GB). See section 5 |
| Earlier setup notebooks | Removed from the current tree; still in git history |

`eval_sim.py` and `eval_sim3.py` are untouched baselines. Never edit them. New ideas go in a new `eval_sim20.py`.

---

## 5. Reproduce

Our layout was `~/umi_mujoco/` with the UMI repo in `universal_manipulation_interface/` and our scripts in `sim/umi_sim/`. Copy `code/*.py` into `sim/umi_sim/` and run from `sim/`. Some scripts and saved configs contain absolute paths from the original machine (`/home/iitgn/umi_mujoco/...`); change them to yours.

### 5.1 Setup
1. Ubuntu, Python 3.9, torch 2.1.0+cu121, mujoco 3.2.3 (full list in `env/requirements.txt`).
2. Clone UMI at the commit in `env/umi_repo_commit.txt`, then apply `code/umi_repo_patch.diff`.
3. Franka model: MuJoCo Menagerie `franka_emika_panda`, expected at `sim/mujoco_menagerie/franka_emika_panda`.
4. Always activate the environment first, and set rendering:
```
source ~/umi_mujoco/umi_env/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
```

### 5.2 Get the trained model
Download `final_policy.ckpt` (1.3 GB): **(ADD LINK HERE)**. Put it at `~/umi_mujoco/final_results/final_policy.ckpt`.

### 5.3 Reproduce the 81% (the final command)
```
CK=~/umi_mujoco/final_results/final_policy.ckpt
python umi_sim/eval_sim19.py --ckpt $CK --lead 6 --xbias 0.012 --zbias 0.012 --zbias_grasp 0.0 \
  --trace --retreat --xbias_mode pregrasp --glead 0 --rt_trig 0.055 --rt_full 0.078 --rt_back -0.10 \
  --n 120 --start_seed 1000 > ~/umi_mujoco/exp/NAME.txt 2>&1
grep -c '^GTX [0-9]* success' ~/umi_mujoco/exp/NAME.txt
```
Takes about 7 minutes per 120 episodes. The result lines appear only when the run finishes, so a count of 0 mid-run is normal. Repeat with `--start_seed 2000`, `3000`, `4000`; expect roughly 94, 87, 100, 108 (within noise). Runs are not bitwise reproducible. Add `--videos N --video_full` for full video clips and `--reltrace` for a per-step csv.

### 5.4 Reproduce the baselines
- **Old setting (60%):** the same command without `--zbias_grasp 0.0`, seeds 1000-1119, expect about 72/120.
- **No offsets (about 1/60):** the same command with `--xbias 0 --zbias 0` (*likely* how we ran it; check `results/logs`).
- **Use seeds 5000 and above for anything new.** Earlier blocks were used for tuning or confirmation.

### 5.5 Retrain the policy
The original training command was not saved. The settings below were recovered from the checkpoint itself (`docs/final_policy_training_config.yaml`): dataset `cup_sim_v2.zarr.zip`, batch 8, learning rate 0.0003, cosine schedule, 500 warmup steps, 10 epochs, 2,500 steps per epoch, encoder not frozen, no image augmentation. The command is **rebuilt**: those settings are confirmed, the other flags are copied from our later DAgger runs. Run from the UMI repo:
```
python train.py --config-name=train_diffusion_unet_timm_umi_workspace \
  task.dataset_path=<path>/cup_sim_v2.zarr.zip \
  dataloader.batch_size=8 val_dataloader.batch_size=8 dataloader.num_workers=2 val_dataloader.num_workers=1 \
  logging.mode=disabled training.rollout_every=100000 training.num_epochs=10 training.max_train_steps=2500 \
  training.max_val_steps=100 training.checkpoint_every=1 training.lr_scheduler=cosine training.lr_warmup_steps=500 \
  policy.obs_encoder.transforms=null checkpoint.topk.k=3
```
Training loss tells you little. **Judge a policy only by running it** (section 5.3).

### 5.6 Regenerate the demos (8 GB, not included)
Run `code/record_demos_v2.py` to produce `cup_sim_v2.zarr.zip` (1,800 episodes). Its arguments are not documented yet; read the top of the script. *Unknown:* what changed in `scene.py` on 2 Nov.

---

## 6. How the final controller works

The policy outputs a target position for the hand. Our controller adds two fixed offsets to it: **+12 mm sideways (`--xbias`)** and **+12 mm up (`--zbias`)**. We also floor the target height at 27 mm above the table. Because the floor is applied before the upward shift, the hand could **never be commanded below 39 mm** (27 + 12). The cup centre is at 30 mm, and our scripted expert grasps at about 35 mm.

**The fix** (`--zbias_grasp 0.0`, `--zb_zone 0.08`): while the hand is lower than 8 cm with the gripper open, and until a grasp is detected, the upward offset is replaced by `--zbias_grasp` (0 = none). After the grasp it ramps back to 12 mm over 30 steps. The median grasp height dropped to 23 mm, and 109 of 120 grasps were below 38 mm. That the fingers were "too high on the cup" is the *likely* mechanism, not proven. The result itself is measured.

---

## 7. The journey

| When (2026) | What we did | What we learned |
|---|---|---|
| Oct | Built the scene in MuJoCo through a series of setup notebooks. Recorded 1,800 expert demos. Trained on Colab first, then on the lab PC. | The full pipeline (scene, recording, training, evaluation) works. |
| Late Oct - 4 Nov | First evaluations. | Policy alone 25-35% vs expert about 100%. Most failures happened at or after releasing the cup. |
| 5 Nov | Added a retreat motion after release (open, lift, back away). Tested its timing. | About 43% pooled. Retreat timing was not the cause (24-26/60 vs 28/60). |
| 6 Nov | Tested offsets and gates. | The two offsets were essential. Adding the 12 mm up-shift: 68/120 (57%) vs 51/120. The sideways offset had a narrow sweet spot (0.006: 19/120, 0.018: 37/120). Height gates and a stillness gate gave no gain. |
| 7 Nov | Forced the *expert* to grasp at different heights. Built DAgger (300 extra episodes recorded from the policy's own situations). | The expert works at 28-40 mm and fails above about 45 mm. The policy failed more when it closed high: 36% failure below 45 mm, 71% at 55 mm or more. Suspected "the policy grabs too high". |
| 9-10 Nov | Retrained on the DAgger data twice. | 63% and 66% on tuning seeds, 56% on fresh seeds. No real gain. Without offsets it still scored about 1/120. |
| 14 Nov | Checked how well the policy copies the demos. | Very well: position error 0.58 cm vs 3.53 cm for "stay still". So bad imitation is not the problem. |
| 15 Nov | Tested two more ideas. Then removed the up-shift during grasp only. | Correcting for arm lag: 0/60. Retrying after a drop: noise. **Removing the up-shift at grasp: 100/120 on tuning seeds.** |
| 19 Nov | Confirmed on four fresh sets. | 389/480 = 81%. Documented and handed over. |

---

## 8. Mistakes, wrong beliefs and lessons

### Things we believed that turned out wrong
- "The image encoder is frozen." Wrong, `freeze_encoder` is false.
- "The gap between simulation and reality explains the failures." Wrong, the demos come from this same simulator.
- "The grip command dips and the cup slips." Refuted.
- "The fingertip pads are too small." Not supported.
- "DAgger data does not cover grasping and carrying." Wrong.
- "The sideways offset 0.008 beats 0.012." Only on one seed block. Overall they tie.
- "Grasp height explains the remaining failures." Not supported. Failures and successes close at similar heights (39 vs 41 mm). An 8-episode comparison that suggested otherwise was a small-sample accident.
- "Re-running a seed may end differently." Mostly it does not; the same seeds fail again.
- The per-step finger force column (`fn` in the trace csv) is **not reliable**. Do not use it as evidence.

### Mistakes in how we worked
- **Tuning and judging on the same seeds.** Numbers chosen on tuning seeds were optimistic (83% on tuning seeds vs 78% on fresh ones). Always confirm on seeds you never used.
- **Quoting one number.** A single fresh set said 78%; four sets give 81% with a 72-90% range.
- **Spending days on big, expensive ideas** (two retrains, DAgger rounds, gates) before looking closely at what the controller did at the moment of grasp. The real gain came from a small change in the controller, not from more training.
- **Overstating causes.** We wrote "about 35% of failures are because of the grasp" when the data only said "associated with".

### Practical slips that cost time
- Forgetting `source umi_env/bin/activate`: `python` is not found and chained commands silently print nothing.
- Disk full while merging DAgger data (the merged file is 12 GB). Check free space first.
- Running two GPU jobs at once.
- Expecting results mid-run (they print only at the end).

### Habits that worked (keep them)
- Never overwrite a script: write a patch script that creates the next `eval_simNN.py`.
- Never modify the baselines, the checkpoint, the base dataset, or the frozen copies.
- Fixed seed blocks, never reused (see section 5.4).
- Judge by success counts on identical seeds, and call something an improvement only if it clearly exceeds noise.

---

## 9. Dead ends: do not retry

| Idea | Result |
|---|---|
| Correcting for arm tracking lag (`--trackfix`) | 0/60 |
| Retry after a dropped cup (`--regrasp`) | 71 vs 68 of 120 (noise); none of 31 re-grasps succeeded |
| Frame mismatch between camera and controller | none found in the code |
| Longer training (80k steps) | no gain |
| More DAgger rounds | 63% / 66% / 56%, no demonstrated gain (never tried with the new grasp fix) |
| Height gating, stillness gating, taller finger pads, friction, 4x gripper force, squeeze threshold | no gain |
| Changing the sideways offset again | ties across three seed sets |
| Changing the object to a flat or square one | not justified |

---

## 10. Open questions and next steps

1. **Why does the policy need the offsets at all?** It copies demos almost perfectly, yet fails without them. Unexplained. Solving this properly would matter more than any further tuning.
2. **Off-centre grip lead.** In a 60-episode trace, closing the gripper 10 mm or more off the cup centre went with high failure (7 of 11 failed vs 5 of 49 otherwise). The 10 mm line was chosen after looking at the data, and n is small. Confirm on seeds 5000+ with `--reltrace`, then `python umi_sim/rt_summary.py NAME.txt`.
3. **Crooked-grip check.** If the finger gap is already narrow as the cup leaves the table (in the 7 carry failures it was 45-48 mm vs 49.4 mm normally), the robot could notice, put the cup down, and retry before lifting. This differs from `--regrasp`, which only reacted after the cup had fallen.
4. **Fade the sideways offset just before closing** so the hand lands closer to the cup centre. A gamble, since whole-run changes did not help.
5. Try `--zbias_grasp 0.0` on the DAgger policies (no retraining needed).

---

## 11. Toward real hardware (opinion, nothing here was tested)

UMI was built for real robots, and our controller uses only gripper width and hand pose, which a real robot can provide. But:
- The two offsets were tuned for **this** simulator and their cause is unknown. Expect to re-tune them, or better, find the real cause first.
- Camera images here are rendered at 224 px. Real cameras will differ in lighting, colour and delay.
- Nothing about camera calibration, robot interface or a real gripper exists in this repo.
- Real cups and tables have different friction and mass than ours.

---

## 12. Credit and contact

Built on [UMI](https://umi-gripper.github.io/) (real-stanford/universal_manipulation_interface). Franka model from MuJoCo Menagerie.

Authors and contact: *(add names and emails here)*
