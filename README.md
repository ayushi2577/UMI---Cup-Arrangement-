# UMI in MuJoCo: Franka picks up a cup and places it on a saucer

A UMI diffusion policy (real-stanford/universal_manipulation_interface) run inside a MuJoCo simulation. A Franka Panda with a wrist camera picks up a cup and places it on a saucer. **Simulation only. Nothing here has been tested on a real robot.**

## Result
About **81% success (389 of 480 episodes)** on four sets of 120 fresh seeds (sets: 78%, 72.5%, 83%, 90%). Quote "about 80%" with the spread, not one number.

| Setting | Success |
|---|---|
| Policy alone, no hand-tuned offsets | about 1 in 60 |
| Offsets, old behaviour | 60% (72/120, seeds 1000-1119) |
| Offsets + grasp-phase fix (final) | 78% on the same seeds; 81% over four sets |

Success means: cup within 3 cm of the saucer centre, on top of it, upright, and no longer held by the gripper.

## What the fix does
The old controller added a constant 12 mm upward nudge to the hand. During the final descent and grasp this likely kept the fingers high on the cup (likely, not proven). The fix (`--zbias_grasp 0.0`) removes the nudge only during that phase and ramps it back after the grasp is detected. The controller uses only gripper width and end-effector pose, no simulator ground truth.

## Folder layout
- `code/` scene, expert, demo recording, and every evaluation script (`eval_sim.py` ... `eval_sim19.py`). Each `eval_simNN.py` was made by a `make_eval_simNN.py` patch script from the previous one. `eval_sim.py` and `eval_sim3.py` are untouched baselines. **Final script: `eval_sim19.py`.** Originally these lived in `~/umi_mujoco/sim/umi_sim/`; put them back there to run.
- `results/logs/` raw logs behind every number above.
- `docs/` training settings, notes, `freeze_78_README.txt`.
- `env/` `requirements.txt`, Python version, the exact UMI repo commit. `code/umi_repo_patch.diff` is our one change to the UMI repo (LR scheduler horizon).
- `model/` not in git (1.3 GB). The checkpoint `final_policy.ckpt` is shared separately: (ADD LINK HERE).

## Setup
Ubuntu, RTX 3050 8 GB, Python 3.9, torch 2.1.0+cu121, mujoco 3.2.3 (full list in `env/requirements.txt`).
1. Clone UMI, check out the commit in `env/umi_repo_commit.txt`, apply `code/umi_repo_patch.diff`.
2. Franka model: MuJoCo Menagerie `franka_emika_panda`, expected at `~/umi_mujoco/sim/mujoco_menagerie/franka_emika_panda`.
3. For rendering: `export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`.

## Data (not included, 8 GB)
1,800 scripted-expert demos made with `code/record_demos_v2.py` (uses the scripted expert in `expert2.py`), saved as `cup_sim_v2.zarr.zip`. Regenerate with that script. (Its exact arguments are not yet documented here.)

## Training
The original training command was not saved. The settings below were recovered from the checkpoint itself (`docs/final_policy_training_config.yaml`): dataset `cup_sim_v2.zarr.zip`, batch size 8, learning rate 0.0003, cosine schedule, 500 warmup steps, 10 epochs, 2500 steps per epoch, encoder not frozen, no image augmentation. This command is **rebuilt**: the settings above are confirmed, the remaining flags are copied from our later DAgger runs (`docs/training_commands_dagger.txt`). Run from the UMI repo folder:
```
python train.py --config-name=train_diffusion_unet_timm_umi_workspace task.dataset_path=<path>/cup_sim_v2.zarr.zip dataloader.batch_size=8 val_dataloader.batch_size=8 dataloader.num_workers=2 val_dataloader.num_workers=1 logging.mode=disabled training.rollout_every=100000 training.num_epochs=10 training.max_train_steps=2500 training.max_val_steps=100 training.checkpoint_every=1 training.lr_scheduler=cosine training.lr_warmup_steps=500 policy.obs_encoder.transforms=null checkpoint.topk.k=3
```
Judge a policy only by rollouts, not by training loss.

## Evaluation (the final command)
```
CK=<path>/final_policy.ckpt
python umi_sim/eval_sim19.py --ckpt $CK --lead 6 --xbias 0.012 --zbias 0.012 --zbias_grasp 0.0 --trace --retreat --xbias_mode pregrasp --glead 0 --rt_trig 0.055 --rt_full 0.078 --rt_back -0.10 --n 120 --start_seed 5000 > NAME.txt 2>&1
grep -c '^GTX [0-9]* success' NAME.txt
```
About 7 minutes per 120 episodes. Add `--videos N --video_full` for videos, `--reltrace` for per-step traces. Use seeds 5000 and up for anything new (earlier blocks were used). Noise is about +/-5 episodes at n=120; a difference of 10 or more is meaningful.

## Known problems
- **Unexplained:** without the hand-tuned offsets (`--xbias 0.012 --zbias 0.012`) the policy almost always fails, even though it copies demos accurately (position error 0.58 cm vs 3.53 cm for "stay still"). Why the offsets are needed is unknown. Arm tracking error, frame mismatch, retry-after-drop, finger geometry and training length were all ruled out as the main cause.
- **Remaining failures (about 19%):** in a 60-episode trace, 7 of 12 failures lost the cup while carrying it, 4 carried it fine and failed at placing or release, 1 never lifted it. Closing the grip 10 mm or more off the cup centre went with much higher failure (7 of 11 vs 5 of 49). This is a lead from a small sample, not a confirmed cause.
- **Not tried:** the DAgger policies with `--zbias_grasp 0.0`; fading the sideways offset just before closing.
- **Do not retry** (no gain): changing the sideways offset again, height gating, taller pads, friction, stronger gripper force, more DAgger rounds, retry after a dropped cup.

## For real hardware (not done, opinion only)
The controller uses only gripper width and end-effector pose, which a real robot can provide. But the offsets were tuned for this simulator and are unexplained, so expect to re-tune them or find the real cause first. The camera setup, robot interface and a real gripper are not built.

## Credit
Built on UMI: https://umi-gripper.github.io/
