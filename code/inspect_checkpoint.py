"""Print the size and key settings of a UMI / diffusion-policy checkpoint.

Usage (environment active, from anywhere):
    python code/inspect_checkpoint.py ~/umi_mujoco/final_results/final_policy.ckpt

It only READS the checkpoint. Nothing is modified.
Paste the printed numbers into the "Model size" table in README.md.

Needs: torch and dill (both already in the UMI environment).
"""
import os
import sys
from collections import OrderedDict

import dill
import torch


def millions(n):
    return round(n / 1e6, 2)


def count(state):
    return sum(v.numel() for v in state.values() if hasattr(v, "numel"))


def main(path):
    path = os.path.expanduser(path)
    print("file:", path)
    print("file size on disk: %.2f GB" % (os.path.getsize(path) / 1e9))

    payload = torch.load(open(path, "rb"), pickle_module=dill, map_location="cpu")
    print("\ntop-level keys:", list(payload.keys()))

    state_dicts = payload.get("state_dicts", {})
    print("state_dict entries:", list(state_dicts.keys()))
    for name, sd in state_dicts.items():
        if isinstance(sd, (dict, OrderedDict)):
            n = count(sd)
            if n:
                print("  %-12s %12d parameters = %s M" % (name, n, millions(n)))

    # Pick the main weights (prefer the averaged copy that evaluation normally uses).
    model_sd = None
    for key in ("ema_model", "model"):
        if key in state_dicts:
            model_sd = state_dicts[key]
            chosen = key
            break
    if model_sd is None:
        print("\nCould not find 'model' or 'ema_model' weights.")
        return

    total = count(model_sd)
    print("\n=== Network size (from '%s') ===" % chosen)
    print("total parameters: %d = %s million" % (total, millions(total)))
    print("weights size if stored as fp32: %.0f MB" % (total * 4 / 1e6))

    # Split by the first part of each weight name.
    groups = OrderedDict()
    for k, v in model_sd.items():
        if not hasattr(v, "numel"):
            continue
        g = k.split(".")[0]
        groups[g] = groups.get(g, 0) + v.numel()
    print("\nparameters by part (first name in the weight path):")
    for g, n in groups.items():
        print("  %-20s %s M  (%.1f%%)" % (g, millions(n), 100.0 * n / total))

    enc = sum(n for g, n in groups.items() if "obs_encoder" in g)
    print("\nimage/observation encoder: %s M" % millions(enc))
    print("rest (diffusion network):  %s M" % millions(total - enc))

    # Key settings from the saved config.
    cfg = payload.get("cfg", None)
    if cfg is None:
        print("\nNo config found inside the checkpoint.")
        return
    print("\n=== Settings saved inside the checkpoint ===")
    try:
        from omegaconf import OmegaConf

        text = OmegaConf.to_yaml(cfg)
    except Exception:
        text = str(cfg)

    wanted = (
        "n_obs_steps", "n_action_steps", "horizon", "num_inference_steps",
        "freeze_encoder", "model_name", "shape_meta", "batch_size",
        "num_epochs", "max_train_steps", "lr", "dataset_path",
        "obs_encoder", "noise_scheduler",
    )
    for line in text.splitlines():
        s = line.strip()
        if any(s.startswith(w) or (w + ":") in s for w in wanted):
            print(line)
    print("\n(Full config: save it with `docs/final_policy_training_config.yaml`.)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
