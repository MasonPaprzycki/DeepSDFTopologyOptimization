import os
import json
import torch
import glob, re
import numpy as np
import shutil

import DeepSDFStruct.deep_sdf.training as training
import DeepSDFStruct.deep_sdf.data as deep_data
import DeepSDFStruct.deep_sdf.networks.deep_sdf_decoder as Decoder

# ------------------------
# Limit CPU threads globally
# ------------------------

NUM_CORES = 16
os.environ["OMP_NUM_THREADS"] = str(NUM_CORES)
os.environ["MKL_NUM_THREADS"] = str(NUM_CORES)
torch.set_num_threads(NUM_CORES)
torch.set_num_interop_threads(min(4, NUM_CORES // 2))

# ------------------------
# Patch DeepSDF loader for flat SdfSamples
# ------------------------
def patch_get_instance_filenames():
    def get_instance_filenames(data_source, split):
        npyfiles = []
        for split_name, classes in split.items():
            for class_name, instances in classes.items():
                for instance_name in instances:
                    npyfiles.append(f"{instance_name}.npz")
        return npyfiles
    deep_data.get_instance_filenames = get_instance_filenames

patch_get_instance_filenames()

# ------------------------
# Train a model on analytic shapes safely
# ------------------------

def trainAShape(model_name, sdf_function, scene_ids,
                resume=True, domainRadius=1.0, sdf_parameters=None,
                latentDim=1,
                FORCE_ONLY_FINAL_SNAPSHOT=False):
    """
    Robust re-implementation of your training driver for analytic shapes,
    with safer checkpoint handling and metadata persistence.
    """

    if sdf_parameters is None:
        sdf_parameters = []

    # --- Folder setup ---
    root = os.path.join("trained_models", model_name)
    split_name = "train"
    split_dir = os.path.join(root, "split")
    samples_dir = os.path.join(root, "SdfSamples")
    latent_dir = os.path.join(root, "LatentCodes")
    model_params_dir = os.path.join(root, "ModelParameters")
    for d in [root, split_dir, samples_dir, latent_dir, model_params_dir]:
        os.makedirs(d, exist_ok=True)

    # --- specs.json ---
    specs_path = os.path.join(root, "specs.json")
    if not os.path.exists(specs_path):
        specs = {
            "Description": f"Train DeepSDF on analytic {model_name} shapes.",
            "NetworkArch": "deep_sdf_decoder",
            "DataSource": root,
            "TrainSplit": "split/TrainSplit.json",
            "NetworkSpecs": {
                "dims": [128] * 6,
                "dropout": list(range(6)),
                "dropout_prob": 0.2,
                "norm_layers": list(range(6)),
                "latent_in": [2],
                "xyz_in_all": False,
                "use_tanh": False,
                "latent_dropout": False,
                "weight_norm": True,
                "geom_dimension": 3 + len(sdf_parameters)
            },
            "CodeLength": latentDim,
            "NumEpochs": 500,
            "SnapshotFrequency": 100,
            "AdditionalSnapshots": [1, 5],
            "LearningRateSchedule": [
                {"Type": "Step", "Initial": 0.001, "Interval": 250, "Factor": 0.5},
                {"Type": "Constant", "Value": 0.001}
            ],
            "SamplesPerScene": 2048,
            "ScenesPerBatch": 1,
            "DataLoaderThreads": 1,
            "ClampingDistance": 0.1,
            "CodeRegularization": True,
            "CodeRegularizationLambda": 1e-4,
            "CodeBound": 1.0
        }
    else:
        with open(specs_path, "r") as f:
            specs = json.load(f)

    # Apply FORCE_ONLY_FINAL_SNAPSHOT if requested
    if FORCE_ONLY_FINAL_SNAPSHOT:
        specs["SnapshotFrequency"] = specs.get("NumEpochs", specs["SnapshotFrequency"])
        specs["AdditionalSnapshots"] = [specs.get("NumEpochs")]

    # always re-save specs after possible modification
    with open(specs_path, "w") as f:
        json.dump(specs, f, indent=2)

    # --- TrainSplit.json ---
    split_path = os.path.join(split_dir, "TrainSplit.json")
    split_dict = {}
    if os.path.exists(split_path):
        with open(split_path) as f:
            split_dict = json.load(f)
    split_dict.setdefault(split_name, {}).setdefault(model_name, [])
    for scene_id in scene_ids:
        scene_key = f"{model_name.lower()}_{scene_id:03d}"
        if scene_key not in split_dict[split_name][model_name]:
            split_dict[split_name][model_name].append(scene_key)
    with open(split_path, "w") as f:
        json.dump(split_dict, f, indent=2)

    # --------------------
    # Helper utilities
    # --------------------
    def epoch_from_filename(path):
        base = os.path.splitext(os.path.basename(path))[0]
        # handle 'latest' explicitly as unknown epoch
        if base == "latest":
            return None
        try:
            # for files named "1.pth", "5.pth", etc.
            return int(base)
        except ValueError:
            # fallback for snapshot-style names
            m = re.search(r"snapshot_(\d+)$", base)
            return int(m.group(1)) if m else None

    def read_checkpoint_safe(path):
        try:
            return torch.load(path, map_location="cpu")
        except Exception:
            return None

    def checkpoint_epoch_from_dict(d):
        if not isinstance(d, dict):
            return None
        for k in ("epoch", "epochs", "iter", "iteration", "iterations", "step"):
            if k in d:
                try:
                    return int(d[k])
                except Exception:
                    pass
        for maybe in ("training", "meta", "state"):
            if maybe in d and isinstance(d[maybe], dict):
                for k in ("epoch", "iter", "iteration"):
                    if k in d[maybe]:
                        try:
                            return int(d[maybe][k])
                        except Exception:
                            pass
        return None

    def list_candidate_checkpoints():
        cands = []
        latest = os.path.join(model_params_dir, "latest.pth")
        if os.path.exists(latest):
            cands.append(latest)

        # accept both snapshot_###.pth and numeric ###.pth files
        snaps = glob.glob(os.path.join(model_params_dir, "*.pth"))
        snaps_sorted = sorted(
            snaps,
            key=lambda p: (
                epoch_from_filename(p)
                if epoch_from_filename(p) is not None
                else float("inf"),
                os.path.getmtime(p),
            ),
        )
        cands.extend(snaps_sorted)
        return cands



    def find_best_checkpoint_with_scene(scene_key):
        """
        Find the best model checkpoint (under ModelParameters/) that can be used
        to resume training for `scene_key`. We inspect LatentCodes/ for any file
        that contains the scene_key (either a numeric latent snapshot or a per-scene file).
        - If a numeric latent file (e.g. LatentCodes/5.pth) contains the scene,
        prefer the matching ModelParameters/5.pth (if present).
        - If a per-scene latent file (e.g. LatentCodes/sphere_000.pth) contains it,
        fall back to the most-recent ModelParameters/*.pth (latest by mtime).
        Returns (model_params_checkpoint_path, epoch_or_None).
        """
        best = (None, -1)

        # 1) scan latent files to find ones that contain this scene_key
        latent_files = sorted(
            glob.glob(os.path.join(latent_dir, "*.pth")),
            key=lambda p: os.path.getmtime(p),
        )
        for latent_fp in latent_files:
            data = read_checkpoint_safe(latent_fp)
            if not isinstance(data, dict):
                continue
            latent_codes = data.get("latent_codes", {})
            if scene_key not in latent_codes:
                continue

            latent_epoch = epoch_from_filename(latent_fp)
            if latent_epoch is not None:
                candidate_model = os.path.join(model_params_dir, f"{latent_epoch}.pth")
                if os.path.exists(candidate_model):
                    if isinstance(latent_epoch, int) and latent_epoch > (best[1] if isinstance(best[1], int) else -1):
                        best = (candidate_model, latent_epoch)
                    continue

                # fallback to latest.pth
                latest_model = os.path.join(model_params_dir, "latest.pth")
                if os.path.exists(latest_model):
                    latest_data = read_checkpoint_safe(latest_model)
                    latest_epoch = (
                        checkpoint_epoch_from_dict(latest_data)
                        or epoch_from_filename(latest_model)
                        or -1
                    )
                    epoch_a = latest_epoch if isinstance(latest_epoch, (int, float)) else -1
                    epoch_b = best[1] if isinstance(best[1], (int, float)) else -1
                    if epoch_a > epoch_b:
                        best = (latest_model, latest_epoch)
                    continue

            # If latent filename isn't numeric (e.g., "sphere_000.pth"), choose most recent model params
            global_ckpts = sorted(
                glob.glob(os.path.join(model_params_dir, "*.pth")),
                key=lambda p: os.path.getmtime(p),
            )
            if global_ckpts:
                latest_model = global_ckpts[-1]
                latest_epoch = (
                    checkpoint_epoch_from_dict(read_checkpoint_safe(latest_model))
                    or epoch_from_filename(latest_model)
                    or -1
                )
                epoch_a = latest_epoch if isinstance(latest_epoch, (int, float)) else -1
                epoch_b = best[1] if isinstance(best[1], (int, float)) else -1
                if epoch_a > epoch_b or best[0] is None:
                    best = (latest_model, latest_epoch)

        # 2) If we didn’t find any latent file that includes this scene, check the model params themselves
        if best[0] is None:
            for ck in list_candidate_checkpoints():
                data = read_checkpoint_safe(ck)
                if not isinstance(data, dict):
                    continue
                latent_codes = data.get("latent_codes", {})
                if scene_key in latent_codes:
                    epoch = (
                        checkpoint_epoch_from_dict(data)
                        or epoch_from_filename(ck)
                        or -1
                    )
                    epoch_a = epoch if isinstance(epoch, (int, float)) else -1
                    epoch_b = best[1] if isinstance(best[1], (int, float)) else -1
                    if epoch_a > epoch_b:
                        best = (ck, epoch)

        return (best[0], best[1] if best[0] is not None else None)

    def is_scene_fully_trained(scene_key):
        target_epoch = int(specs.get("NumEpochs", 0))
        ckpt, epoch = find_best_checkpoint_with_scene(scene_key)
        return bool(ckpt and epoch is not None and int(epoch) >= target_epoch)

    def save_scene_latent_from_checkpoint(scene_key, ckpt_path, to_path):
        """
        Given a model checkpoint path (from ModelParameters), find the appropriate
        latent file that contains the scene_key and save out a per-scene latent.
        Works both when latent file is numeric (LatentCodes/5.pth) or per-scene (LatentCodes/sphere_000.pth).
        """
        if ckpt_path is None:
            return False

        # First try to extract epoch from ckpt_path and use LatentCodes/<epoch>.pth
        epoch = epoch_from_filename(ckpt_path)
        if epoch is not None:
            latent_path = os.path.join(latent_dir, f"{epoch}.pth")
            if os.path.exists(latent_path):
                data = read_checkpoint_safe(latent_path)
                if isinstance(data, dict):
                    latent_codes = data.get("latent_codes", {})
                    if scene_key in latent_codes:
                        latent = latent_codes[scene_key]
                        if isinstance(latent, torch.Tensor):
                            latent = latent.detach().cpu()
                        torch.save({"latent_codes": {scene_key: latent}}, to_path)
                        return True

        # If that didn't work, scan all latent files to find one containing scene_key
        for latent_fp in sorted(glob.glob(os.path.join(latent_dir, "*.pth")), key=lambda p: os.path.getmtime(p), reverse=True):
            data = read_checkpoint_safe(latent_fp)
            if not isinstance(data, dict):
                continue
            latent_codes = data.get("latent_codes", {})
            if scene_key in latent_codes:
                latent = latent_codes[scene_key]
                if isinstance(latent, torch.Tensor):
                    latent = latent.detach().cpu()
                torch.save({"latent_codes": {scene_key: latent}}, to_path)
                return True

        return False
    # --------------------
    # Per-scene training loop
    # --------------------
    for scene_id in scene_ids:
        scene_key = f"{model_name.lower()}_{scene_id:03d}"
        scene_latent_file = os.path.join(latent_dir, f"{scene_key}.pth")
        samples_file = os.path.join(samples_dir, f"{scene_key}.npz")

        # Check full training before latent skip
        if is_scene_fully_trained(scene_key):
            ckpt, epoch = find_best_checkpoint_with_scene(scene_key)
            if ckpt:
                if save_scene_latent_from_checkpoint(scene_key, ckpt, scene_latent_file):
                    print(f"[INFO] Scene {scene_key} already fully trained (epoch {epoch}). Latent saved → {scene_latent_file}")
                    continue

        # If latent exists but not fully trained, ignore latent and continue
        if os.path.exists(scene_latent_file):
            try:
                ld = torch.load(scene_latent_file, map_location="cpu")
                if not (isinstance(ld, dict) and "latent_codes" in ld and scene_key in ld["latent_codes"]):
                    os.remove(scene_latent_file)
            except Exception:
                try:
                    os.remove(scene_latent_file)
                except Exception:
                    pass

        # Generate SDF samples if missing
        if not os.path.exists(samples_file):
            print(f"[INFO] Generating SDF samples for {scene_key}…")
            n_points = 50_000
            queries = torch.empty(n_points, 3 + len(sdf_parameters))
            queries[:, :3] = (torch.rand(n_points, 3) * 2 - 1) * domainRadius
            for i, (low, high) in enumerate(sdf_parameters):
                queries[:, 3 + i] = torch.rand(n_points) * (high - low) + low
            sdf_vals = sdf_function(queries).squeeze(1)
            data = torch.cat([queries, sdf_vals.unsqueeze(1)], dim=1).numpy()
            idx = 3 + len(sdf_parameters)
            pos = data[np.abs(data[:, idx]) < specs["ClampingDistance"]]
            neg = data[np.abs(data[:, idx]) >= specs["ClampingDistance"]]
            n_pos = min(len(pos), specs["SamplesPerScene"] // 2)
            n_neg = specs["SamplesPerScene"] - n_pos
            if len(pos) > 0:
                pos = pos[np.random.choice(len(pos), n_pos, replace=False)]
            if len(neg) > 0:
                neg = neg[np.random.choice(len(neg), n_neg, replace=False)]
            np.savez_compressed(samples_file, pos=pos, neg=neg)

        if not os.path.exists(scene_latent_file):
            latent_code = 0.01 * torch.randn(1, int(specs.get("CodeLength", latentDim)))
            torch.save({"latent_codes": {scene_key: latent_code}}, scene_latent_file)

        # Do NOT delete checkpoints that may contain other scenes
        # We only prune snapshots that *exclusively* contain this scene.
        target_epoch = int(specs.get("NumEpochs", 0))
        for ckpt in glob.glob(os.path.join(model_params_dir, "*.pth")):
            data = read_checkpoint_safe(ckpt)
            if not isinstance(data, dict):
                continue
            latent_codes = data.get("latent_codes", {})
            if scene_key not in latent_codes:
                continue
            if len(latent_codes.keys()) > 1:
                # shared checkpoint, never delete
                continue
            epoch = checkpoint_epoch_from_dict(data) or epoch_from_filename(ckpt)
            if epoch is not None and int(epoch) < target_epoch:
                try:
                    os.remove(ckpt)
                    print(f"[DEBUG] Removed partial snapshot {ckpt} for scene {scene_key} (epoch {epoch} < {target_epoch}).")
                except Exception:
                    pass

        # Resume checkpoint logic
        continue_from = None
        if resume:
            ckpt_with_scene, epoch_with_scene = find_best_checkpoint_with_scene(scene_key)
            if ckpt_with_scene:
                continue_from = ckpt_with_scene[:-4]
                print(f"[INFO] Resuming {scene_key} from epoch {epoch_with_scene} ({ckpt_with_scene})")
            else:
                # fallback: latest global checkpoint
                global_ckpts = sorted(
                    glob.glob(os.path.join(model_params_dir, "*.pth")),
                    key=lambda p: os.path.getmtime(p),
                )
                if global_ckpts:
                    latest_global = global_ckpts[-1]
                    continue_from = latest_global[:-4]
                    print(f"[INFO] No scene-specific checkpoint found; resuming from global latest {latest_global}")
                else:
                    print(f"[INFO] No checkpoints found at all for {scene_key}; starting fresh.")

        print(f"[INFO] Training scene {scene_key} … (resume_from: {continue_from})")
        training.train_deep_sdf(
            experiment_directory=root,
            data_source=root,
            continue_from=continue_from,
            batch_split=1,
        )

        # After training, ensure final checkpoint and latent extraction
        candidate_ckpt, candidate_epoch = find_best_checkpoint_with_scene(scene_key)
        final_ckpt = candidate_ckpt
        final_epoch = candidate_epoch
        if final_ckpt:
            save_scene_latent_from_checkpoint(scene_key, final_ckpt, scene_latent_file)
            print(f"[INFO] Saved latent for {scene_key} from checkpoint {os.path.basename(final_ckpt)} (epoch {final_epoch})")
        else:
            print(f"[WARN] No checkpoint containing {scene_key} after training.")

        #Ensure latest.pth exists for visualizer
        latest_ckpt = os.path.join(model_params_dir, "latest.pth")
        if not os.path.exists(latest_ckpt) and final_ckpt:
            try:
                shutil.copy(final_ckpt, latest_ckpt)
                print(f"[INFO] Copied {final_ckpt} → latest.pth")
            except Exception:
                pass

    print("[INFO] Training loop complete for all specified scenes.")
