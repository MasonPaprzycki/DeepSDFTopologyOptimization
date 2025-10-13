import os
import json
import torch
import glob, re
import numpy as np

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
        """
        Return npz files based only on the split dict.
        `data_source` is unused here but kept for API compatibility.
        """
        npyfiles = []
        for split_name, classes in split.items():
            for class_name, instances in classes.items():
                for instance_name in instances:
                    npyfiles.append(f"{instance_name}.npz")
        return npyfiles

    deep_data.get_instance_filenames = get_instance_filenames

patch_get_instance_filenames()

# ------------------------
# Safe save helpers
# ------------------------
def safe_save_npz(path, **arrays):
    """
    Save a npz file safely on Windows. Overwrites existing files.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)

def safe_save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# ------------------------
# Train a single analytic shape (preserve all latents)
# ------------------------
def trainAShape(model_name, sdf_function, scene_ids, resume=True, domainRadius=1.0, sdf_parameters=None):
    
    if sdf_parameters is None:
        sdf_parameters = []

    root = os.path.join("trained_models", model_name)
    split_name = "train"

    split_dir = os.path.join(root, "split")
    samples_dir = os.path.join(root, "SdfSamples")
    latent_dir = os.path.join(root, "LatentCodes")
    model_params_dir = os.path.join(root, "ModelParameters")
    for d in [root, split_dir, samples_dir, latent_dir, model_params_dir]:
        os.makedirs(d, exist_ok=True)

    # -------------------- specs.json --------------------
    specs_path = os.path.join(root, "specs.json")
    if not os.path.exists(specs_path):
        specs = {
            "Description": f"Train DeepSDF on analytic {model_name} shapes.",
            "NetworkArch": "deep_sdf_decoder",
            "DataSource": root,
            "TrainSplit": "split/TrainSplit.json",
            "NetworkSpecs": {
                "dims": [128, 128, 128, 128, 128, 128],
                "dropout": [0, 1, 2, 3, 4, 5],
                "dropout_prob": 0.2,
                "norm_layers": [0, 1, 2, 3, 4, 5],
                "latent_in": [2],
                "xyz_in_all": False,
                "use_tanh": False,
                "latent_dropout": False,
                "weight_norm": True,
                "geom_dimension": 3 + len(sdf_parameters)
            },
            "CodeLength": 1,
            "NumEpochs": 500,
            "SnapshotFrequency": 100,
            "AdditionalSnapshots": [1, 5],
            "LearningRateSchedule": [
                { "Type": "Step", "Initial": 0.001, "Interval": 250, "Factor": 0.5 },
                { "Type": "Constant", "Value": 0.001 }
            ],
            "SamplesPerScene": 2048,
            "ScenesPerBatch": 1,
            "DataLoaderThreads": 1,
            "ClampingDistance": 0.1,
            "CodeRegularization": True,
            "CodeRegularizationLambda": 1e-4,
            "CodeBound": 1.0
        }
        os.makedirs(os.path.dirname(specs_path), exist_ok=True)
        with open(specs_path, "w") as f:
            json.dump(specs, f, indent=2)
        print(f"[INFO] Wrote specs.json for {model_name}")

    # -------------------- TrainSplit.json --------------------
    split_path = os.path.join(split_dir, "TrainSplit.json")
    split_dict = {}
    if os.path.exists(split_path):
        with open(split_path) as f:
            split_dict = json.load(f)
    split_dict.setdefault(split_name, {}).setdefault(model_name, [])

    # -------------------- Generate SDF samples --------------------
    for scene_id in scene_ids:
        scene_key = f"{model_name.lower()}_{scene_id:03d}"
        samples_path = os.path.join(samples_dir, f"{scene_key}.npz")
        if not os.path.exists(samples_path) or scene_key not in split_dict[split_name][model_name]:
            print(f"[INFO] Generating SDF samples for {scene_key}…")
            n_points = 50_000
            queries = torch.empty(n_points, 3 + len(sdf_parameters))
            queries[:, :3] = (torch.rand(n_points, 3) * 2 - 1) * domainRadius
            for i, (low, high) in enumerate(sdf_parameters):
                queries[:, 3 + i] = torch.rand(n_points) * (high - low) + low

            sdf_values = sdf_function(queries).squeeze(1)
            all_samples = torch.cat([queries, sdf_values.unsqueeze(1)], dim=1).numpy()
            clamp_dist = 0.1
            sdf_idx = 3 + len(sdf_parameters)

            pos = all_samples[np.abs(all_samples[:, sdf_idx]) < clamp_dist]
            neg = all_samples[np.abs(all_samples[:, sdf_idx]) >= clamp_dist]

            samples_per_scene = 2048
            n_pos = min(len(pos), samples_per_scene // 2)
            n_neg = samples_per_scene - n_pos

            if len(pos) > 0:
                pos = pos[np.random.choice(len(pos), n_pos, replace=False)]
            if len(neg) > 0:
                neg = neg[np.random.choice(len(neg), n_neg, replace=False)]

            os.makedirs(os.path.dirname(samples_path), exist_ok=True)
            np.savez_compressed(samples_path, pos=pos, neg=neg)
            split_dict[split_name][model_name].append(scene_key)
        else:
            print(f"[INFO] Skipping sample generation for {scene_key}")

    with open(split_path, "w") as f:
        json.dump(split_dict, f, indent=2)

    # -------------------- Initialize per-scene latents --------------------
    for scene_id in scene_ids:
        scene_key = f"{model_name.lower()}_{scene_id:03d}"
        scene_latent_file = os.path.join(latent_dir, f"{scene_key}.pth")
        if not os.path.exists(scene_latent_file):
            latent_code = 0.01 * torch.randn(1, specs["CodeLength"])
            torch.save({"latent_codes": {scene_key: latent_code}}, scene_latent_file)
            print(f"[INFO] Created latent file for {scene_key}")
        else:
            print(f"[INFO] Latent file exists for {scene_key}; skipping initialization")

    # -------------------- Train DeepSDF for each scene separately --------------------
    def pick_latest_file_by_mtime(paths):
        existing = [p for p in paths if os.path.exists(p)]
        if not existing:
            return None
        return max(existing, key=os.path.getmtime)

    def pick_best_resume_ckpt(model_params_dir):
        latest_pth = os.path.join(model_params_dir, "latest.pth")
        if os.path.exists(latest_pth):
            return latest_pth
        snaps = glob.glob(os.path.join(model_params_dir, "snapshot_*.pth"))
        if not snaps:
            return None
        max_idx = -1
        best_snap = None
        for s in snaps:
            m = re.search(r"snapshot_(\d+)\.pth$", s)
            if m:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
                    best_snap = s
        return best_snap or pick_latest_file_by_mtime(snaps)

    continue_from = pick_best_resume_ckpt(model_params_dir) if resume else None

    for scene_id in scene_ids:
        scene_key = f"{model_name.lower()}_{scene_id:03d}"
        print(f"[INFO] Training scene {scene_key} …")
        # DeepSDF will pick up latent from LatentCodes/scene_key.pth
        training.train_deep_sdf(
            experiment_directory=root,
            data_source=root,
            continue_from=continue_from,
            batch_split=1
        )

    print("[INFO] All scenes trained. Existing latent files untouched; new scenes got their own latents.")
