import os
import json
import tempfile
import shutil
import torch
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
def trainAShape(model_name, sdf_function, scene_ids, resume=True, domainRadius=1.0, sdf_parameters=[]):
    root = os.path.join("trained_models", model_name)
    split_name = "train"
    
    # Directories
    split_dir = os.path.join(root, "split")
    samples_dir = os.path.join(root, "SdfSamples")
    latent_dir = os.path.join(root, "LatentCodes")
    model_params_dir = os.path.join(root, "ModelParameters")
    for d in [root, split_dir, samples_dir, latent_dir, model_params_dir]:
        os.makedirs(d, exist_ok=True)

    # specs.json
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
                {"Type": "Step", "Initial": 0.0005, "Interval": 250, "Factor": 0.5},
                {"Type": "Step", "Initial": 0.001, "Interval": 250, "Factor": 0.5}
            ],
            "SamplesPerScene": 2048,
            "ScenesPerBatch": 1,
            "DataLoaderThreads": 1,
            "ClampingDistance": 0.1,
            "CodeRegularization": True,
            "CodeRegularizationLambda": 1e-4,
            "CodeBound": 1.0
        }
        safe_save_json(specs_path, specs)
        print(f"[INFO] Wrote specs.json for {model_name}")

    # TrainSplit.json
    split_path = os.path.join(split_dir, "TrainSplit.json")
    split_dict = {}
    if os.path.exists(split_path):
        with open(split_path) as f:
            split_dict = json.load(f)
    split_dict.setdefault(split_name, {}).setdefault(model_name, [])

    # -------------------- Generate SDF samples per scene --------------------
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
            pos = all_samples[np.abs(all_samples[:, 3]) < clamp_dist]
            neg = all_samples[np.abs(all_samples[:, 3]) >= clamp_dist]
            safe_save_npz(samples_path, pos=pos, neg=neg)
            split_dict[split_name][model_name].append(scene_key)
        else:
            print(f"[INFO] Skipping sample generation for {scene_key}")

    safe_save_json(split_path, split_dict)

    # -------------------- Preserve old latent codes --------------------
    latest_latent_file = os.path.join(latent_dir, "latest.pth")
    if os.path.exists(latest_latent_file):
        existing_latents = torch.load(latest_latent_file, map_location="cpu").get("latent_codes", {})
    else:
        existing_latents = {}

    # Train DeepSDF
    print(f"[INFO] Training model '{model_name}' with {len(scene_ids)} scenes…")
    training.train_deep_sdf(
        experiment_directory=root,
        data_source=root,
        continue_from=None,
        batch_split=1
    )
    # -------------------- Merge old latents with new ones --------------------
    new_latents = torch.load(latest_latent_file, map_location="cpu").get("latent_codes", {})
    merged_latents = {**existing_latents, **new_latents}
    torch.save({"latent_codes": merged_latents}, latest_latent_file)
    print(f"[INFO] Training complete. Latent codes updated in {latest_latent_file}")