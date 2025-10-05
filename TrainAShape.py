
import os
import json
import tempfile
import shutil
import torch
import numpy as np
import DeepSDFStruct.deep_sdf.training as training
import DeepSDFStruct.deep_sdf.data as deep_data

# ------------------------
# Limit CPU threads globally
# ------------------------
os.environ["OMP_NUM_THREADS"] = "7"
os.environ["MKL_NUM_THREADS"] = "7"
torch.set_num_threads(7)
torch.set_num_interop_threads(2)

# ------------------------
# Patch DeepSDF loader so it looks for flat SdfSamples
# ------------------------
def patch_get_instance_filenames():
    def get_instance_filenames(data_source, split):
        npyfiles = []
        for split_name, classes in split.items():
            for class_name, instances in classes.items():
                for instance_name in instances:
                    filename = f"{instance_name}.npz"
                    npyfiles.append(filename)   # <-- only the string, not a tuple
        return npyfiles
    deep_data.get_instance_filenames = get_instance_filenames

patch_get_instance_filenames()

# ------------------------
# Safe save helpers
# ------------------------
def safe_save_npz(path, **arrays):
    dir_ = os.path.dirname(path)
    os.makedirs(dir_, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=dir_, suffix=".npz") as tmp:
        tmp_name = tmp.name
    np.savez_compressed(tmp_name, **arrays)
    shutil.move(tmp_name, path)

def safe_save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = tempfile.NamedTemporaryFile('w', delete=False, dir=os.path.dirname(path))
    json.dump(data, tmp, indent=2)
    tmp.flush()
    os.fsync(tmp.fileno())
    tmp.close()
    shutil.move(tmp.name, path)

# ------------------------
# Train a single analytic shape
# ------------------------
def trainAShape(model_name, sdf_function, scene_id, resume=True, domainRadius=1.0, sdf_parameters=[]):
    root = os.path.join("trained_models", model_name)
    split_name = "train"

    # Directories
    split_dir = os.path.join(root, "split")        # where TrainSplit.json lives
    samples_dir = os.path.join(root, "SdfSamples") # all .npz go flat here
    latent_dir = os.path.join(root, "LatentCodes")
    model_params_dir = os.path.join(root, "ModelParameters")

    for d in [root, split_dir, samples_dir, latent_dir, model_params_dir]:
        os.makedirs(d, exist_ok=True)

    scene_str = f"{scene_id:03d}" if isinstance(scene_id, int) else str(scene_id)
    scene_key = f"{model_name.lower()}_{scene_str}"

    # ------------------------ specs.json ------------------------
    specs_path = os.path.join(root, "specs.json")
    if not os.path.exists(specs_path):
        specs = {
            "Description": f"Train DeepSDF on multiple analytic {model_name} shapes.",
            "NetworkArch": "deep_sdf_decoder",
            "DataSource": root,
            "TrainSplit": "split/TrainSplit.json",
            "ReconstructionSplit": "",
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
                "geom_dimension": 3 + len(sdf_parameters),
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
        print("[INFO] Wrote specs.json to", specs_path)

    # ------------------------ TrainSplit.json ------------------------
    split_path = os.path.join(split_dir, "TrainSplit.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            split_dict = json.load(f)
    else:
        split_dict = {}

    split_dict.setdefault(split_name, {}).setdefault(model_name, [])

    # ------------------------ Generate SDF samples ------------------------
    samples_path = os.path.join(samples_dir, f"{scene_key}.npz")

    if os.path.exists(samples_path) and scene_key in split_dict[split_name][model_name]:
        print(f"[INFO] Skipping sample generation for {scene_key} (already exists).")
    else:
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

        if scene_key not in split_dict[split_name][model_name]:
            split_dict[split_name][model_name].append(scene_key)

        safe_save_json(split_path, split_dict)

    # ------------------------ Train DeepSDF ------------------------
    checkpoint_file = os.path.join(model_params_dir, "latest.pth")

    # Skip training if checkpoint exists AND scene is in TrainSplit.json
    if os.path.exists(checkpoint_file) and scene_key in split_dict[split_name][model_name]:
        print(f"[INFO] Scene {scene_key} already trained, skipping training.")
        return  # <-- early exit

    continue_from = checkpoint_file if resume and os.path.exists(checkpoint_file) else None
    if continue_from:
        print(f"[INFO] Resuming training from {continue_from}")
    else:
        print("[INFO] Starting training from scratch…")

    training.train_deep_sdf(
        experiment_directory=root,
        data_source=root,
        continue_from=continue_from
    )


    print(f"[INFO] Training complete for scene {scene_key}")
