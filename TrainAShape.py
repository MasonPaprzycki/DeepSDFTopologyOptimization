import os
import json
import torch
import glob, re
import numpy as np
import shutil
import DeepSDFStruct.deep_sdf.data as deep_data
import DeepSDFStruct.deep_sdf.training as training
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
def trainAShape(
    model_name,
    sdf_function,
    scene_ids,
    resume=True,
    domainRadius=1.0,
    sdf_parameters=None,
    latentDim=1,
    FORCE_ONLY_FINAL_SNAPSHOT=False
    ):
    
    if sdf_parameters is None:
        sdf_parameters = []

    # ---------------- Folder setup ----------------
    root = os.path.join("trained_models", model_name)
    split_dir = os.path.join(root, "split")
    model_params_dir = os.path.join(root, "ModelParameters")
    scenes_dir = os.path.join(root, "Scenes")
    samples_dir = os.path.join(root, "SdfSamples")
    top_latent_dir = os.path.join(root, "LatentCodes")  # DeepSDF top-level

    for d in [root, split_dir, model_params_dir, scenes_dir, samples_dir, top_latent_dir]:
        os.makedirs(d, exist_ok=True)

    # ---------------- Specs ----------------
    specs_path = os.path.join(root, "specs.json")
    if os.path.exists(specs_path):
        with open(specs_path) as f:
            specs = json.load(f)
    else:
        specs = {
            "Description": f"Train DeepSDF on analytic {model_name} shapes.",
            "NetworkArch": "deep_sdf_decoder",
            "DataSource": root,
            "TrainSplit": "split/TrainSplit.json",
            "NetworkSpecs": {
                "dims": [128]*6,
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
                {"Type": "Step", "Initial":0.001, "Interval":250, "Factor":0.5},
                {"Type": "Constant", "Value":0.001}
            ],
            "SamplesPerScene": 5000,
            "ScenesPerBatch": 1,
            "DataLoaderThreads": 1,
            "ClampingDistance": 0.1,
            "CodeRegularization": True,
            "CodeRegularizationLambda": 1e-4,
            "CodeBound": 1.0
        }

    if FORCE_ONLY_FINAL_SNAPSHOT:
        specs["SnapshotFrequency"] = specs.get("NumEpochs", specs["SnapshotFrequency"])
        specs["AdditionalSnapshots"] = [specs.get("NumEpochs")]

    with open(specs_path, "w") as f:
        json.dump(specs, f, indent=2)

    # ---------------- TrainSplit.json ----------------
    train_split_path = os.path.join(split_dir, "TrainSplit.json")
    if os.path.exists(train_split_path):
        with open(train_split_path) as f:
            split_dict = json.load(f)
    else:
        split_dict = {"train": {}}
    split_dict.setdefault("train", {}).setdefault(model_name, [])
    for scene_id in scene_ids:
        key = f"{model_name.lower()}_{scene_id:03d}"
        if key not in split_dict["train"][model_name]:
            split_dict["train"][model_name].append(key)
    with open(train_split_path, "w") as f:
        json.dump(split_dict, f, indent=2)

    # ---------------- Training Loop ----------------
    for scene_id in scene_ids:
        scene_key = f"{model_name.lower()}_{scene_id:03d}"
        scene_folder = os.path.join(scenes_dir, f"{scene_id:03d}")
        scene_latent_dir = os.path.join(scene_folder, "LatentCodes")
        os.makedirs(scene_latent_dir, exist_ok=True)

        # ---------------- Skip if per-scene latest.pth exists ----------------
        scene_latest_file = os.path.join(scene_latent_dir, "latest.pth")
        if resume and os.path.exists(scene_latest_file):
            scene_data = torch.load(scene_latest_file, map_location="cpu")
            epochs_done = scene_data.get("epochs", {}).get(scene_key, 0)
            if epochs_done >= specs["NumEpochs"]:
                print(f"[INFO] Scene {scene_key} already fully trained, skipping...")
                continue
        else:
            epochs_done = 0

        # ---------------- SDF Samples ----------------
        samples_file = os.path.join(samples_dir, f"{scene_key}.npz")
        if not os.path.exists(samples_file):
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

        # ---------------- Train ----------------
        training.train_deep_sdf(
            experiment_directory=root,
            data_source=root,  # expects SdfSamples/ here
            continue_from=None if epochs_done == 0 else "latest",
            batch_split=1
        )

        # ---------------- Reload trained latent from top-level ----------------
        top_latest_file = os.path.join(top_latent_dir, "latest.pth")
        top_data = torch.load(top_latest_file, map_location="cpu")
        latent_trained = top_data["latent_codes"][scene_key]

        # ---------------- Save per-scene snapshots ----------------
        snapshots = specs["AdditionalSnapshots"] + [specs["NumEpochs"]]
        for snap in snapshots:
            snap_file = os.path.join(scene_latent_dir, f"{snap}.pth")
            torch.save({"latent_codes": {scene_key: latent_trained}, "epochs": {scene_key: snap}}, snap_file)

        # ---------------- Write per-scene latest.pth after training ----------------
        torch.save({"latent_codes": {scene_key: latent_trained}, "epochs": {scene_key: specs["NumEpochs"]}},
                   scene_latest_file)

        print(f"[INFO] Finished training scene {scene_key}")
