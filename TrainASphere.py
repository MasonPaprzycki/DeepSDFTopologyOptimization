import os
import json
import tempfile
import shutil
import numpy as np
import torch

# -------------------------------------------------------------------
# Limit CPU threads BEFORE heavy torch ops
# -------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

import DeepSDFStruct.sdf_primitives as sdf_primitives
import DeepSDFStruct.deep_sdf.training as training

# -------------------------------------------------------------------
# Directory layout
# -------------------------------------------------------------------
root = "trained_models/sphere"
class_name = "sphere"
split_name = "train"
scene_id   = "sphere_000"

samples_dir = os.path.join(root, "SdfSamples", class_name, split_name)
splits_dir  = os.path.join(root, "splits")
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(splits_dir, exist_ok=True)

# -------------------------------------------------------------------
# Generate SDF samples
# -------------------------------------------------------------------
print("Generating SDF samples …")
basicSphere = sdf_primitives.SphereSDF(center=[0, 0, 0], radius=1.0)

n_points = 50_000
queries  = torch.rand(n_points, 3) * 2 - 1   # uniform cube [-1,1]^3
sdf      = basicSphere._compute(queries).squeeze(1)

all_samples = torch.cat([queries, sdf.unsqueeze(1)], dim=1).numpy()

clamp_dist  = 0.1
mask        = np.abs(all_samples[:, 3]) < clamp_dist
pos         = all_samples[mask]
neg         = all_samples[~mask]

print(f"pos samples: {pos.shape}, neg samples: {neg.shape}")

# -------------------------------------------------------------------
# Safe save: write to a temp file and move
# -------------------------------------------------------------------
def safe_save_npz(path, **arrays):
    dir_ = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(delete=False, dir=dir_, suffix=".npz") as tmp:
        tmp_name = tmp.name
    np.savez_compressed(tmp_name, **arrays)
    shutil.move(tmp_name, path)


samples_path = os.path.join(samples_dir, f"{scene_id}.npz")
safe_save_npz(samples_path, pos=pos, neg=neg)

# quick sanity check before training
data = np.load(samples_path)
print("Saved npz keys:", data.files)
print("pos shape:", data["pos"].shape, "neg shape:", data["neg"].shape)

# -------------------------------------------------------------------
# Train split JSON
# -------------------------------------------------------------------
split_path = os.path.join(splits_dir, "TrainSplit.json")
split_dict = {class_name: {split_name: [scene_id]}}

def safe_save_json(path, data):
    tmp = tempfile.NamedTemporaryFile('w', delete=False,
                                      dir=os.path.dirname(path))
    json.dump(data, tmp, indent=2)
    tmp.flush()
    os.fsync(tmp.fileno())
    tmp.close()
    shutil.move(tmp.name, path)

safe_save_json(split_path, split_dict)
print("Wrote TrainSplit.json to", split_path)

# -------------------------------------------------------------------
# Launch training
# -------------------------------------------------------------------
if __name__ == "__main__":
    # specs.json must contain:
    # "DataSource": "trained_models/sphere",
    # "TrainSplit": "splits/TrainSplit.json"
    print("Starting training …")
    training.train_deep_sdf(root, root)
    print("Done.")
