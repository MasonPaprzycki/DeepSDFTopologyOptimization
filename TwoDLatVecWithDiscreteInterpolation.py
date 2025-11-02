import os
import sys
import json
import torch
import numpy as np
from VisualizeAShape import visualize_a_shape
import DeepSDFStruct.sdf_primitives as sdf_primitives
from Model import Model, Scenes  # your unified classes

# ============================================================
# Experiment Configuration
# ============================================================
EXPERIMENT_NAME = "TwoDLatentVecWithDiscreteInterpolationAndLinScaling"
EXPERIMENT_ROOT = os.path.join("experiments", EXPERIMENT_NAME)
os.makedirs(EXPERIMENT_ROOT, exist_ok=True)


def exp_path(*args):
    """Prefix paths with experiment root."""
    return os.path.join(EXPERIMENT_ROOT, *args)

# ============================================================
# Build Scene Definitions
# ============================================================
operator_codes = {"Sphere": 0, "Cylinder": 1, "Torus": 2}
radii = [0.1, 0.3, 0.5, 0.7, 0.9]
torus_ratio = 0.5

scenes: Scenes = {}
scene_id = 0

for shape, op_code in operator_codes.items():
    for R in radii:
        # Define SDF function
        if shape == "Sphere":
            def sdf_fn(q, R=R):
                return sdf_primitives.SphereSDF(center=[0, 0, 0], radius=R)._compute(q[:, :3])
            param_dict = {"R": float(R), "operator": float(op_code)}

        elif shape == "Cylinder":
            def sdf_fn(q, R=R):
                return sdf_primitives.CylinderSDF(point=[0, 0, 0], axis="y", radius=R)._compute(q[:, :3])
            param_dict = {"R": float(R), "operator": float(op_code)}

        elif shape == "Torus":
            r_small = R * torus_ratio
            def sdf_fn(q, R=R, r_small=r_small):
                return sdf_primitives.TorusSDF(center=[0, 0, 0], R=R, r=r_small)._compute(q[:, :3])
            param_dict = {"R": float(R), "r_small": float(r_small), "operator": float(op_code)}

        # Store in scenes
        scenes[f"scene_{scene_id}"] = {
            0: (
                sdf_fn,
                [(v, v) for v in param_dict.values()]
            )
        }
        scene_id += 1


# ============================================================
# Train the Model
# ============================================================
model_name = "Latent2D_AllShapes"
latent_dim = 2

print(f"[INFO] Starting training for {model_name} with {len(scenes)} total scenes...")

model = Model(
    base_directory=exp_path("trained_models"),
    model_name=model_name,
    scenes=scenes,
    latentDim=latent_dim,
    resume=True
)
model.trainModel()

print(f"[INFO] Training complete for model '{model_name}'.")


# ============================================================
# Extract and Save Latent Vectors
# ============================================================
latent_vecs = []
scene_keys = []
operator_vals = []
radii_vals = []

for scene_name, sdf_entry in scenes.items():
    scene_obj = model.get_scene(scene_name)
    latent_vec = scene_obj.get_latent_vector().detach().cpu().numpy()
    latent_vecs.append(latent_vec)

    # Pull parameters
    param_values = list(sdf_entry.values())[0][1][0]
    radii_vals.append(param_values[0])
    operator_vals.append(param_values[-1])
    scene_keys.append(scene_name)

latent_vecs = np.stack(latent_vecs, axis=0)

latent_json_path = exp_path("latent_vectors.json")
with open(latent_json_path, "w") as f:
    json.dump({
        "latent_vectors": latent_vecs.tolist(),
        "scene_keys": scene_keys,
        "operators": operator_vals,
        "radii": radii_vals
    }, f, indent=2)

print(f"[INFO] Latent vectors saved to: {latent_json_path}")


# ============================================================
# Interpolation and Mesh Generation
# ============================================================
interpolated_dir = exp_path("trained_models", model_name, "InterpolatedShapes")
os.makedirs(interpolated_dir, exist_ok=True)

x_vals = np.linspace(latent_vecs[:, 0].min(), latent_vecs[:, 0].max(), 10)
y_vals = np.linspace(latent_vecs[:, 1].min(), latent_vecs[:, 1].max(), 10)

manifest = {}

print(f"[INFO] Generating interpolation grid ({len(x_vals)}x{len(y_vals)})...")

for xi in x_vals:
    for yi in y_vals:
        latent_point = torch.tensor([[xi, yi]], dtype=torch.float32)
        suffix = f"x{xi:+.3f}_y{yi:+.3f}"

        visualize_a_shape(
            model_name=model_name,
            latent=latent_point,
            experiment_root=EXPERIMENT_ROOT,
            save_suffix=suffix
        )

        manifest[f"{suffix}.ply"] = [float(xi), float(yi)]

manifest_path = os.path.join(interpolated_dir, "latent_grid_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"[INFO] Interpolation complete.")
print(f"[INFO] Manifest saved at: {manifest_path}")
print(f"[INFO] Outputs in: {interpolated_dir}")
