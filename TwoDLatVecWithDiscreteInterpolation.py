import os
import torch
import numpy as np
import json
from VisualizeAShape import visualize_a_shape
import DeepSDFStruct.deep_sdf.data as deep_data
import DeepSDFStruct.deep_sdf.training as training

# -------------------------
# Experiment setup
# -------------------------
EXPERIMENT_NAME = "TwoDLatentVecWithDiscreteInterpolationAndLinScaling"
EXPERIMENT_ROOT = os.path.join("experiments", EXPERIMENT_NAME)
os.makedirs(EXPERIMENT_ROOT, exist_ok=True)

def exp_path(*args):
    return os.path.join(EXPERIMENT_ROOT, *args)

# -------------------------
# Operator definitions
# -------------------------
# In this environment, you need to define your SDFs as callable functions.
# Example SDFs (replace with actual SDF formulas as needed):
def sphere_sdf(xyz, params):
    R = float(params[0])
    return torch.norm(xyz, dim=1) - R

def cylinder_sdf(xyz, params):
    R = float(params[0])
    # Cylinder along y-axis
    return torch.sqrt(xyz[:, 0]**2 + xyz[:, 2]**2) - R

def torus_sdf(xyz, params):
    R = float(params[0])
    r_small = float(params[1])
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    q = torch.stack([torch.sqrt(x**2 + z**2) - R, y], dim=1)
    return torch.norm(q, dim=1) - r_small

operator_codes = {"Sphere": 0, "Cylinder": 1, "Torus": 2}
radii = [0.1, 0.3, 0.5, 0.7, 0.9]
torus_radii_ratio = 0.5

# -------------------------
# Build scenes
# -------------------------
scenes = {}
scene_id = 0
for shape, op_code in operator_codes.items():
    for R in radii:
        if shape == "Sphere":
            sdf_fn = lambda xyz, p, R=R: sphere_sdf(xyz, [R])
            params = [(float(R), float(op_code))]
        elif shape == "Cylinder":
            sdf_fn = lambda xyz, p, R=R: cylinder_sdf(xyz, [R])
            params = [(float(R), float(op_code))]
        elif shape == "Torus":
            r_small = R * torus_radii_ratio
            sdf_fn = lambda xyz, p, R=R, r_small=r_small: torus_sdf(xyz, [R, r_small])
            params = [(float(R), float(r_small), float(op_code))]
        scenes[str(scene_id)] = {0: (sdf_fn, params)}
        scene_id += 1

# -------------------------
# Train model
# -------------------------
from Model import Model

model_name = "Latent2D_AllShapes"
latentDim = 2

model = Model(base_directory=exp_path("trained_models"), model_name=model_name, scenes=scenes, latentDim=latentDim)
model.trainModel()  # uses DeepSDFStruct training pipeline

# -------------------------
# Extract latent vectors
# -------------------------
latent_vecs = []
scene_keys = []
operator_vals = []
radii_vals = []

for scene_id, sdf_params in scenes.items():
    scene_key = f"{model_name.lower()}_{int(scene_id):03d}"
    scene = model.get_scene(scene_key)
    latent_vec = scene.get_latent_vector().numpy()
    latent_vecs.append(latent_vec)

    param_values = list(sdf_params.values())[0][1][0]  # params tuple
    radii_vals.append(param_values[0])
    operator_vals.append(param_values[-1])
    scene_keys.append(scene_key)

latent_vecs = np.stack(latent_vecs, axis=0)

# -------------------------
# Save latent vectors for plotting
# -------------------------
latent_json_path = os.path.join(EXPERIMENT_ROOT, "latent_vectors.json")
with open(latent_json_path, "w") as f:
    json.dump({
        "latent_vectors": latent_vecs.tolist(),
        "scene_keys": scene_keys,
        "operators": operator_vals,
        "radii": radii_vals
    }, f, indent=2)

print(f"[INFO] Saved latent vectors to {latent_json_path}")

# -------------------------
# Interpolate latent grid and generate meshes
# -------------------------
interpolated_dir = os.path.join(EXPERIMENT_ROOT, "InterpolatedShapes")
os.makedirs(interpolated_dir, exist_ok=True)

x_vals = np.linspace(latent_vecs[:,0].min(), latent_vecs[:,0].max(), 10)
y_vals = np.linspace(latent_vecs[:,1].min(), latent_vecs[:,1].max(), 10)

manifest = {}
for xi in x_vals:
    for yi in y_vals:
        latent_point = torch.tensor([[xi, yi]], dtype=torch.float32)
        filename = f"{'+%.3f' % xi}_{'+%.3f' % yi}.ply"
        visualize_a_shape(
            model_name=model_name,
            latent=latent_point,
            experiment_root=EXPERIMENT_ROOT
        )
        manifest[filename] = [float(xi), float(yi)]

manifest_path = os.path.join(interpolated_dir, "latent_grid_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"[INFO] Interpolation complete. Manifest saved at {manifest_path}")
