import os
import json
import torch
import numpy as np
from DeepSDFStruct.sdf_primitives import SphereSDF, CornerSpheresSDF, CylinderSDF, TorusSDF
import VisualizeAShape
from typing import Dict
from Model import Model, Scene, SDFCallable  # import the class version

# ---------------------------
# Experiment folder
# ---------------------------
EXPERIMENT_NAME = "OneLinParamW1dDiscreteInterpolation"
EXPERIMENT_ROOT = os.path.abspath(os.path.join("experiments", EXPERIMENT_NAME))
os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
print(f"[DEBUG] EXPERIMENT_ROOT = {EXPERIMENT_ROOT}")

# ---------------------------
# Define models and SDFs
# ---------------------------
models: Dict[str, Dict] = {}
iterations = 19
radii = np.linspace(0, 1, iterations + 2)[1:-1].tolist()
torus_ratio = 0.5

def make_sdf_list(sdf_class, param_key, radii, extra_params=None):
    sdf_list = []
    param_list = []
    for r in radii:
        if extra_params:
            sdf_fn = lambda q, _: sdf_class(**extra_params(r))._compute(q[:, :3])
            params = extra_params(r)
        else:
            sdf_fn = lambda q, _: sdf_class(**{param_key: r})._compute(q[:, :3])
            params = {param_key: r}
        sdf_list.append(sdf_fn)
        param_list.append(list(params.values()))
    return sdf_list, param_list

# Sphere
sphere_sdfs, sphere_params = make_sdf_list(SphereSDF, "radius", radii)
models["Sphere"] = {"scenes": {i: (sdf, param) for i, (sdf, param) in enumerate(zip(sphere_sdfs, sphere_params))}}

# Cylinder
cylinder_sdfs, cylinder_params = make_sdf_list(CylinderSDF, "radius", radii, lambda r: {"point":[0,0,0], "axis":"y", "radius":r})
models["Cylinder"] = {"scenes": {i: (sdf, param) for i, (sdf, param) in enumerate(zip(cylinder_sdfs, cylinder_params))}}

# Torus
def torus_params_fn(R):
    return {"center": [0,0,0], "R": R, "r": R*torus_ratio}
torus_sdfs, torus_params = make_sdf_list(TorusSDF, "R", radii, torus_params_fn)
models["Torus"] = {"scenes": {i: (sdf, param) for i, (sdf, param) in enumerate(zip(torus_sdfs, torus_params))}}

# CornerSphere
cornersphere_sdfs, cornersphere_params = make_sdf_list(CornerSpheresSDF, "radius", radii)
models["CornerSphere"] = {"scenes": {i: (sdf, param) for i, (sdf, param) in enumerate(zip(cornersphere_sdfs, cornersphere_params))}}

# ---------------------------
# Train models using the new Model class
# ---------------------------
trained_models: Dict[str, Model] = {}

for model_name, scenes_dict in models.items():
    print(f"[INFO] Training model: {model_name}")
    model = Model(
        base_directory=os.path.join(EXPERIMENT_ROOT, "trained_models"),
        model_name=model_name,
        scenes=scenes_dict["scenes"],
        resume=True,
        latentDim=1
    )
    model.trainModel()
    trained_models[model_name] = model

# ---------------------------
# Visualize trained scenes
# ---------------------------
for model_name, model in trained_models.items():
    print(f"\nVisualizing model: {model_name}")
    for scene_key, scene in model.trained_scenes.items():
        VisualizeAShape.visualize_a_shape(
            model_name=model_name,
            scene_id=int(scene_key.split("_")[-1]),
            experiment_root=EXPERIMENT_ROOT
        )

# ---------------------------
# Interpolate latent vectors (1D)
# ---------------------------
print("\nGenerating interpolated latent shapes (-1 → 1)...")
latent_values = np.linspace(-1, 1, 10)

for model_name, model in trained_models.items():
    interp_dir = os.path.join(EXPERIMENT_ROOT, "trained_models", model_name, "InterpolatedShapes")
    os.makedirs(interp_dir, exist_ok=True)
    manifest = {}

    for val in latent_values:
        latent_point = torch.tensor([[val]], dtype=torch.float32)
        filename = f"{val:+.3f}.ply"

        VisualizeAShape.visualize_a_shape(
            model_name=model_name,
            latent=latent_point,
            save_suffix=None,
            experiment_root=EXPERIMENT_ROOT
        )
        manifest[filename] = float(val)

    with open(os.path.join(interp_dir, "latent_1d_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] Interpolated 1D latent meshes saved in: {interp_dir}")
