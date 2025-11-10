import os
import sys
import os
import sys
import json
import numpy as np
import torch
import VisualizeAShape
import DeepSDFStruct.sdf_primitives as sdf_primitives
from IterativeModel import Model, Scenes  # updated classes

# ============================================================
# Experiment Configuration
# ============================================================
EXPERIMENT_NAME = "OneLinParamW1dDiscreteInterpolation"
EXPERIMENT_ROOT = os.path.join("experiments", EXPERIMENT_NAME)
os.makedirs(EXPERIMENT_ROOT, exist_ok=True)

def exp_path(*args):
    """Prefix paths with the experiment root."""
    return os.path.join(EXPERIMENT_ROOT, *args)

# ============================================================
# Continuous Operator Experiment
# ============================================================
if __name__ == "__main__":
    model_name = "AllShapes"

    # Radii and torus ratio
    radii = [0.1, 0.3, 0.5, 0.7, 0.9]
    torus_ratio = 0.5
    operator_codes = {"Sphere": 0, "CornerSphere": 1, "Cylinder": 2, "Torus": 3}

    # ---------------------------
    # Build unified model with multiple primitives
    # ---------------------------
    sdfs = []
    params = []

    # Spheres
    for r in radii:
        def sphere_fn(q, radius=r):
            sdf = sdf_primitives.SphereSDF(center=[0, 0, 0], radius=radius)
            return sdf._compute(q[:, :3])

        sdfs.append(sphere_fn)
        params.append({"R": float(r), "operator": float(operator_codes["Sphere"])})

    # CornerSpheres
    for r in radii:
        def corner_fn(q, radius=r):
            sdf = sdf_primitives.CornerSpheresSDF(radius=radius)
            return sdf._compute(q[:, :3])

        sdfs.append(corner_fn)
        params.append({"R": float(r), "operator": float(operator_codes["CornerSphere"])})

    # Cylinders
    for r in radii:
        def cylinder_fn(q, radius=r):
            sdf = sdf_primitives.CylinderSDF(point=[0, 0, 0], axis="y", radius=radius)
            return sdf._compute(q[:, :3])

        sdfs.append(cylinder_fn)
        params.append({"R": float(r), "operator": float(operator_codes["Cylinder"])})

    # Tori
    for R in radii:
        r_small = R * torus_ratio

        def torus_fn(q, R=R, r_small=r_small):
            sdf = sdf_primitives.TorusSDF(center=[0, 0, 0], R=R, r=r_small)
            return sdf._compute(q[:, :3])

        sdfs.append(torus_fn)
        params.append({"R": float(R), "r_small": float(r_small), "operator": float(operator_codes["Torus"])})

    # ---------------------------
    # Build Scenes dict
    # ---------------------------
    scenes: Scenes = {
        model_name: {
            i: (
                sdf_fn,
                [(v, v) for v in param_dict.values()]  # (min,max) tuple list
            )
            for i, (sdf_fn, param_dict) in enumerate(zip(sdfs, params))
        }
    }

    # ---------------------------
    # Train unified model using Model class
    # ---------------------------
    print(f"[INFO] Training model: {model_name} with {len(scenes[model_name])} total scenes...")
    model = Model(
        base_directory=exp_path("trained_models"),
        model_name=model_name,
        scenes=scenes,
        resume=True,
        latentDim=0  # no latent code for operator experiments
    )

    model.trainModel()
    print(f"[INFO] Training complete for model '{model_name}'.")

    # ---------------------------
    # Visualize interpolation over operator ∈ [0, 4] and varying radius
    # ---------------------------
    print("\n[INFO] Visualizing operator and radius interpolation...")
    operator_values = np.arange(0.0, 4.1, 0.1)
    radius_values = np.linspace(0.1, 0.9, 5)

    interpolated_dir = exp_path("trained_models", model_name, "InterpolatedShapes")
    os.makedirs(interpolated_dir, exist_ok=True)
    manifest = {}

    for r in radius_values:
        for op in operator_values:
            param_values = torch.tensor([[r, float(op)]], dtype=torch.float32)
            suffix = f"r{r:.2f}_op{op:.2f}"

            meshes = VisualizeAShape.visualize_a_shape(
                model_name=model_name,
                param_values=[param_values],
                save_suffix=suffix,
                experiment_root=EXPERIMENT_ROOT
            )

            # Save metadata
            for mesh in meshes:
                manifest[mesh.metadata.get("filename", suffix + ".ply")] = {
                    "R": r,
                    "operator": op
                }

    manifest_path = os.path.join(interpolated_dir, "interpolation_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] Interpolated meshes saved in: {interpolated_dir}")
    print(f"[INFO] Manifest saved as: {manifest_path}")
    print("\n[INFO] Experiment complete.")
    print(f"[INFO] Output saved under: trained_models/{model_name}/Meshes")
