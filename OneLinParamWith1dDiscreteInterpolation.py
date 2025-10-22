import os
import numpy as np
import Trainer
import VisualizeAShape
import DeepSDFStruct.sdf_primitives as sdf_primitives
import sys
import json

# ============================================================
# Experiment Configuration
# ============================================================
EXPERIMENT_NAME = "OneLinParamW1dDiscreteInterpolation"
EXPERIMENT_ROOT = os.path.join("experiments", EXPERIMENT_NAME)
os.makedirs(EXPERIMENT_ROOT, exist_ok=True)


def exp_path(*args):
    """Prefix paths with the experiment root."""
    return os.path.join(EXPERIMENT_ROOT, *args)

# ---------------------------
# Tee-like logger to capture stdout to a txt file
# ---------------------------
class TeeLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = TeeLogger(os.path.join(EXPERIMENT_ROOT, "results.txt"))

# ============================================================
# Continuous Operator Experiment
# ============================================================
if __name__ == "__main__":
    trainer = Trainer.Trainer(base_dir=exp_path("trained_models"))
    model_name = "AllShapes"

    # Radii and torus ratio
    radii = [0.1, 0.3, 0.5, 0.7, 0.9]
    torus_ratio = 0.5
    operator_codes = {"Sphere": 0, "CornerSphere": 1, "Cylinder": 2, "Torus": 3}

    # Helper: convert list of SDFs + params -> scene dict
    def build_scene_dict(sdfs, params):
        return {i: (sdf, list(p.values())) for i, (sdf, p) in enumerate(zip(sdfs, params))}

    # ---------------------------
    # Build unified model with multiple primitives
    # ---------------------------
    sdfs = []
    params = []

    # Sphere
    for r in radii:
        sdfs.append(lambda q, r=r: sdf_primitives.SphereSDF(center=[0,0,0], radius=r)._compute(q[:, :3]))
        params.append({"R": float(r), "operator": float(operator_codes["Sphere"])})

    # CornerSphere
    for r in radii:
        sdfs.append(lambda q, r=r: sdf_primitives.CornerSpheresSDF(radius=r)._compute(q[:, :3]))
        params.append({"R": float(r), "operator": float(operator_codes["CornerSphere"])})

    # Cylinder
    for r in radii:
        sdfs.append(lambda q, r=r: sdf_primitives.CylinderSDF(point=[0,0,0], axis="y", radius=r)._compute(q[:, :3]))
        params.append({"R": float(r), "operator": float(operator_codes["Cylinder"])})

    # Torus
    for R in radii:
        r_small = R * torus_ratio
        sdfs.append(lambda q, R=R, r_small=r_small: sdf_primitives.TorusSDF(center=[0,0,0], R=R, r=r_small)._compute(q[:, :3]))
        params.append({"R": float(R), "r_small": float(r_small), "operator": float(operator_codes["Torus"])})

    # Wrap into Scenes dict
    scenes = {"scenes": build_scene_dict(sdfs, params)}

    # Wrap into Models dict
    models: Trainer.Models = {model_name: scenes}

    # ---------------------------
    # Train the unified model
    # ---------------------------
    print(f"[INFO] Training model: {model_name} with {len(sdfs)} total scenes...")
    trainer.train_models(
        models=models,
        latentDim=0,  # no latent code
        resume=True
    )
    print(f"[INFO] Training complete for model '{model_name}'.")

    # ---------------------------
    # Visualize interpolation over operator ∈ [0, 4] and varying radius
    # ---------------------------
    print("\n[INFO] Visualizing operator and radius interpolation...")
    operator_values = np.arange(0.0, 4.1, 0.1)
    radius_values = np.linspace(0.1, 0.9, 5)

    interpolated_dir = os.path.join(EXPERIMENT_ROOT, "trained_models", model_name, "InterpolatedShapes")
    os.makedirs(interpolated_dir, exist_ok=True)
    manifest = {}

    for r in radius_values:
        for op in operator_values:
            param_values = [r, float(op)]
            suffix = f"r{r:.2f}_op{op:.2f}"
            meshes = VisualizeAShape.visualize_a_shape(
                model_name=model_name,
                param_values=[param_values],
                save_suffix=suffix,
                experiment_root=EXPERIMENT_ROOT
            )
            # Save mesh info in manifest
            for mesh in meshes:
                manifest[mesh.metadata.get("filename", suffix + ".ply")] = {"R": r, "operator": op}

    # Save JSON manifest
    manifest_path = os.path.join(interpolated_dir, "interpolation_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] Interpolated meshes saved in: {interpolated_dir}")
    print(f"[INFO] Manifest saved as: {manifest_path}")
    print("\n[INFO] Experiment complete.")
    print(f"[INFO] Output saved under: trained_models/{model_name}/Meshes")
