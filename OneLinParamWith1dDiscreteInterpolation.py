import os
import numpy as np
import Trainer
import VisualizeAShape
import DeepSDFStruct.sdf_primitives as sdf_primitives

EXPERIMENT_NAME = "OneLinParamW1dDiscreteInterpolation"
EXPERIMENT_ROOT = os.path.join("experiments", EXPERIMENT_NAME)
os.makedirs(EXPERIMENT_ROOT, exist_ok=True)

# Local function to prefix paths
def exp_path(*args):
    """Prefix paths with the experiment root."""
    return os.path.join(EXPERIMENT_ROOT, *args)



# ============================================================
# Continuous Operator Experiment
# ============================================================
# Trains a single model "AllShapes" that learns 4 primitives:
# 0 = Sphere, 1 = CornerSphere, 2 = Cylinder, 3 = Torus
# Then visualizes interpolations across operator ∈ [0, 3].
# ============================================================
if __name__ == "__main__":
    # ---------------------------
    # Initialize trainer
    # ---------------------------

    # All training artifacts are under experiment root
    trainer = Trainer.DeepSDFTrainer(base_dir=exp_path("trained_models"))
    model_name = "AllShapes"

    # ---------------------------
    # Define training shapes
    # ---------------------------
    radii = [0.1, 0.3, 0.5, 0.7, 0.9]
    torus_ratio = 0.5  # small radius = R * ratio
    operator_codes = {"Sphere": 0, "CornerSphere": 1, "Cylinder": 2, "Torus": 3}

    model_scenes = []
    sdf_parameters = []

    # Sphere
    for r in radii:
        model_scenes.append(sdf_primitives.SphereSDF(center=[0, 0, 0], radius=r)._compute)
        sdf_parameters.append({"R": r, "operator": operator_codes["Sphere"]})

    # CornerSphere
    for r in radii:
        model_scenes.append(sdf_primitives.CornerSpheresSDF(radius=r)._compute)
        sdf_parameters.append({"R": r, "operator": operator_codes["CornerSphere"]})

    # Cylinder
    for r in radii:
        model_scenes.append(sdf_primitives.CylinderSDF(point=[0, 0, 0], axis="y", radius=r)._compute)
        sdf_parameters.append({"R": r, "operator": operator_codes["Cylinder"]})

    # Torus
    for R in radii:
        r_small = R * torus_ratio
        model_scenes.append(sdf_primitives.TorusSDF(center=[0, 0, 0], R=R, r=r_small)._compute)
        sdf_parameters.append({"R": R, "operator": operator_codes["Torus"]})

    # ---------------------------
    # Train the unified model
    # ---------------------------
    print(f"\n[INFO] Training model: {model_name} with {len(model_scenes)} total shapes...")
    trainer.train_models({model_name: model_scenes}, resume=True, latentDim=0)
    print(f"[INFO] Training complete for model '{model_name}'.")

    # ---------------------------
    # Visualize interpolation over operator ∈ [0, 3]
    # ---------------------------
    print("\n[INFO] Visualizing operator interpolation (0 → 3)...")

    # Sample continuous operator values
    operator_values = np.linspace(0.0, 3.0, num=10)
    param_values = [[float(op)] for op in operator_values]

    # ---------------------------
    # Call visualize_a_shape once — it will loop internally
    # ---------------------------
    meshes = VisualizeAShape.visualize_a_shape(
        model_name="AllShapes",
        param_values=[[op] for op in operator_values],
        save_suffix="operator_interp",
        # Provide experiment root only to localize path resolution
        experiment_root=EXPERIMENT_ROOT
    )

    print(f"[INFO] Generated {len(meshes)} interpolated operator meshes (0→3).")


    # ---------------------------
    # Done
    # ---------------------------
    print("\n[INFO] Experiment complete.")
    print(f"[INFO] Output saved under: trained_models/{model_name}/Meshes")
