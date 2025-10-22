import os
import torch
import numpy as np
import Trainer
import DeepSDFStruct.sdf_primitives as sdf_primitives
import matplotlib.pyplot as plt
import json
import VisualizeAShape

# ====================================================
# Experiment setup
# ====================================================
EXPERIMENT_NAME = "TwoDLatentVecWithDiscreteInterpolationAndLinScaling"
EXPERIMENT_ROOT = os.path.join("experiments", EXPERIMENT_NAME)
os.makedirs(EXPERIMENT_ROOT, exist_ok=True)

def exp_path(*args):
    return os.path.join(EXPERIMENT_ROOT, *args)

# ====================================================
# Main
# ====================================================
if __name__ == "__main__":
    trainer = Trainer.Trainer(base_dir=exp_path("trained_models"))

    radii = [0.1, 0.3, 0.5, 0.7, 0.9]
    torus_radii_ratio = 0.5
    operator_codes = {"Sphere": 0, "CornerSphere": 1, "Cylinder": 2, "Torus": 3}
    operator_colors = ["red", "green", "blue", "purple"]

    model_name = "Latent2D_AllShapes"

    # ---------------------------
    # Build scene list as SDFCallable + param_ranges
    # ---------------------------
    sdfs = []
    params_list = []

    for shape, op_code in operator_codes.items():
        for R in radii:
            if shape == "Sphere":
                sdfs.append(lambda q, R=R: sdf_primitives.SphereSDF(center=[0,0,0], radius=R)._compute(q[:, :3]))
                params_list.append({"R": float(R), "operator": float(op_code)})
            elif shape == "CornerSphere":
                sdfs.append(lambda q, R=R: sdf_primitives.CornerSpheresSDF(radius=R)._compute(q[:, :3]))
                params_list.append({"R": float(R), "operator": float(op_code)})
            elif shape == "Cylinder":
                sdfs.append(lambda q, R=R: sdf_primitives.CylinderSDF(point=[0,0,0], axis="y", radius=R)._compute(q[:, :3]))
                params_list.append({"R": float(R), "operator": float(op_code)})
            elif shape == "Torus":
                r_small = R * torus_radii_ratio
                sdfs.append(lambda q, R=R, r_small=r_small: sdf_primitives.TorusSDF(center=[0,0,0], R=R, r=r_small)._compute(q[:, :3]))
                params_list.append({"R": float(R), "r_small": float(r_small), "operator": float(op_code)})

    # Helper to convert list of SDFs + params -> scene dict
    def build_scene_dict(sdfs, params):
        return {i: (sdf, list(p.values())) for i, (sdf, p) in enumerate(zip(sdfs, params))}

    scenes = {"scenes": build_scene_dict(sdfs, params_list)}
    models: Trainer.Models = {model_name: scenes}

    # ---------------------------
    # Train unified model (2D latent)
    # ---------------------------
    print(f"[INFO] Training model: {model_name} with {len(sdfs)} total scenes...")
    trainer.train_models(
        models=models,
        latentDim=2,
        resume=True
    )
    print(f"[INFO] Training complete for model '{model_name}'.")

    # ---------------------------
    # Load latent codes
    # ---------------------------
    latent_file = os.path.join(trainer.base_dir, model_name, "LatentCodes", "latest.pth")
    if not os.path.exists(latent_file):
        raise FileNotFoundError(f"No latent file found for {model_name}")

    latents = torch.load(latent_file, map_location="cpu").get("latent_codes", {})
    latent_vecs = []
    shapes = []
    radii_vals = []

    for (key, vec), param in zip(latents.items(), params_list):
        latent_vecs.append(vec.squeeze().numpy())
        shapes.append(param[-1])  # last value = operator code
        radii_vals.append(param[0])
        print(f"[INFO] Scene: {key}, Operator: {param[-1]}, R: {param[0]}, Latent: {vec.squeeze().numpy()}")

    latent_vecs = torch.tensor(latent_vecs)
    if latent_vecs.shape[1] != 2:
        print(f"[WARN] Latent dimension = {latent_vecs.shape[1]}. Expected 2 for visualization.")
        latent_vecs = latent_vecs[:, :2]

    # ---------------------------
    # Visualize latent space
    # ---------------------------
    plt.figure(figsize=(8,6))
    for op_code, color in zip(operator_codes.values(), operator_colors):
        mask = torch.tensor(shapes) == op_code
        plt.scatter(
            latent_vecs[mask,0],
            latent_vecs[mask,1],
            c=color,
            s=[r*300 for r in torch.tensor(radii_vals)[mask]],
            label=list(operator_codes.keys())[op_code],
            alpha=0.7,
            edgecolor="k"
        )
    plt.title(f"2D Latent Space — {model_name}")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.legend(title="Shape Type")
    plt.grid(True)
    plt.tight_layout()

    output_dir = os.path.join(trainer.base_dir, model_name, "Visualizations")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "latent_space.png")
    plt.savefig(plot_path, dpi=300)
    print(f"[INFO] Saved latent space visualization to: {plot_path}")

    try:
        plt.show()
    except Exception:
        print("[WARN] Could not open interactive window (headless environment).")

    # ---------------------------
    # Interpolate latent vectors (10x10 grid)
    # ---------------------------
    interpolated_dir = os.path.join(trainer.base_dir, model_name, "InterpolatedShapes")
    os.makedirs(interpolated_dir, exist_ok=True)
    manifest = {}

    x_vals = np.linspace(latent_vecs[:,0].min(), latent_vecs[:,0].max(), 10)
    y_vals = np.linspace(latent_vecs[:,1].min(), latent_vecs[:,1].max(), 10)

    print(f"[INFO] Generating 10x10 latent interpolation grid...")

    for xi in x_vals:
        for yi in y_vals:
            latent_point = torch.tensor([[xi, yi]], dtype=torch.float32)
            filename = f"{'+%.3f' % xi}_{'+%.3f' % yi}.ply"
            VisualizeAShape.visualize_a_shape(
                model_name=model_name,
                latent=latent_point,
                save_suffix=None,
                experiment_root=EXPERIMENT_ROOT,
            )
            manifest[filename] = [float(xi), float(yi)]

    # Save JSON manifest
    manifest_path = os.path.join(interpolated_dir, "latent_grid_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] Interpolated meshes saved under: {interpolated_dir}")
    print(f"[INFO] Manifest saved as: {manifest_path}")

