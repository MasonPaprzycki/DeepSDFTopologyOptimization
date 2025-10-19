import os
import sys
import torch
import Trainer
import VisualizeAShape
import DeepSDFStruct.sdf_primitives as sdf_primitives
import numpy as np
import json

# ---------------------------
# Experiment folder
# ---------------------------
EXPERIMENT_NAME = "OneLinParamW1dDiscreteInterpolation"
EXPERIMENT_ROOT = os.path.abspath(os.path.join("experiments", EXPERIMENT_NAME))
print(f"[DEBUG] EXPERIMENT_ROOT = {EXPERIMENT_ROOT}")

trainer = Trainer.DeepSDFTrainer(base_dir=os.path.join(EXPERIMENT_ROOT, "trained_models"))
print(f"[DEBUG] trainer.base_dir = {trainer.base_dir}")


# ---------------------------
# Tee-like logger to capture stdout
# ---------------------------
class TeeLogger:
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

# ---------------------------
# Main experiment
# ---------------------------
if __name__ == "__main__":
    trainer = Trainer.DeepSDFTrainer(base_dir=os.path.join(EXPERIMENT_ROOT, "trained_models"))

    # ---------------------------
    # Define SDF functions and parameters per model
    # ---------------------------
    iterations = 19
    radii = np.linspace(0, 1, iterations + 2)[1:-1].tolist()
    torus_ratio = 0.5

    model_scenes = {}
    sdf_parameters = {}

    # Sphere: only radius varies
    # Sphere
    model_scenes["Sphere"] = [
        (lambda r: lambda q: sdf_primitives.SphereSDF(center=[0,0,0], radius=r)._compute(q[:, :3]))(r)
        for r in radii
    ]

    # Cylinder
    model_scenes["Cylinder"] = [
        (lambda r: lambda q: sdf_primitives.CylinderSDF(point=[0,0,0], axis="y", radius=r)._compute(q[:, :3]))(r)
        for r in radii
    ]

    # Torus
    model_scenes["Torus"] = [
        (lambda R: lambda q: sdf_primitives.TorusSDF(center=[0,0,0], R=R, r=R*torus_ratio)._compute(q[:, :3]))(R)
        for R in radii
    ]

    # CornerSphere
    model_scenes["CornerSphere"] = [
        (lambda r: lambda q: sdf_primitives.CornerSpheresSDF(radius=r)._compute(q[:, :3]))(r)
        for r in radii
    ]

    # ---------------------------
    # Train all models
    # ---------------------------
    trainer.train_models(
        model_scenes,
        resume=True,
        latentDim=1
    )

    # ---------------------------
    # Visualize trained scenes
    # ---------------------------
    for model_name, scenes in model_scenes.items():
        print(f"\nVisualizing model: {model_name}")
        for scene_id in range(len(scenes)):
            VisualizeAShape.visualize_a_shape(
                model_name,
                scene_id=scene_id,
                experiment_root=EXPERIMENT_ROOT
            )

    # ---------------------------
    # Print latent vectors with associated parameters
    # ---------------------------
    print("\nLatent vectors with parameters for all models and scenes:")
    for model_name, scenes in model_scenes.items():
        latent_dir = os.path.join(EXPERIMENT_ROOT, "trained_models", model_name, "LatentCodes")
        latest_file = os.path.join(latent_dir, "latest.pth")
        if not os.path.exists(latest_file):
            print(f"No latest.pth found for {model_name}")
            continue

        latest_data = torch.load(latest_file, map_location="cpu")
        all_latents = latest_data.get("latent_codes", {})

        for scene_key in sorted(all_latents.keys()):
            latent_code = all_latents[scene_key]
            idx = int(scene_key.split("_")[-1])
            params = sdf_parameters[model_name][idx]
            print(f"Model: {model_name}, Scene: {scene_key}, Parameters: {params}, "
                  f"Latent vector: {latent_code.numpy().flatten()}")

    # ---------------------------
    # Interpolate latent vectors (1D latent from -1 to 1)
    # ---------------------------
    print("\nGenerating interpolated latent shapes (-1 → 1)...")
    latent_values = np.linspace(-1, 1, 10)

    for model_name in model_scenes.keys():
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

        # Save JSON manifest
        with open(os.path.join(interp_dir, "latent_1d_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"[INFO] Interpolated 1D latent meshes saved in: {interp_dir}")
