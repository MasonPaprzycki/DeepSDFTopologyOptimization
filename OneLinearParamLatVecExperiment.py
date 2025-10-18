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
EXPERIMENT_ROOT = os.path.abspath(EXPERIMENT_NAME)
os.makedirs(EXPERIMENT_ROOT, exist_ok=True)

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

log_path = os.path.join(EXPERIMENT_ROOT, "results.txt")
sys.stdout = TeeLogger(log_path)

# ---------------------------
# Experiment description
# visualize how well a one dimensional latent space can represent a linear parameter of a model.
# ---------------------------
if __name__ == "__main__":
    trainer = Trainer.DeepSDFTrainer(base_dir=os.path.join(EXPERIMENT_ROOT, "trained_models"))

    # ---------------------------
    # Define SDF functions and parameters per model
    # ---------------------------
    iterations = 19
    radii = np.linspace(0, 1, iterations + 2)[1:-1].tolist()
    torus_radii_ratio = 0.5  # r = R * ratio

    model_scenes = {}
    sdf_parameters = {}

    # Sphere
    model_scenes["Sphere"] = [sdf_primitives.SphereSDF(center=[0,0,0], radius=r)._compute for r in radii]
    sdf_parameters["Sphere"] = [{"center": [0,0,0], "radius": r} for r in radii]

    # CornerSphere
    model_scenes["CornerSphere"] = [sdf_primitives.CornerSpheresSDF(radius=r)._compute for r in radii]
    sdf_parameters["CornerSphere"] = [{"radius": r} for r in radii]

    # Cylinder
    model_scenes["Cylinder"] = [sdf_primitives.CylinderSDF(point=[0,0,0], axis="y", radius=r)._compute for r in radii]
    sdf_parameters["Cylinder"] = [{"point": [0,0,0], "axis": "y", "radius": r} for r in radii]

    # Torus
    model_scenes["Torus"] = [sdf_primitives.TorusSDF(center=[0,0,0], R=R, r=R*torus_radii_ratio)._compute for R in radii]
    sdf_parameters["Torus"] = [{"center": [0,0,0], "R": R, "r": R*torus_radii_ratio} for R in radii]

    # ---------------------------
    # Train all models 
    # ---------------------------
    trainer.train_models(model_scenes, resume=True)

    # ---------------------------
    # Visualize trained scenes
    # ---------------------------
    for model_name in model_scenes.keys():
        print(f"\nVisualizing model: {model_name}")
        scenes_count = len(model_scenes[model_name])
        for scene_id in range(scenes_count):
            VisualizeAShape.visualize_a_shape(
                model_name,
                scene_id=scene_id,
                experiment_root=EXPERIMENT_ROOT
            )

    # ---------------------------
    # Print latent vectors with associated parameters
    # ---------------------------
    print("\nLatent vectors with parameters for all models and scenes:")
    for model_name in model_scenes.keys():
        latent_dir = os.path.join(EXPERIMENT_ROOT, "trained_models", model_name, "LatentCodes")
        latest_latent_file = os.path.join(latent_dir, "latest.pth")

        if not os.path.exists(latest_latent_file):
            print(f"No latest.pth found for model {model_name}")
            continue

        latest_data = torch.load(latest_latent_file, map_location="cpu")
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
    interpolation_steps = 10
    latent_values = np.linspace(-1, 1, interpolation_steps)

    for model_name in model_scenes.keys():
        interpolated_dir = os.path.join(EXPERIMENT_ROOT, "trained_models", model_name, "InterpolatedShapes")
        os.makedirs(interpolated_dir, exist_ok=True)
        manifest = {}

        for val in latent_values:
            latent_point = torch.tensor([[val]], dtype=torch.float32)
            # filename based on latent value
            filename = f"{val:+.3f}.ply"
            mesh_path = os.path.join(interpolated_dir, filename)

            # generate mesh
            VisualizeAShape.visualize_a_shape(
                model_name=model_name,
                latent=latent_point,
                save_suffix=None,
                experiment_root=EXPERIMENT_ROOT
            )

            manifest[filename] = float(val)

        # Save JSON manifest
        manifest_path = os.path.join(interpolated_dir, "latent_1d_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"[INFO] Interpolated 1D latent meshes saved in: {interpolated_dir}")
        print(f"[INFO] Manifest saved as: {manifest_path}")
