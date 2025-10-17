import os
import torch
import Trainer
import VisualizeAShape
import DeepSDFStruct.sdf_primitives as sdf_primitives

# ---------------------------
# Experiment folder
# ---------------------------
EXPERIMENT_NAME = "OneLinParamW1dDiscreteInterpolation"
EXPERIMENT_ROOT = os.path.abspath(EXPERIMENT_NAME)
os.makedirs(EXPERIMENT_ROOT, exist_ok=True)

# ---------------------------
# Experiment description
# ---------------------------
if __name__ == "__main__":
    trainer = Trainer.DeepSDFTrainer(base_dir=os.path.join(EXPERIMENT_ROOT, "trained_models"))

    # ---------------------------
    # Define SDF functions and parameters per model
    # ---------------------------
    radii = [0.1, 0.3, 0.5, 0.7, 0.9]
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
    # Train all models/scenes
    # ---------------------------
    trainer.train_models(model_scenes, resume=True)

    # ---------------------------
    # Visualize models using experiment root
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
            # Extract scene index from key
            idx = int(scene_key.split("_")[-1])
            params = sdf_parameters[model_name][idx]
            print(f"Model: {model_name}, Scene: {scene_key}, Parameters: {params}, "
                  f"Latent vector: {latent_code.numpy().flatten()}")
