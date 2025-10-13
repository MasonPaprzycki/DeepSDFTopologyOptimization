import os
import torch
import Trainer
import DeepSDFStruct.sdf_primitives as sdf_primitives

# ---------------------------
# Experiment to confirm correlations between latent vectors and linear shape parameters
# ---------------------------
if __name__ == "__main__":
    trainer = Trainer.DeepSDFTrainer()

    # ---------------------------
    # Define SDF functions and parameters per model
    # ---------------------------
    radii = [0.1, 0.3, 0.5, 0.7, 0.9]  # small number for testing
    torus_radii = [0.05, 0.15, 0.25, 0.35, 0.45]

    model_scenes = {}
    scene_params = {}  # store parameters for each scene

    # Sphere
    model_scenes["Sphere"] = [sdf_primitives.SphereSDF(center=[0,0,0], radius=r)._compute for r in radii]
    scene_params["Sphere"] = [{"center": [0,0,0], "radius": r} for r in radii]

    # CornerSphere
    model_scenes["CornerSphere"] = [sdf_primitives.CornerSpheresSDF(radius=r)._compute for r in radii]
    scene_params["CornerSphere"] = [{"radius": r} for r in radii]

    # Cylinder
    model_scenes["Cylinder"] = [sdf_primitives.CylinderSDF(point=[0,0,0], axis="y", radius=r)._compute for r in radii]
    scene_params["Cylinder"] = [{"point": [0,0,0], "axis": "y", "radius": r} for r in radii]

    # Torus
    model_scenes["Torus"] = [sdf_primitives.TorusSDF(center=[0,0,0], R=R, r=r)._compute for R in radii for r in torus_radii]
    scene_params["Torus"] = [{"center": [0,0,0], "R": R, "r": r} for R in radii for r in torus_radii]

    # ---------------------------
    # Train all models/scenes
    # ---------------------------
    trainer.train_models(model_scenes, resume=True)

    # ---------------------------
    # Visualize models
    # ---------------------------
    for model_name in ["Sphere", "Cylinder", "Torus"]:
        print(f"\nVisualizing model: {model_name}")
        trainer.visualize_model(model_name, all_scenes=True)

    # ---------------------------
    # Print latent vectors with associated parameters
    # ---------------------------
    print("\nLatent vectors with parameters for all models and scenes:")
    for model_name in model_scenes.keys():
        latent_dir = os.path.join(trainer.base_dir, model_name, "LatentCodes")
        if not os.path.exists(latent_dir):
            print(f"No latent folder found for model {model_name}")
            continue

        for idx, fname in enumerate(sorted(os.listdir(latent_dir))):
            if fname.endswith(".pth"):
                latent_path = os.path.join(latent_dir, fname)
                latent_dict = torch.load(latent_path, map_location="cpu")
                scene_key = list(latent_dict["latent_codes"].keys())[0]
                latent_code = latent_dict["latent_codes"][scene_key]

                # Get associated parameters
                params = scene_params[model_name][idx]
                print(f"Model: {model_name}, Scene: {scene_key}, Parameters: {params}, Latent vector: {latent_code.numpy().flatten()}")
