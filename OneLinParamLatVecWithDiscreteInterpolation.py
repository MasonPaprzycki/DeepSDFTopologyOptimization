import os
import torch
import numpy as np
import Trainer
import VisualizeAShape
import DeepSDFStruct.sdf_primitives as sdf_primitives

# ---------------------------
# Single-model experiment: continuous parameter R + continuous "operator"
# ---------------------------
if __name__ == "__main__":
    trainer = Trainer.DeepSDFTrainer()

    # ---------------------------
    # Define radii and operator codes
    # ---------------------------
    radii = [0.1, 0.3, 0.5, 0.7, 0.9]
    torus_radii_ratio = 0.5  # small r = R/2
    operator_codes = {"Sphere": 0, "CornerSphere": 1, "Cylinder": 2, "Torus": 3}

    # ---------------------------
    # Collect all scenes into a single model
    # ---------------------------
    model_name = "AllShapes"
    model_scenes = []
    scene_params = []

    # Sphere
    for r in radii:
        model_scenes.append(sdf_primitives.SphereSDF(center=[0, 0, 0], radius=r)._compute)
        scene_params.append({"R": r, "operator": operator_codes["Sphere"]})

    # CornerSphere
    for r in radii:
        model_scenes.append(sdf_primitives.CornerSpheresSDF(radius=r)._compute)
        scene_params.append({"R": r, "operator": operator_codes["CornerSphere"]})

    # Cylinder
    for r in radii:
        model_scenes.append(sdf_primitives.CylinderSDF(point=[0, 0, 0], axis="y", radius=r)._compute)
        scene_params.append({"R": r, "operator": operator_codes["Cylinder"]})

    # Torus
    for R in radii:
        r = R * torus_radii_ratio
        model_scenes.append(sdf_primitives.TorusSDF(center=[0, 0, 0], R=R, r=r)._compute)
        scene_params.append({"R": R, "operator": operator_codes["Torus"]})

    # ---------------------------
    # Train all scenes under one model
    # ---------------------------
    trainer.train_models({model_name: model_scenes}, resume=True)

    # ---------------------------
    # Visualize all learned scenes
    # ---------------------------
    print(f"\nVisualizing model: {model_name}")
    trainer.visualize_model(model_name, all_scenes=True)

    # ---------------------------
    # Print latent vectors with parameters
    # ---------------------------
    print("\nLatent vectors with parameters for all scenes:")
    latent_dir = os.path.join(trainer.base_dir, model_name, "LatentCodes")
    all_latents = []

    if not os.path.exists(latent_dir):
        print(f"No latent folder found for model {model_name}")
    else:
        for idx, fname in enumerate(sorted(os.listdir(latent_dir))):
            if fname.endswith(".pth"):
                latent_path = os.path.join(latent_dir, fname)
                latent_dict = torch.load(latent_path, map_location="cpu")
                scene_key = list(latent_dict["latent_codes"].keys())[0]
                latent_code = latent_dict["latent_codes"][scene_key]
                params = scene_params[idx]
                print(f"Scene: {scene_key}, Parameters: {params}, Latent vector: {latent_code.numpy().flatten()}")
                all_latents.append(latent_code.unsqueeze(0))

    # ---------------------------
    # Interpolation visualization across operator (0 → 3)
    # ---------------------------
    print("\nInterpolating across operator values...")

    operator_values = np.arange(0.0, 3.01, 0.2)
    interpolated_meshes = []

    for op in operator_values:
        print(f"  -> Visualizing operator={op:.2f}")
        # VisualizeAShape should accept a latent OR an external conditioning vector
        # Here we assume scene_id=0 (reuse model), but inject operator value into latent conditioning if supported
        mesh = VisualizeAShape.visualize_a_shape(
            model_name,
            scene_id=0,  # you can modify visualize_a_shape to take custom inputs
            operator_value=op  # <--- You’ll add this arg in your visualize function
        )
        interpolated_meshes.append(mesh)

    print(f"[INFO] Done interpolating {len(interpolated_meshes)} operator values.")
