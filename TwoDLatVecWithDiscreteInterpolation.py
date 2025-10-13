import os
import torch
import Trainer
import DeepSDFStruct.sdf_primitives as sdf_primitives
import matplotlib.pyplot as plt

if __name__ == "__main__":
    trainer = Trainer.DeepSDFTrainer()

    radii = [0.1, 0.3, 0.5, 0.7, 0.9]
    torus_radii_ratio = 0.5
    operator_codes = {"Sphere": 0, "CornerSphere": 1, "Cylinder": 2, "Torus": 3}
    operator_colors = ["red", "green", "blue", "purple"]

    model_name = "Latent2D_AllShapes"
    model_scenes = []
    scene_meta = []  # store shape metadata

    # ---------------------------
    # Build scene list
    # ---------------------------
    for shape, op_code in operator_codes.items():
        for R in radii:
            if shape == "Sphere":
                model_scenes.append(sdf_primitives.SphereSDF(center=[0, 0, 0], radius=R)._compute)
            elif shape == "CornerSphere":
                model_scenes.append(sdf_primitives.CornerSpheresSDF(radius=R)._compute)
            elif shape == "Cylinder":
                model_scenes.append(sdf_primitives.CylinderSDF(point=[0, 0, 0], axis="y", radius=R)._compute)
            elif shape == "Torus":
                r = R * torus_radii_ratio
                model_scenes.append(sdf_primitives.TorusSDF(center=[0, 0, 0], R=R, r=r)._compute)

            scene_meta.append({"shape": shape, "operator": op_code, "R": R})

    # ---------------------------
    # Train unified model
    # ---------------------------
    trainer.train_models({model_name: model_scenes}, resume=True)

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

    for (key, vec), meta in zip(latents.items(), scene_meta):
        latent_vecs.append(vec.squeeze().numpy())
        shapes.append(meta["operator"])
        radii_vals.append(meta["R"])

    latent_vecs = torch.tensor(latent_vecs)
    if latent_vecs.shape[1] != 2:
        print(f"[WARN] Latent dimension = {latent_vecs.shape[1]}. Expected 2 for visualization.")
        print("Plotting first 2 dimensions anyway.")
        latent_vecs = latent_vecs[:, :2]

    # ---------------------------
    # Visualize latent space
    # ---------------------------
    plt.figure(figsize=(8, 6))
    for op_code, color in zip(operator_codes.values(), operator_colors):
        mask = torch.tensor(shapes) == op_code
        plt.scatter(
            latent_vecs[mask, 0],
            latent_vecs[mask, 1],
            c=color,
            s=[r * 300 for r in torch.tensor(radii_vals)[mask]],
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

    # ---------------------------
    # Save and show figure
    # ---------------------------
    output_dir = os.path.join(trainer.base_dir, model_name, "Visualizations")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "latent_space.png")
    plt.savefig(plot_path, dpi=300)
    print(f"[INFO] Saved latent space visualization to: {plot_path}")

    # Show interactive window if available
    try:
        plt.show()
    except Exception:
        print("[WARN] Could not open interactive window (headless environment).")

    print(f"[INFO] Visualization complete for {model_name}")
