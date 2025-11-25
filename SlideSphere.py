import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v2 as imageio

from DeepSDFStruct.sdf_primitives import SphereSDF
from Model import Model
import VisualizeAShape

# ======================================================
# Experiment Setup
# ======================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(SCRIPT_DIR)


EXPERIMENT_NAME = "SlidingSphere"
EXPERIMENT_ROOT = os.path.join(REPO_ROOT, "experiments", EXPERIMENT_NAME)

os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
os.makedirs(os.path.join(EXPERIMENT_ROOT, "plots"), exist_ok=True)
os.makedirs(os.path.join(EXPERIMENT_ROOT, "Meshes"), exist_ok=True)

print(f"[INFO] Experiment directory: {EXPERIMENT_ROOT}")

# ======================================================
# Scene Generation: Sphere at different x-positions
# ======================================================
def make_sphere_scene(cx: float):
    """Return SDF function for a sphere of radius 0.4 at x=cx."""
    return lambda xyz, params=None: SphereSDF(
        center=torch.tensor([cx, 0.0, 0.0], dtype=xyz.dtype, device=xyz.device),
        radius=0.4
    )._compute(xyz)

# Generate 12 scenes along x in [-0.8, 0.8]
num_scenes = 12
x_positions = np.linspace(-0.8, 0.8, num_scenes)

scenes = {}
for cx in x_positions:
    key = f"sphere_{cx:.2f}"
    scenes[key] = {0: (make_sphere_scene(cx), [])}

print(f"[INFO] Created {num_scenes} sphere scenes: {list(scenes.keys())}")

# ======================================================
# Initialize Model
# ======================================================
model = Model(
    base_directory=EXPERIMENT_ROOT,
    model_name="SlidingSphereModel",
    scenes=scenes,
    resume=False,
    latentDim=1,  # 1D latent
    NumEpochs=60, # more epochs for smooth latent interpolation
)

print("[INFO] Model initialized. Starting training...")

# ======================================================
# Training & Visualization
# ======================================================
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # Train model
    model.trainModel()
    print("[INFO] Training complete.")

    # ======================================================
    # Collect 1D latent codes
    # ======================================================
    print("[DEBUG] Scene keys: ", list(model.trained_scenes.keys()))
    scene_keys = sorted(
        list(model.trained_scenes.keys()), 
        key=lambda k: float(k.split("_")[2])
    )

    latents = torch.stack([
        model.trained_scenes[k].get_latent_vector().detach().cpu()
        for k in scene_keys
    ])
    latents_np = latents.numpy()

    # ------------------------------------------------------
    # Plot latent space (1D along x-axis)
    # ------------------------------------------------------
    plt.figure(figsize=(6, 3))
    plt.scatter(latents_np[:,0], np.zeros_like(latents_np[:,0]), color='royalblue', s=80)

    for i, key in enumerate(scene_keys):
        plt.text(latents_np[i,0] + 0.01, 0.0, key.split("_")[2], fontsize=9)

    plt.xlabel("Latent Dimension 1")
    plt.yticks([])
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(EXPERIMENT_ROOT, "plots", "latent_space_1d.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[INFO] 1D latent space plot saved to: {plot_path}")

    # ======================================================
    # Generate meshes along latent interpolation
    # ======================================================
    num_interp_steps = num_scenes
    interpolated_latents = latents  # already ordered along x
    mesh_paths = []

    print("[INFO] Generating meshes for each latent...")


    for i, latent in enumerate(interpolated_latents):
        # Extract center from scene key
        cx = float(scene_keys[i].split("_")[2])
        meshes = VisualizeAShape.visualize_a_shape(
            model_name="SlidingSphereModel",
            latent=latent,
            grid_res=96,
            clamp_dist=0.1,
            save_suffix=f"interp_{i:02d}",
            experiment_root=EXPERIMENT_ROOT,
            grid_center=(cx, 0.0, 0.0)
        )

        if meshes:
            mesh = meshes[0]
            mesh_path = os.path.join(EXPERIMENT_ROOT, "Meshes", f"interp_{i:02d}.ply")
            mesh.export(mesh_path)
            mesh_paths.append(mesh_path)
            print(f"[INFO] Saved mesh: {mesh_path}")
        else:
            print(f"[WARN] No mesh output for step {i}")

    print("[INFO] Mesh generation complete.")

    # ======================================================
    # Render animation
    # ======================================================
    print("[INFO] Rendering animation...")

    frames = []
    for mesh_path in mesh_paths:
        mesh = trimesh.load(mesh_path)
        if hasattr(mesh, "geometry") and isinstance(mesh.geometry, dict):
            geom_list = [
                geom for geom in mesh.geometry.values()
                if hasattr(geom, "vertices") and hasattr(geom, "faces")
            ]
            mesh = trimesh.util.concatenate(geom_list)

        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='3d')

        ax.add_collection3d(Poly3DCollection(
            mesh.vertices[mesh.faces],
            facecolor='lightblue',
            edgecolor='k',
            linewidth=0.1,
            alpha=1.0
        ))

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)
        ax.view_init(elev=25, azim=30)
        ax.set_box_aspect([1, 1, 1])
        plt.axis('off')

        frame_path = mesh_path.replace(".ply", ".png")
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        frames.append(imageio.imread(frame_path))

    gif_path = os.path.join(EXPERIMENT_ROOT, "sliding_sphere.gif")
    imageio.mimsave(gif_path, frames, duration=0.2)
    print(f"[INFO] Animation saved: {gif_path}")
