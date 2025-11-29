import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from DeepSDFStruct.sdf_primitives import SphereSDF
from Model import Model
import VisualizeAShape

# ======================================================
# Experiment Setup
# ======================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(SCRIPT_DIR)

EXPERIMENT_NAME = "TrainOnASphere"
EXPERIMENT_ROOT = os.path.join(REPO_ROOT, "experiments", EXPERIMENT_NAME)

os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
os.makedirs(os.path.join(EXPERIMENT_ROOT, "plots"), exist_ok=True)
os.makedirs(os.path.join(EXPERIMENT_ROOT, "Meshes"), exist_ok=True)

print(f"[INFO] Experiment directory: {EXPERIMENT_ROOT}")

# ======================================================
# Scene: A Single Sphere
# ======================================================
def single_sphere_sdf():
    """Return SDF function for a sphere at the origin, radius 0.4."""
    return lambda xyz, params=None: SphereSDF(
        center=torch.tensor([0.0, 0.0, 0.0], dtype=xyz.dtype, device=xyz.device),
        radius=0.4
    )._compute(xyz)

scenes = {
    "sphere": {
        0: (single_sphere_sdf(), [])
    }
}

print("[INFO] Single sphere SDF scene created.")

# ======================================================
# Initialize Model
# ======================================================
model = Model(
    base_directory=EXPERIMENT_ROOT,
    model_name="TrainOnASphereModel",
    scenes=scenes,
    resume=False,
    latentDim=1,
    NumEpochs=100,
    domainRadius=0.45
)

print("[INFO] Model initialized. Starting training...")

# ======================================================
# Training & Visualization
# ======================================================
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # Train the model
    model.trainModel()
    print("[INFO] Training complete.")

    # ======================================================
    # Extract latent for the single scene
    # ======================================================
    if len(model.trained_scenes) != 1:
        raise RuntimeError("Expected exactly one trained scene.")

    scene_key = list(model.trained_scenes.keys())[0]
    latent = model.trained_scenes[scene_key].get_latent_vector().detach().cpu()

    print(f"[INFO] Latent for {scene_key}: {latent}")

    # ======================================================
    # Visualize shape once → use the mesh file that visualize_a_shape exports
    # ======================================================
    print("[INFO] Visualizing sphere from trained latent...")

    meshes = VisualizeAShape.visualize_a_shape(
        model_name="TrainOnASphereModel",
        latent=latent,
        grid_res=96,
        clamp_dist=0.1,
        save_suffix="single",
        experiment_root=EXPERIMENT_ROOT,
        grid_center=(0.0, 0.0, 0.0)
    )

    if not meshes:
        print("[WARN] No mesh produced by VisualizeAShape.")
        print("[INFO] Done.")
        quit()

    # VisualizeAShape already saved the mesh with a name like:
    #   trainonaspheremodel_None_single_case00_mesh.ply
    # So find the latest mesh file instead of exporting a duplicate.
    mesh_dir = os.path.join(EXPERIMENT_ROOT, "Meshes")
    all_mesh_files = [
        f for f in os.listdir(mesh_dir)
        if f.endswith(".ply") and f.startswith("trainonaspheremodel")
    ]
    if not all_mesh_files:
        raise FileNotFoundError("Mesh should have been exported by VisualizeAShape but none found.")

    # Take the most recent mesh (usually only one)
    all_mesh_files.sort()
    final_mesh_path = os.path.join(mesh_dir, all_mesh_files[-1])
    print(f"[INFO] Using exported mesh from VisualizeAShape: {final_mesh_path}")

    mesh = trimesh.load(final_mesh_path)

    # ======================================================
    # Simple static render of the result (no animation)
    # ======================================================
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

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=25, azim=30)
    ax.set_box_aspect([1, 1, 1])
    plt.axis('off')

    render_path = os.path.join(EXPERIMENT_ROOT, "plots", "trained_sphere_render.png")
    plt.savefig(render_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"[INFO] Render saved to: {render_path}")
    print("[INFO] Done.")
