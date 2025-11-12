import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v2 as imageio
import trimesh

from DeepSDFStruct.sdf_primitives import SphereSDF
from Model import Model
import VisualizeAShape


# ---------------------------
# Experiment setup
# ---------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(SCRIPT_DIR)


EXPERIMENT_NAME = "ParamSphereSlide"
EXPERIMENT_ROOT = os.path.join(REPO_ROOT, "experiments", EXPERIMENT_NAME)
os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
os.makedirs(os.path.join(EXPERIMENT_ROOT, "plots"), exist_ok=True)

print(f"[DEBUG] EXPERIMENT_ROOT = {EXPERIMENT_ROOT}")

# ---------------------------
# Define a parameterized Sphere scene that slides along x
# ---------------------------

def make_sliding_sphere_scene():
    def sdf_fn(xyz: torch.Tensor, params: torch.Tensor | None):
        if params is None:
            cx = 0.0
        else:
            cx = params[:, 0] if params.dim() > 1 else params[0]
        center = torch.tensor([cx, 0.0, 0.0])
        return SphereSDF(center=center, radius=0.4)._compute(xyz)
    return sdf_fn


# ---------------------------
# Scene dictionary
# ---------------------------

scenes = {
    "sphere_slide": {
        0: (make_sliding_sphere_scene(), [(-0.8, 0.8)])
    }
}


# ---------------------------
# Initialize the model
# ---------------------------

model = Model(
    base_directory=EXPERIMENT_ROOT,
    model_name="model",
    scenes=scenes,
    resume=False,
    latentDim=1,
    NumEpochs=5,
)


# ---------------------------
# Training and visualization
# ---------------------------

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    print("[INFO] Starting training...")
    model.trainModel()

    # -----------------------------------------
    # Latent interpolation visualization
    # -----------------------------------------
    print("[INFO] Interpolating between first and last scenes...")

    trained_keys = list(model.trained_scenes.keys())
    first_key, last_key = trained_keys[0], trained_keys[-1]

    latent_start = model.trained_scenes[first_key].get_latent_vector().detach()
    latent_end = model.trained_scenes[last_key].get_latent_vector().detach()

    print(f"[INFO] Interpolating between '{first_key}' and '{last_key}'")

    num_steps = 10
    interpolated_latents = [
        (1 - t) * latent_start + t * latent_end
        for t in np.linspace(0.0, 1.0, num_steps)
    ]

    mesh_paths = []

    num_steps = 10
    x_positions = np.linspace(-0.8, 0.8, num_steps)

    for i, latent in enumerate(interpolated_latents):
        grid_center = (x_positions[i], 0.0, 0.0)
        meshes = VisualizeAShape.visualize_a_shape(
            model_name="model",
            latent=latent,
            grid_res=96,
            clamp_dist=0.1,
            save_suffix=f"latent_interp_{i:02d}",
            experiment_root=EXPERIMENT_ROOT,
            grid_center=grid_center
        )

        if meshes:
            mesh_dir = os.path.join(EXPERIMENT_ROOT, "trained_models", "model", "Meshes")
            os.makedirs(mesh_dir, exist_ok=True)
            mesh_filename = f"model_latent_latent_interp_{i:02d}_mesh.ply"
            mesh_path = os.path.join(mesh_dir, mesh_filename)
            mesh_paths.append(mesh_path)

    print(f"[INFO] Latent interpolation visualization complete!")

    
    # -----------------------------------------
    # Render animation from meshes
    # -----------------------------------------
    print("[INFO] Rendering animation...")

    frames = []
    for mesh_path in mesh_paths:
        mesh = trimesh.load(mesh_path)

        # Handle general Geometry types
        if hasattr(mesh, "geometry") and isinstance(mesh.geometry, dict):
            # It's a Scene -> combine geometries
            combined = []
            for geom in mesh.geometry.values():
                if hasattr(geom, "vertices") and hasattr(geom, "faces"):
                    combined.append(geom)
            mesh = trimesh.util.concatenate(combined)

        elif not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
            raise TypeError(f"Loaded object from {mesh_path} is not a triangular mesh!")

        # Now we are guaranteed to have a Trimesh-like object
        vertices = mesh.vertices
        faces = mesh.faces

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')

        ax.add_collection3d(Poly3DCollection(
            vertices[faces],
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

        frame_path = os.path.join(EXPERIMENT_ROOT, f"frame_{os.path.basename(mesh_path)}.png")
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        frames.append(imageio.imread(frame_path))

    gif_path = os.path.join(EXPERIMENT_ROOT, "sphere_slide_animation.gif")
    imageio.mimsave(gif_path, frames, duration=0.2)

    print(f"[INFO] Animation saved to {gif_path}")