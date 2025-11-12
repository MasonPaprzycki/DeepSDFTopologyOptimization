import os
import torch
import matplotlib
matplotlib.use("Agg")  # Headless backend — no GUI
import matplotlib.pyplot as plt

from DeepSDFStruct.sdf_primitives import SphereSDF, CornerSpheresSDF, CylinderSDF, TorusSDF
from Model import Model  # import your class version

# ---------------------------
# Paths
# ---------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(SCRIPT_DIR)

EXPERIMENT_NAME = "Latent2DRadiusSweep"
EXPERIMENT_ROOT = os.path.join(REPO_ROOT, "experiments", EXPERIMENT_NAME)
os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
os.makedirs(os.path.join(EXPERIMENT_ROOT, "plots"), exist_ok=True)

print(f"[DEBUG] EXPERIMENT_ROOT = {EXPERIMENT_ROOT}")

radii = [0.2, 0.4, 0.6, 0.8]

def make_sphere_scene(r):
    return lambda xyz, params=None: SphereSDF(center=torch.zeros(3), radius=r)._compute(xyz)

def make_cylinder_scene(r):
    return lambda xyz, params=None: CylinderSDF(point=torch.zeros(3), axis="y", radius=r)._compute(xyz)

def make_torus_scene(r):
    return lambda xyz, params=None: TorusSDF(center=torch.zeros(3), R=r, r=r/3)._compute(xyz)

def make_corner_sphere_scene(r):
    return lambda xyz, params=None: CornerSpheresSDF(radius=r)._compute(xyz)

scenes = {}
for r in radii:
    scenes[f"sphere_{r}"] = {0: (make_sphere_scene(r), [])}
    scenes[f"cylinder_{r}"] = {0: (make_cylinder_scene(r), [])}
    scenes[f"torus_{r}"] = {0: (make_torus_scene(r), [])}
    scenes[f"corner_sphere_{r}"] = {0: (make_corner_sphere_scene(r), [])}
    
# ---------------------------
# Model initialization
# ---------------------------
model = Model(
    base_directory=EXPERIMENT_ROOT,
    model_name="Latent2DRadiusSweep",  # <-- no underscores to avoid KeyError
    scenes=scenes,
    resume=True,
    latentDim=2,
    NumEpochs=20
)

# ---------------------------
# Main experiment
# ---------------------------
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # Train
    model.trainModel()

    # ---------------------------
    # Collect 2D latent codes
    # ---------------------------
    scene_keys = list(model.trained_scenes.keys())
    latents = torch.stack([model.trained_scenes[k].get_latent_vector().detach().cpu() for k in scene_keys])
    latents_np = latents.numpy()

    # ---------------------------  
    # Plot 2D latent space  
    # ---------------------------  
    plt.figure(figsize=(6,6))
    plt.scatter(latents_np[:,0], latents_np[:,1], color='royalblue', s=80)

    # Annotate each point with the scene name only
    for i, key in enumerate(scene_keys):
        # strip model prefix
        scene_label = "_".join(key.split("_")[1:])
        plt.text(latents_np[i,0] + 0.02, latents_np[i,1] + 0.02, scene_label, fontsize=9)

    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(EXPERIMENT_ROOT, "plots", "latent_space_2d.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"[INFO] 2D latent space plot saved to: {plot_path}")
