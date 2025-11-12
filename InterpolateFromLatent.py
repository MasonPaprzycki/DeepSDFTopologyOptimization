import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend — no windows pop up
import matplotlib.pyplot as plt

from DeepSDFStruct.sdf_primitives import SphereSDF, CornerSpheresSDF, CylinderSDF, TorusSDF
from Model import Model  # import the class version

# ---------------------------
# Paths
# ---------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(SCRIPT_DIR)  # adjust ".." if needed

EXPERIMENT_NAME = "OneLinParamW1dDiscreteInterpolation"
EXPERIMENT_ROOT = os.path.join(REPO_ROOT, "experiments", EXPERIMENT_NAME)
os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
os.makedirs(os.path.join(EXPERIMENT_ROOT, "plots"), exist_ok=True)

print(f"[DEBUG] EXPERIMENT_ROOT = {EXPERIMENT_ROOT}")

# ---------------------------
# SDF definitions
# ---------------------------

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
    model_name="model",
    scenes=scenes,
    resume=True,
    latentDim=1,
    NumEpochs=20
)

# ---------------------------
# Main training
# ---------------------------

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # Train the model
    model.trainModel()

    # ---------------------------
    # Collect latent codes
    # ---------------------------

    scene_keys = list(model.trained_scenes.keys())
    latents = [model.trained_scenes[key].get_latent_vector() for key in scene_keys]

    # ---------------------------
    # Sort by shape then by radius
    # ---------------------------

    

    def sort_key(name):
        parts = name.split("_")
        # last part is radius, everything before is shape
        shape = "_".join(parts[:-1])
        try:
            r_val = float(parts[-1])
        except ValueError:
            r_val = 0.0
        return (shape, r_val)

    sorted_indices = sorted(range(len(scene_keys)), key=lambda i: sort_key(scene_keys[i]))
    sorted_scene_keys = [scene_keys[i] for i in sorted_indices]
    sorted_latents = [latents[i] for i in sorted_indices]

    # ---------------------------
    # Plot final latent codes
    # ---------------------------

    plt.figure(figsize=(8, 4))
    plt.scatter(range(len(sorted_scene_keys)), sorted_latents, color='blue')
    plt.xticks(range(len(sorted_scene_keys)), sorted_scene_keys, rotation=45, ha='right')
    plt.xlabel("Scene")
    plt.ylabel("Latent Value")
    plt.title(f"Latent Codes for Model '{model.model_name}'")
    plt.grid(True)
    plt.tight_layout()

    # Save to disk
    plot_path = os.path.join(EXPERIMENT_ROOT, "plots", "final_latents.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"[INFO] Latent code plot saved to {plot_path}")
