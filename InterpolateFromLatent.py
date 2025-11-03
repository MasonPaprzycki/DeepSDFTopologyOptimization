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

def torus_fn(xyz: torch.Tensor, params: torch.Tensor | None = None) -> torch.Tensor:
    return TorusSDF(center=torch.tensor([0.0,0.0,0.0]), R=0.5, r=0.2)._compute(xyz)

def sphere_fn(xyz: torch.Tensor, params: torch.Tensor | None = None) -> torch.Tensor:
    return SphereSDF(center=torch.tensor([0.0,0.0,0.0]), radius=0.5)._compute(xyz)

def cylinder_fn(xyz: torch.Tensor, params: torch.Tensor | None = None) -> torch.Tensor:
    return CylinderSDF(point=torch.tensor([0.0,0.0,0.0]), axis="y", radius=0.5)._compute(xyz)

def corner_sphere_fn(xyz: torch.Tensor, params: torch.Tensor | None = None) -> torch.Tensor:
    return CornerSpheresSDF(radius=0.5)._compute(xyz)

scenes = {
    "torus": {0: (torus_fn, [])},
    "sphere": {0: (sphere_fn, [])},
    "cylinder": {0: (cylinder_fn, [])},
    "corner_sphere": {0: (corner_sphere_fn, [])}
}

# ---------------------------
# Model initialization
# ---------------------------

model = Model(
    base_directory=EXPERIMENT_ROOT,
    model_name="model",
    scenes=scenes,
    resume=True,
    latentDim=1
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
    # Plot final latent codes
    # ---------------------------

    plt.figure(figsize=(8,4))
    plt.scatter(range(len(scene_keys)), latents, color='blue')
    plt.xticks(range(len(scene_keys)), scene_keys, rotation=45, ha='right')
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
