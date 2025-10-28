import os
import json
import torch
import numpy as np
from DeepSDFStruct.sdf_primitives import SphereSDF, CornerSpheresSDF, CylinderSDF, TorusSDF
from Model import Model # import the class version
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import torch

# --- Assumed available in your environment ---
# from your_sdf_module import SphereSDF, CylinderSDF, TorusSDF, CornerSpheresSDF
# from your_model_module import Model
# import VisualizeAShape
# ------------------------------------------------

EXPERIMENT_NAME = "OneLinParamW1dDiscreteInterpolation"
EXPERIMENT_ROOT = os.path.abspath(os.path.join("experiments", EXPERIMENT_NAME))
os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
print(f"[DEBUG] EXPERIMENT_ROOT = {EXPERIMENT_ROOT}")

# ---------------------------
# SDFs
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
    "torus":{
        0: (torus_fn, [])
    },
    "sphere":{
        0: (sphere_fn, [])
    },
    "cylinder":{
        0: (cylinder_fn, [])
    },
    "corner_sphere":{
        0: (corner_sphere_fn, [])
    }
}

model = Model(
    base_directory=EXPERIMENT_ROOT,
    model_name="model", 
    scenes=scenes,
    resume=True,
    latentDim=1,

)
model.trainModel()

# Collect scene keys and their latent vectors
scene_keys = list(model.trained_scenes.keys())
latents = [model.trained_scenes[key].get_latent_vector().item() for key in scene_keys]

# Plot
plt.figure(figsize=(8,4))
plt.scatter(range(len(scene_keys)), latents, color='blue')
plt.xticks(range(len(scene_keys)), scene_keys, rotation=45, ha='right')
plt.xlabel("Scene")
plt.ylabel("Latent Value")
plt.title(f"Latent Codes for Model '{model.model_name}'")
plt.grid(True)
plt.tight_layout()
plt.show()