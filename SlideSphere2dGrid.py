import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from DeepSDFStruct.sdf_primitives import SphereSDF
from Model import Model
import matplotlib.colors as mcolors
import multiprocessing as mp

def main(): 
    # ======================================================
    # Experiment Setup
    # ======================================================
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.abspath(SCRIPT_DIR)

    EXPERIMENT_NAME = "SlidingSphere2D"
    EXPERIMENT_ROOT = os.path.join(REPO_ROOT, "experiments", EXPERIMENT_NAME)
    os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
    os.makedirs(os.path.join(EXPERIMENT_ROOT, "frames"), exist_ok=True)

    print(f"[INFO] Experiment directory: {EXPERIMENT_ROOT}")

    # ======================================================
    # Generate scenes (must match Model expectations)
    #
    # KEY REQUIREMENTS FROM YOUR TRAINING CODE:
    #   - scenes is a dict[str, dict[int, (sdf_fn, param_ranges)]]
    #   - scene keys MUST be the strings used literally in TrainSplit.json
    #   - scene IDs are strings, not ints
    #   - each operator = key 0
    # ======================================================

    def make_sphere_scene(cx: float):
        def sdf_fn(xyz, params=None):
            return SphereSDF(
                center=torch.tensor([cx, 0.0, 0.0], dtype=xyz.dtype, device=xyz.device),
                radius=0.5
            )._compute(xyz)
        return sdf_fn

    num_scenes = 12
    x_positions = np.linspace(-0.8, 0.8, num_scenes)

    scenes = {}
    for cx in x_positions:
        # scene_id must be EXACT and stable because Model will later re-read this key
        scene_id = f"sphere_{cx:.2f}"  # consistent and sortable
        scenes[scene_id] = {0: (make_sphere_scene(cx), [])}

    print(f"[INFO] Created {num_scenes} scenes")

    # ======================================================
    # Train Model
    # ======================================================
    model = Model(
        base_directory=EXPERIMENT_ROOT,
        model_name="SlidingSphereModel2D",
        scenes=scenes,
        resume=False,
        latentDim=1,
        NumEpochs=500,
    )

    print("[INFO] Training model...")
    model.trainModel()
    print("[INFO] Training complete.")

    # ======================================================
    # Sort trained scene keys by x-value
    # (Model stores scenes as SlidingSphereModel2D_<scene_id>)
    # ======================================================

    def extract_x_from_key(k: str):
        # k example: "slidingspheremodel2d_sphere_-0.80"
        raw = k.split("_")[-1]
        return float(raw)

    sorted_keys = sorted(model.trained_scenes.keys(), key=extract_x_from_key)
    sorted_scenes = [model.trained_scenes[k] for k in sorted_keys]

    latents = torch.stack([s.get_latent_vector() for s in sorted_scenes]).float()
    latent_min = latents[0]
    latent_max = latents[-1]

    print("[INFO] Latent min/max:", latent_min.item(), latent_max.item())

    # ======================================================
    # Build 2D grid
    # ======================================================
    grid_res = 256
    xv = np.linspace(-3, 3, grid_res)
    yv = np.linspace(-3, 3, grid_res)
    xx, yy = np.meshgrid(xv, yv)

    xyz_np = np.stack([xx, yy, np.zeros_like(xx)], axis=-1)
    xyz = torch.tensor(xyz_np, dtype=torch.float32).reshape(-1, 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xyz = xyz.to(device)

    # ======================================================
    # SDF evaluation helper
    # ======================================================
    def eval_sdf(latent_vec: torch.Tensor):
        sdf = model.compute_sdf_from_latent(
            latent_vector=latent_vec,
            xyz=xyz,
            params=None,
        )
        if sdf.dim() == 2:
            sdf = sdf[:, 0]
        return sdf.cpu().numpy().reshape(grid_res, grid_res)

    # ======================================================
    # Custom colormap (no TypedDict, no casts)
    # ======================================================
    cdict = {
        "red": [
            (0.0, 0.2, 0.2),
            (0.5, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ],
        "green": [
            (0.0, 0.0, 0.0),
            (0.5, 1.0, 1.0),
            (1.0, 0.4, 0.4),
        ],
        "blue": [
            (0.0, 0.6, 0.6),
            (0.5, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        ]
    }

    custom_cmap = mcolors.LinearSegmentedColormap("sdf_custom", cdict)
    norm = mcolors.TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=0.6)

    # ======================================================
    # Latent-space sweep
    # ======================================================
    interp_steps = 100
    latent_values = torch.linspace(0.0, 1.0, interp_steps).to(device)

    frame_dir = os.path.join(EXPERIMENT_ROOT, "frames_latents")
    os.makedirs(frame_dir, exist_ok=True)

    print("[INFO] Generating latent-sweep frames...")

    frames = []
    for i, t in enumerate(latent_values):
        latent_vec = (1 - t) * latent_min + t * latent_max
        sdf_img = eval_sdf(latent_vec)

        plt.figure(figsize=(5, 5))
        plt.imshow(
            sdf_img,
            extent=(-3, 3, -3, 3),
            cmap=custom_cmap,
            norm=norm,
            origin="lower",
        )
        plt.colorbar(label="SDF")
        plt.title(f"t = {t.item():.2f}")
        plt.xlabel("x")
        plt.ylabel("y")

        frame_path = os.path.join(frame_dir, f"latent_{i:03d}.png")
        plt.savefig(frame_path, dpi=120)
        plt.close()

        frames.append(imageio.imread(frame_path))

    print("[INFO] Frames generated.")

    gif_path = os.path.join(EXPERIMENT_ROOT, "latent_space_sweep.gif")
    imageio.mimsave(gif_path, frames, duration=0.05)

    print(f"[INFO] Saved: {gif_path}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
