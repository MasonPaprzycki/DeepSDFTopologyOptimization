import os
import json
import torch
import numpy as np
import trimesh
from skimage import measure
from DeepSDFStruct.deep_sdf.networks.deep_sdf_decoder import DeepSDFDecoder as Decoder


def visualize_a_shape(
    model_name,
    scene_id=None,
    grid_res=128,
    clamp_dist=0.1,
    param_values=None,
    latent=None,
    save_suffix=None,
    experiment_root=None,
    grid_center=(0.0, 0.0, 0.0),
):
    """
    Visualize a DeepSDF-trained shape following the same folder structure that Model.py actually writes.
    (No nested 'model/' folder under the experiment directory.)
    """

    if scene_id is None and latent is None:
        print("[WARN] No scene ID or latent vector provided.")
        return []

    if not param_values:
        param_values = [None]

    meshes = []

    # ---------------- Paths ----------------
    # Model.py writes directly to experiment_root, not to experiment_root/model_name
    root = experiment_root if experiment_root else os.getcwd()

    model_params_dir = os.path.join(root, "ModelParameters")
    latent_dir = os.path.join(root, "LatentCodes")
    specs_file = os.path.join(root, "specs.json")

    # ---------------- Find checkpoints ----------------
    if not os.path.exists(model_params_dir):
        raise FileNotFoundError(f"[ERR] Missing folder: {model_params_dir}")

    decoder_ckpts = [
        f for f in os.listdir(model_params_dir)
        if f.endswith(".pth") and f[:-4].isdigit()
    ]
    if not decoder_ckpts:
        raise FileNotFoundError(f"No trained decoder checkpoints found in {model_params_dir}")
    latest_epoch = max(int(f[:-4]) for f in decoder_ckpts)
    decoder_checkpoint = os.path.join(model_params_dir, f"{latest_epoch}.pth")
    print(f"[INFO] Using decoder checkpoint → {decoder_checkpoint}")

    # ---------------- Load latent ----------------
    if latent is not None:
        latent_vector = latent.view(1, -1)
        print("[INFO] Using provided latent vector directly.")
    elif scene_id is not None:
        if not os.path.exists(latent_dir):
            raise FileNotFoundError(f"[ERR] Missing folder: {latent_dir}")

        latent_ckpts = [
            f for f in os.listdir(latent_dir)
            if f.endswith(".pth") and f[:-4].isdigit()
        ]
        if not latent_ckpts:
            raise FileNotFoundError(f"No latent checkpoints found in {latent_dir}")
        latest_latent_epoch = max(int(f[:-4]) for f in latent_ckpts)
        latent_path = os.path.join(latent_dir, f"{latest_latent_epoch}.pth")

        latent_data = torch.load(latent_path, map_location="cpu")
        latent_codes = latent_data["latent_codes"]["weight"]

        latent_vector = latent_codes[scene_id].view(1, -1)
        print(f"[INFO] Loaded latent vector for scene {scene_id} from epoch {latest_latent_epoch}.")
    else:
        raise RuntimeError("No latent data available.")

    # ---------------- Load decoder ----------------
    with open(specs_file) as f:
        specs = json.load(f)

    geom_dim = specs["NetworkSpecs"].get("geom_dimension", 3)
    decoder = Decoder(
        latent_size=specs["CodeLength"],
        dims=specs["NetworkSpecs"]["dims"],
        geom_dimension=geom_dim,
        norm_layers=tuple(specs["NetworkSpecs"].get("norm_layers", ())),
        latent_in=tuple(specs["NetworkSpecs"].get("latent_in", ())),
        weight_norm=specs["NetworkSpecs"].get("weight_norm", False),
        xyz_in_all=specs["NetworkSpecs"].get("xyz_in_all", False),
        use_tanh=specs["NetworkSpecs"].get("use_tanh", False),
    )

    ckpt = torch.load(decoder_checkpoint, map_location="cpu")
    decoder.load_state_dict(ckpt["model_state_dict"])
    decoder.eval()

    # ---------------- Grid ----------------
    x = np.linspace(grid_center[0] - 1.2, grid_center[0] + 1.2, grid_res)
    y = np.linspace(grid_center[1] - 1.2, grid_center[1] + 1.2, grid_res)
    z = np.linspace(grid_center[2] - 1.2, grid_center[2] + 1.2, grid_res)
    grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), -1)
    xyz_points = torch.from_numpy(grid.reshape(-1, 3)).float()

    # ---------------- Loop over parameter cases ----------------
    for idx, param_case in enumerate(param_values):
        pts = xyz_points.clone()

        if param_case is not None:
            param_tensor = torch.as_tensor(param_case, dtype=torch.float32).view(1, -1)
            pts = torch.cat([pts, param_tensor.repeat(pts.shape[0], 1)], dim=1)
            print(f"[INFO] Visualizing param case {idx+1}: {param_case}")
        else:
            print(f"[INFO] Visualizing default case {idx+1}")

        # Evaluate SDF
        sdf_vals = []
        with torch.no_grad():
            for i in range(0, len(pts), 50000):
                chunk = pts[i:i + 50000]
                latent_repeat = latent_vector.repeat(chunk.size(0), 1)
                decoder_input = torch.cat([latent_repeat, chunk], dim=1)
                sdf_chunk = decoder(decoder_input)
                sdf_vals.append(sdf_chunk.squeeze(1).cpu())
        sdf = torch.cat(sdf_vals).numpy()

        volume = np.clip(sdf.reshape(grid_res, grid_res, grid_res), -clamp_dist, clamp_dist)
        min_sdf, max_sdf = volume.min(), volume.max()
        if not (min_sdf < 0 < max_sdf):
            print(f"[WARN] No zero-crossing found (min={min_sdf:.4f}, max={max_sdf:.4f}) — skipping mesh.")
            continue

        verts, faces, normals, _ = measure.marching_cubes(volume, level=0.0)
        scale = x[1] - x[0]
        verts = verts * scale + np.array([x[0], y[0], z[0]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

        # ---------------- Save mesh ----------------
        mesh_dir = os.path.join(root, "Meshes")
        os.makedirs(mesh_dir, exist_ok=True)

        suffix = []
        if param_case is not None:
            safe_param = "_".join([
                f"{float(v):+.2f}".replace("+", "p").replace("-", "m")
                for v in np.atleast_1d(param_case)
            ])
            suffix.append(f"params{safe_param}")
        if save_suffix:
            suffix.append(save_suffix)
        suffix.append(f"case{idx:02d}")
        suffix_str = "_" + "_".join(suffix)

        mesh_filename = f"{model_name.lower()}_{scene_id}{suffix_str}_mesh.ply"
        mesh_path = os.path.join(mesh_dir, mesh_filename)
        mesh.export(mesh_path)
        print(f"[INFO] Saved mesh → {mesh_path}")

        meshes.append(mesh)

    return meshes
