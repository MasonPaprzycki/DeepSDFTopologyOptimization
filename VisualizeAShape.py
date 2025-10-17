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
    param_values=None,     # list of parameter vectors, e.g. [[1,2,3], [2,3,4]]
    latent=None,           # directly supplied latent vector, optional
    save_suffix=None,
    experiment_root=None,
):
    """
    Visualize a DeepSDF-trained shape with optional discrete parameter vectors.

    Args:
        model_name (str): Name of the model (directory under 'trained_models').
        scene_id (int, optional): Scene ID to load latent from. If None, must provide 'latent'.
        grid_res (int): Resolution of SDF grid.
        clamp_dist (float): Clamp distance for SDF values.
        param_values (List[List[float]], optional): List of parameter vectors, each defining a case.
        latent (torch.Tensor, optional): Direct latent vector override.
        save_suffix (str, optional): Suffix for saved mesh files.

    Returns:
        List[trimesh.Trimesh]: List of meshes generated.
    """

    # ---------------- Preconditions ----------------
    if scene_id is None and latent is None:
        print("there was no scenes or latent cases to visualize")
        return []

    # Ensure parameter cases are iterable
    if param_values is None or len(param_values) == 0:
        param_values = [None]  # visualize default case (no params)

    meshes = []

    # ---------------- Paths ----------------
    if experiment_root is not None:
        root = os.path.join(experiment_root, "trained_models", model_name)
    else:
        root = os.path.join("trained_models", model_name)
        
    scene_str = f"{scene_id:03d}" if isinstance(scene_id, int) else str(scene_id)
    scene_key = f"{model_name.lower()}_{scene_str}" if scene_id is not None else None

    scene_dir = os.path.join(root, "Scenes", scene_str) if scene_id is not None else root
    latent_dir = os.path.join(scene_dir, "LatentCodes") if scene_id is not None else None
    decoderCheckpoint = os.path.join(root, "ModelParameters", "latest.pth")
    specs_file = os.path.join(root, "specs.json")

    # ---------------- Load latent ----------------
    if latent is not None:
        latentVector = latent.view(1, -1)
        print("[INFO] Using provided latent vector directly.")

    elif scene_id is not None:
        scene_latest_latent = os.path.join(latent_dir, "latest.pth") # type: ignore
        if not os.path.exists(scene_latest_latent):
            raise FileNotFoundError(f"No latest.pth found for scene {scene_key}")
        latents = torch.load(scene_latest_latent, map_location="cpu")["latent_codes"]
        if scene_key not in latents:
            raise KeyError(f"Scene key '{scene_key}' not found in latent codes.")
        latentVector = latents[scene_key].view(1, -1)
        print(f"[INFO] Loaded latent for scene {scene_key}.")

    else:
        print("there was no scenes or latent cases to visualize")
        return []

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

    ckpt = torch.load(decoderCheckpoint, map_location="cpu")
    decoder.load_state_dict(ckpt["model_state_dict"])
    decoder.eval()

    # ---------------- Grid setup ----------------
    x = y = z = np.linspace(-1.2, 1.2, grid_res)
    grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), -1)
    xyz_points = torch.from_numpy(grid.reshape(-1, 3)).float()

  # ---------------- Visualize each param case ----------------
    for idx, param_case in enumerate(param_values or [None]):
        pts = xyz_points.clone()

        # ---------------- Append parameters to coordinates if given ----------------
        if param_case is not None:
            # Validate parameter format
            if not isinstance(param_case, (list, tuple, np.ndarray, torch.Tensor)):
                raise TypeError(
                    f"Each param_values entry must be a list, tuple, numpy array, or tensor. Got {type(param_case)}."
                )

            # Convert to torch tensor
            param_tensor = torch.as_tensor(param_case, dtype=torch.float32).view(1, -1)
            param_repeat = param_tensor.repeat(pts.shape[0], 1)
            pts = torch.cat([pts, param_repeat], dim=1)
            print(f"[INFO] Visualizing case {idx + 1}/{len(param_values)} with parameters: {param_case}")
        else:
            print(f"[INFO] Visualizing default (no parameter conditioning) case {idx + 1}.")

        # ---------------- Evaluate SDF ----------------
        sdf_vals = []
        batch = 50000
        with torch.no_grad():
            for i in range(0, len(pts), batch):
                chunk = pts[i:i + batch]
                latent_repeat = latentVector.repeat(chunk.size(0), 1)
                decoder_input = torch.cat([latent_repeat, chunk], dim=1)
                sdf_chunk = decoder(decoder_input)
                sdf_vals.append(sdf_chunk.squeeze(1).cpu())

        sdf = torch.cat(sdf_vals).numpy()
        volume = np.clip(sdf.reshape(grid_res, grid_res, grid_res), -clamp_dist, clamp_dist)

        # ---------------- Surface extraction ----------------
        verts, faces, normals, _ = measure.marching_cubes(volume, level=0.0)
        scale = x[1] - x[0]
        verts = verts * scale + np.array([x[0], y[0], z[0]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

        # ---------------- Save mesh ----------------
        mesh_dir = (
            os.path.join(scene_dir, "Meshes")
            if scene_id is not None
            else os.path.join(root, "Meshes")
        )
        os.makedirs(mesh_dir, exist_ok=True)

        # ---------------- Build filename suffix ----------------
        suffix_parts = []
        if isinstance(param_case, (list, tuple, np.ndarray, torch.Tensor)):
            # Ensure we iterate safely over parameter case
            safe_param_str = "_".join([
                f"{float(v):+.2f}".replace("+", "p").replace("-", "m")
                for v in np.atleast_1d(param_case)
            ])
            suffix_parts.append(f"params{safe_param_str}")

        if save_suffix:
            suffix_parts.append(str(save_suffix))

        suffix_parts.append(f"case{idx:02d}")
        suffix = "_" + "_".join(suffix_parts)

        # ---------------- File naming ----------------
        if scene_id is not None:
            mesh_filename = f"{model_name.lower()}_{scene_id:03d}{suffix}_mesh.ply"
        else:
            mesh_filename = f"{model_name.lower()}_latent{suffix}_mesh.ply"

        mesh_path = os.path.join(mesh_dir, mesh_filename)
        mesh.export(mesh_path)
        print(f"[INFO] Saved mesh → {mesh_path}")

        meshes.append(mesh)

    return meshes
