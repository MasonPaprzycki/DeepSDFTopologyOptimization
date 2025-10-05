import os
import json
import torch
import numpy as np
import trimesh
from skimage import measure
from DeepSDFStruct.deep_sdf.networks.deep_sdf_decoder import DeepSDFDecoder as Decoder

import os
import json
import torch
import numpy as np
import trimesh
from skimage import measure

# Make sure Decoder is imported from your DeepSDF code
from DeepSDFStruct.deep_sdf.networks.deep_sdf_decoder import DeepSDFDecoder as Decoder


def visualize_a_shape(model_name, scene_id=0, grid_res=128, clamp_dist=0.1):
    """
    Visualize a specific shape from a trained DeepSDF model with multiple shapes per model folder.
    Safely handles volumes that may not cross zero.

    Args:
        model_name (str): Name of the model folder (e.g., "Sphere").
        scene_id (int or str): Scene index (zero-padded internally).
        grid_res (int): Resolution of the 3D grid.
        clamp_dist (float): Maximum SDF value for clamping to ensure marching cubes works.

    Returns:
        trimesh.Trimesh: The reconstructed mesh.
    """

    # ------------------------
    # Paths
    # ------------------------
    root = os.path.join("trained_models", model_name)
    scene_str = f"{scene_id:03d}" if isinstance(scene_id, int) else str(scene_id)
    scene_key = f"{model_name.lower()}_{scene_str}"

    latentCheckpoint = os.path.join(root, "LatentCodes", "latest.pth")
    decoderCheckpoint = os.path.join(root, "ModelParameters", "latest.pth")
    modelSpecs = os.path.join(root, "specs.json")

    # ------------------------
    # Load latent vector
    # ------------------------
    latents = torch.load(latentCheckpoint, map_location="cpu")["latent_codes"]

    if scene_key not in latents:
        raise KeyError(f"Scene key '{scene_key}' not found in latent codes for {model_name}.")

    latent_entry = latents[scene_key]
    if isinstance(latent_entry, dict):
        latentVector = latent_entry.get("latent_code")
        if latentVector is None:
            raise RuntimeError(f"latent dict does not contain 'latent_code': {latent_entry.keys()}")
    elif isinstance(latent_entry, torch.Tensor):
        latentVector = latent_entry
    else:
        raise RuntimeError(f"Unexpected latent type: {type(latent_entry)}")

    latentVector = latentVector.view(1, -1)

    # ------------------------
    # Load model specs
    # ------------------------
    with open(modelSpecs) as f:
        specs = json.load(f)

    latent_size = specs["CodeLength"]
    hidden_dims = specs["NetworkSpecs"]["dims"]

    # ------------------------
    # Build decoder
    # ------------------------
    decoder = Decoder(
        latent_size=latent_size,
        dims=hidden_dims,
        geom_dimension=3,
        norm_layers=tuple(specs["NetworkSpecs"].get("norm_layers", ())),
        latent_in=tuple(specs["NetworkSpecs"].get("latent_in", ())),
        weight_norm=specs["NetworkSpecs"].get("weight_norm", False),
        xyz_in_all=specs["NetworkSpecs"].get("xyz_in_all", False),
        use_tanh=specs["NetworkSpecs"].get("use_tanh", False)
    )

    ckpt = torch.load(decoderCheckpoint, map_location="cpu")
    decoder.load_state_dict(ckpt["model_state_dict"])
    decoder.eval()

    # ------------------------
    # Evaluate SDF on 3D grid
    # ------------------------
    x = y = z = np.linspace(-1.2, 1.2, grid_res)
    grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), -1)
    pts = torch.from_numpy(grid.reshape(-1, 3)).float()

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
    volume = sdf.reshape(grid_res, grid_res, grid_res)



    # After computing volume (volume = sdf.reshape(...))

    min_val, max_val = volume.min(), volume.max()
    if min_val == max_val:
        # completely flat volume, fallback to tiny negative–positive range
        print(f"Warning: volume is flat (min=max={min_val}). Creating tiny perturbation.")
        volume = np.clip(volume, min_val - 1e-5, min_val + 1e-5)
        level = min_val
    elif min_val > 0:
        # all values positive → shape is "empty", use min_val for marching cubes
        print(f"Warning: volume entirely positive (min={min_val:.4f}, max={max_val:.4f}). Using level={min_val:.4f}")
        level = min_val
    elif max_val < 0:
        # all values negative → shape is "full", use max_val for marching cubes
        print(f"Warning: volume entirely negative (min={min_val:.4f}, max={max_val:.4f}). Using level={max_val:.4f}")
        level = max_val
    else:
        # zero is inside range → use level=0
        level = 0.0

    verts, faces, normals, _ = measure.marching_cubes(volume, level=level)

    # ------------------------
    # Marching cubes
    # ------------------------
    scale = (x[1] - x[0])
    verts = verts * scale + np.array([x[0], y[0], z[0]])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    # Save mesh
    meshFileName = f"{scene_key}_mesh.ply"
    mesh.export(meshFileName)
    print(f"Saved mesh to {meshFileName}")

    return mesh
