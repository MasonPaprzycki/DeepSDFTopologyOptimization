import os
import json
import torch
import numpy as np
import trimesh
from skimage import measure
from DeepSDFStruct.deep_sdf.networks.deep_sdf_decoder import DeepSDFDecoder as Decoder


def visualize_a_shape(
    model_name,
    scene_id=0,
    grid_res=128,
    clamp_dist=0.1,
    operator_value=None,
    latent_override=None,
    save_suffix=None,
):
    # ---------------- Paths ----------------
    root = os.path.join("trained_models", model_name)
    scene_str = f"{scene_id:03d}" if isinstance(scene_id, int) else str(scene_id)
    scene_key = f"{model_name.lower()}_{scene_str}"

    scene_dir = os.path.join(root, "Scenes", scene_str)
    latent_dir = os.path.join(scene_dir, "LatentCodes")
    decoderCheckpoint = os.path.join(root, "ModelParameters", "latest.pth")
    specs_file = os.path.join(root, "specs.json")
    scene_latest_latent = os.path.join(latent_dir, "latest.pth")

    # ---------------- Load latent ----------------
    if latent_override is not None:
        latentVector = latent_override.view(1, -1)
    else:
        if os.path.exists(scene_latest_latent):
            latents = torch.load(scene_latest_latent, map_location="cpu")["latent_codes"]
        else:
            raise FileNotFoundError(f"No latest.pth found for scene {scene_key}")

        if scene_key not in latents:
            raise KeyError(f"Scene key '{scene_key}' not found in latent codes.")
        latentVector = latents[scene_key].view(1, -1)

    # ---------------- Optional operator injection ----------------
    if operator_value is not None:
        latentVector = latentVector.clone()
        if latentVector.shape[1] >= 2:
            latentVector[0, 1] = operator_value
        else:
            latentVector = torch.cat([latentVector, torch.tensor([[operator_value]])], dim=1)
        print(f"[INFO] Injected operator_value={operator_value:.2f} into latent vector.")

    # ---------------- Load decoder ----------------
    with open(specs_file) as f:
        specs = json.load(f)

    decoder = Decoder(
        latent_size=specs["CodeLength"],
        dims=specs["NetworkSpecs"]["dims"],
        geom_dimension=3,
        norm_layers=tuple(specs["NetworkSpecs"].get("norm_layers", ())),
        latent_in=tuple(specs["NetworkSpecs"].get("latent_in", ())),
        weight_norm=specs["NetworkSpecs"].get("weight_norm", False),
        xyz_in_all=specs["NetworkSpecs"].get("xyz_in_all", False),
        use_tanh=specs["NetworkSpecs"].get("use_tanh", False),
    )

    ckpt = torch.load(decoderCheckpoint, map_location="cpu")
    decoder.load_state_dict(ckpt["model_state_dict"])
    decoder.eval()

    # ---------------- Evaluate SDF ----------------
    x = y = z = np.linspace(-1.2, 1.2, grid_res)
    grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), -1)
    pts = torch.from_numpy(grid.reshape(-1, 3)).float()

    sdf_vals = []
    batch = 50000
    with torch.no_grad():
        for i in range(0, len(pts), batch):
            chunk = pts[i:i+batch]
            latent_repeat = latentVector.repeat(chunk.size(0), 1)
            decoder_input = torch.cat([latent_repeat, chunk], dim=1)
            sdf_chunk = decoder(decoder_input)
            sdf_vals.append(sdf_chunk.squeeze(1).cpu())

    sdf = torch.cat(sdf_vals).numpy()
    volume = np.clip(sdf.reshape(grid_res, grid_res, grid_res), -clamp_dist, clamp_dist)

    # ---------------- Surface extraction ----------------
    level = 0.0
    min_val, max_val = volume.min(), volume.max()
    if min_val == max_val:
        volume = np.clip(volume, min_val - 1e-5, min_val + 1e-5)
        level = min_val
    elif min_val > 0:
        level = min_val
    elif max_val < 0:
        level = max_val

    verts, faces, normals, _ = measure.marching_cubes(volume, level=level)
    scale = x[1] - x[0]
    verts = verts * scale + np.array([x[0], y[0], z[0]])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    # ---------------- Save mesh ----------------
    mesh_dir = os.path.join(scene_dir, "Meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    suffix = f"_op{operator_value:.2f}" if operator_value is not None else ""
    if save_suffix:
        suffix += f"_{save_suffix}"

    meshFileName = os.path.join(mesh_dir, f"{scene_key}{suffix}_mesh.ply")
    mesh.export(meshFileName)
    print(f"[INFO] Saved mesh → {meshFileName}")

    return mesh
