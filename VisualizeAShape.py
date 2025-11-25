import os
import json
import torch
import numpy as np
import trimesh
import torch
import numpy as np

from skimage import measure
from DeepSDFStruct.deep_sdf.networks.deep_sdf_decoder import DeepSDFDecoder as Decoder


def visualize_a_shape(
    model_name,
    scene_id=None,
    grid_center=(0.0, 0.0, 0.0),
    grid_res=128,
    clamp_dist=0.1,
    param_values=None,
    latent=None,
    save_suffix=None,
    experiment_root=None,
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

    # geom_dimension describes: xyz_dim (always 3) + num_params
    geom_dim = specs["NetworkSpecs"].get("geom_dimension", 3)
    xyz_dim = 3
    num_params = max(0, int(geom_dim - xyz_dim))

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

    # load network weights
    ckpt = torch.load(decoder_checkpoint, map_location="cpu")
    decoder.load_state_dict(ckpt["model_state_dict"])
    decoder.eval()

    # sanitize latent (ensure shape / dtype)
    latent_vector = latent_vector.detach().float().contiguous().view(1, -1)
    # pass num_params into the sampler so it can build correct decoder inputs for FD
    xyz_points, x, y, z = build_dynamic_sampling_grid(
        decoder=decoder,
        latent_vector=latent_vector,
        grid_res=grid_res,
        device=next(decoder.parameters()).device,
        grid_center=grid_center,
        num_params=num_params
    )


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

def build_dynamic_sampling_grid(
    decoder,
    latent_vector,
    grid_res,
    grid_center=(0.0, 0.0, 0.0),
    init_range=3.0,
    fd_eps=1e-3,
    max_steps=1000,
    step_scale=0.5,
    tol_surface=1e-3,
    n_surface_probes=512,
    probe_radius=0.15,
    probe_steps=20,
    bbox_margin_ratio=0.12,
    fallback_center=(0.0, 0.0, 0.0),
    fallback_radius=1.5,
    device=None,
    num_params: int = 0,
):
    """
    Builds a dynamic sampling grid around the shape surface found by probing from grid_center.
    This version enforces consistent dtype/device and robust shapes for single-point evaluation.
    """

    if device is None:
        device = latent_vector.device if isinstance(latent_vector, torch.Tensor) else torch.device("cpu")
    latent_vector = latent_vector.to(device)

    def make_decoder_input(eval_pts: torch.Tensor) -> torch.Tensor:
        # eval_pts: (M, 3) or (M, 3+num_params) depending on caller
        M = int(eval_pts.shape[0])
        latent_repeat = latent_vector.repeat(M, 1).to(device)
        pts_t = eval_pts.to(device)
        if num_params > 0:
            dummy_params = torch.zeros((M, num_params), dtype=pts_t.dtype, device=device)
            return torch.cat([latent_repeat, pts_t, dummy_params], dim=1)
        return torch.cat([latent_repeat, pts_t], dim=1)

    def batched_sdf_evals(pts: torch.Tensor):
        """
        Evaluate SDF (scalar) and gradient (3D) at pts using central finite differences.
        pts: (N,3) or (3,) torch tensor
        returns: (sdf0: (N,), grad: (N,3))
        """
        with torch.no_grad():
            # normalize pts shape to (N,3)
            if pts.ndim == 1:
                pts = pts.unsqueeze(0)
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError(f"batched_sdf_evals expects pts with shape (N,3), got {pts.shape}")

            pts = pts.to(device)
            M = int(pts.shape[0])

            # create fd offset vectors with same dtype/device as pts
            dtype = pts.dtype
            dev = pts.device
            eps = float(fd_eps)  # numeric scalar for denominators

            offs_x = torch.tensor([eps, 0.0, 0.0], dtype=dtype, device=dev).unsqueeze(0).repeat(M, 1)
            offs_y = torch.tensor([0.0, eps, 0.0], dtype=dtype, device=dev).unsqueeze(0).repeat(M, 1)
            offs_z = torch.tensor([0.0, 0.0, eps], dtype=dtype, device=dev).unsqueeze(0).repeat(M, 1)

            # construct evaluation points: base, +/- offs_x, +/- offs_y, +/- offs_z
            eval_pts = torch.cat([
                pts,
                pts + offs_x, pts - offs_x,
                pts + offs_y, pts - offs_y,
                pts + offs_z, pts - offs_z
            ], dim=0)

            # feed through decoder and split
            sdf_all = decoder(make_decoder_input(eval_pts)).squeeze(1)  # (7*M,)
            if sdf_all.shape[0] != 7 * M:
                raise RuntimeError(f"Expected {7*M} SDF values, got {sdf_all.shape[0]}")

            sdf0 = sdf_all[:M]
            # slices are sized M each
            s_px = sdf_all[M:2 * M]
            s_mx = sdf_all[2 * M:3 * M]
            s_py = sdf_all[3 * M:4 * M]
            s_my = sdf_all[4 * M:5 * M]
            s_pz = sdf_all[5 * M:6 * M]
            s_mz = sdf_all[6 * M:7 * M]

            # central differences
            g0 = (s_px - s_mx) / (2.0 * eps)
            g1 = (s_py - s_my) / (2.0 * eps)
            g2 = (s_pz - s_mz) / (2.0 * eps)

            grad = torch.stack([g0, g1, g2], dim=1)  # (M,3)

        return sdf0, grad

    # --- ensure grid_center is a 3-vector on correct device/dtype
    grid_center = torch.tensor(grid_center, dtype=torch.float32, device=device)
    if grid_center.numel() == 1:
        grid_center = grid_center.repeat(3)
    if grid_center.numel() != 3:
        raise ValueError(f"grid_center must be length-3, got {grid_center.numel()} elements")
    pt = grid_center.view(1, 3).to(device).detach().clone()  # (1,3) single probe point

    surface_points = []

    # Step from center toward surface searching for a zero crossing
    for _ in range(max_steps):
        sdf_val, grad = batched_sdf_evals(pt)  # sdf_val: (1,), grad: (1,3)
        sdf_abs = float(sdf_val.abs().item())
        if sdf_abs < tol_surface:
            surface_points.append(pt[0].cpu().numpy())
            break

        gnorm = float(grad.norm().item())
        if gnorm < 1e-8:
            # jitter to escape degenerate region; preserve dtype/device
            pt = (pt + (torch.randn_like(pt) * float(fd_eps))).detach()
            continue

        # compute unit gradient (shape consistent)
        g_unit = grad / (gnorm + 1e-12)  # (1,3)
        sign = -1.0 if float(sdf_val.item()) > 0.0 else 1.0
        step_len = float(step_scale * min(sdf_abs, float(init_range)))
        pt = (pt + sign * g_unit * step_len).detach()

    # If found surface: probe around it
    if surface_points:
        pt_surface = torch.tensor(surface_points[0], dtype=torch.float32, device=device).view(1, 3)
        # create random directions and probes (shape: (n_surface_probes, 3))
        directions = torch.randn(n_surface_probes, 3, device=device, dtype=pt_surface.dtype)
        directions = directions / (directions.norm(dim=1, keepdim=True) + 1e-12)
        rand_r = torch.rand(n_surface_probes, 1, device=device, dtype=pt_surface.dtype)
        probes = pt_surface.repeat(n_surface_probes, 1) + directions * probe_radius * rand_r

        for _ in range(probe_steps):
            sdf_vals, grads = batched_sdf_evals(probes)
            # signs: negative for positive SDF (move inward), positive for negative SDF
            signs = torch.where(sdf_vals >= 0.0, -1.0, 1.0).unsqueeze(1)
            gnorms = grads.norm(dim=1, keepdim=True)
            g_unit = grads / (gnorms + 1e-12)
            step_len = step_scale * sdf_vals.abs().unsqueeze(1)
            probes = (probes + signs * g_unit * step_len).detach()

        surface_points.extend(probes.cpu().numpy())

    # If no surface points found: return fallback uniform grid
    if not surface_points:
        cx, cy, cz = fallback_center
        x = np.linspace(cx - fallback_radius, cx + fallback_radius, grid_res)
        y = np.linspace(cy - fallback_radius, cy + fallback_radius, grid_res)
        z = np.linspace(cz - fallback_radius, cz + fallback_radius, grid_res)
        grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), -1)
        return torch.from_numpy(grid.reshape(-1, 3)).float(), x, y, z

    # Compute bounding box and build uniform grid around probe cloud
    cloud = np.array(surface_points)
    lo, hi = cloud.min(axis=0), cloud.max(axis=0)
    margin = bbox_margin_ratio * max((hi - lo).max(), 0.0)
    lo = lo - margin
    hi = hi + margin
    min_extent = 2.0 * float(fd_eps)
    hi = np.maximum(hi, lo + min_extent)

    x = np.linspace(float(lo[0]), float(hi[0]), grid_res)
    y = np.linspace(float(lo[1]), float(hi[1]), grid_res)
    z = np.linspace(float(lo[2]), float(hi[2]), grid_res)
    grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), -1)
    xyz_points = torch.from_numpy(grid.reshape(-1, 3)).float()

    return xyz_points, x, y, z
