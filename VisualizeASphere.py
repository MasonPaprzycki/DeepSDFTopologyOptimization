import os, json, torch
import numpy as np
from skimage import measure
import trimesh
from DeepSDFStruct.deep_sdf.networks.deep_sdf_decoder import DeepSDFDecoder as Decoder

# -------------------------------
# Paths
# -------------------------------
root = "trained_models/sphere"
decoderCheckpoint = os.path.join(root, "ModelParameters", "latest.pth")
latentCheckpoint  = os.path.join(root, "LatentCodes", "latest.pth")
modelSpecs        = os.path.join(root, "specs.json")

# ------------------------------------------------
# Load network architecture
# ------------------------------------------------
with open(modelSpecs) as f:
    specs = json.load(f)

latent_size = specs["CodeLength"]
hidden_dims = specs["NetworkSpecs"]["dims"]

decoder = Decoder(
    latent_size   = latent_size,
    dims          = hidden_dims,
    geom_dimension= 3,
    norm_layers   = tuple(specs["NetworkSpecs"].get("norm_layers", ())),
    latent_in     = tuple(specs["NetworkSpecs"].get("latent_in", ())),
    weight_norm   = specs["NetworkSpecs"].get("weight_norm", False),
    xyz_in_all    = specs["NetworkSpecs"].get("xyz_in_all", False),
    use_tanh      = specs["NetworkSpecs"].get("use_tanh", False),
)

# ------------------------------------------------
# Load decoder parameters
# ------------------------------------------------
ckpt = torch.load(decoderCheckpoint, map_location="cpu")
decoder.load_state_dict(ckpt["model_state_dict"])
decoder.eval()



# -------------------------------
# Load latent vector robustly
# -------------------------------
latents = torch.load(latentCheckpoint, map_location="cpu")
codes   = latents["latent_codes"]           # OrderedDict
first_key = next(iter(codes))

first_item = codes[first_key]               # could be tensor OR dict

if isinstance(first_item, torch.Tensor):
    latentVector = first_item.view(1, -1)   # ensure shape (1, latent_dim)
elif isinstance(first_item, dict) and "latent_code" in first_item:
    latentVector = first_item["latent_code"].view(1, -1)
else:
    raise RuntimeError(
        f"Unexpected latent code type: {type(first_item)} "
        f"with keys {getattr(first_item,'keys',lambda:[])()}"
    )

print("Latent vector shape:", latentVector.shape)


# ------------------------------------------------
# Evaluate SDF on a 3-D grid
# ------------------------------------------------
grid_res = 128
x = y = z = np.linspace(-1.2, 1.2, grid_res)   # a bit larger than the unit sphere
grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), -1)
pts  = torch.from_numpy(grid.reshape(-1, 3)).float()

print("Querying decoder …")
with torch.no_grad():
    sdf_vals = []
    batch = 50000

    # ensure latent vector is (1, latent_dim)
    if latentVector.dim() == 1:
        latentVector = latentVector.unsqueeze(0)

    for i in range(0, len(pts), batch):
        chunk = pts[i:i + batch]
        latent_repeat = latentVector.repeat(chunk.size(0), 1)
        decoder_input = torch.cat([latent_repeat, chunk], dim=1)
        sdf_chunk = decoder(decoder_input)
        sdf_vals.append(sdf_chunk.squeeze(1).cpu())

    sdf = torch.cat(sdf_vals).numpy()


volume = sdf.reshape(grid_res, grid_res, grid_res)

# ------------------------------------------------
# Extract mesh with marching cubes
# ------------------------------------------------
print("Running marching cubes …")
verts, faces, normals, values = measure.marching_cubes(volume, level=0.0)

# Convert grid coords back to world coords
scale = (x[1] - x[0])
verts = verts * scale + np.array([x[0], y[0], z[0]])

mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

mesh.export("sphere_mesh.ply")
print("Saved mesh to sphere_mesh.ply")
