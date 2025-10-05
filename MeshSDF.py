import trimesh
import mesh_to_sdf
import torch
from typing import Any

class MeshSDF:
    def __init__(self, mesh_path: str, repair: bool = True, scale_to_unit: bool = True):
        # Load the mesh
        self.mesh: trimesh.Trimesh = trimesh.load(mesh_path, force="mesh")  # type: ignore

        # Attempt to repair if not watertight
        if not getattr(self.mesh, "is_watertight", False):
            print("⚠️ Mesh not watertight.")
            if repair:
                print("Attempting to repair mesh...")
                self.mesh.fill_holes()
                self.mesh.remove_degenerate_faces()
                self.mesh.remove_unreferenced_vertices()
                if self.mesh.is_watertight:
                    print("Mesh successfully repaired and is now watertight.")
                else:
                    print("Mesh repair failed. SDF sign may be unreliable.")

        # Center and scale mesh to fit in [-1, 1]^3
        if scale_to_unit:
            self.center_and_scale()

    def center_and_scale(self):
        # Center the mesh at the origin
        centroid = self.mesh.centroid
        self.mesh.vertices -= centroid

        # Scale mesh to fit in unit cube [-1,1]
        max_extent = self.mesh.bounds[1] - self.mesh.bounds[0]  # [x, y, z] extents
        scale = 1.0 / max(max_extent)  # scale so largest side fits [-1,1]
        self.mesh.vertices *= scale

        print(f"Mesh centered at origin and scaled by {scale:.4f} to fit in [-1,1]^3.")

    def compute(self, points: torch.Tensor) -> torch.Tensor:
        # Convert to numpy
        pts = points.detach().cpu().numpy()
        # Call signed_distance from mesh_to_sdf
        sdf_func = getattr(mesh_to_sdf, "signed_distance")
        sdf_values = sdf_func(self.mesh, pts)
        return torch.from_numpy(sdf_values.astype("float32")).unsqueeze(1)

