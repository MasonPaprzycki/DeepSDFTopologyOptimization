import os
import torch
from typing import Dict, List, Callable
import TrainAShape
import VisualizeAShape


class DeepSDFTrainer:
    """
    Wrapper to train and visualize multiple DeepSDF models with multiple scenes.
    """

    def __init__(self, base_dir="trained_models"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    # ------------------------
    # Train multiple models/scenes
    # ------------------------
    def train_models(
        self,
        model_scenes: Dict[str, List[Callable]],
        starting_ids: Dict[str, int] = None,
        resume: bool = True,
    ):
        """
        Train multiple models, each with multiple scenes.

        Args:
            model_scenes: Dict of model names -> list of SDF functions (one per scene)
            starting_ids: Dict of starting scene_id per model (default 0)
            resume: If True, continue training from latest checkpoints
        """
        if starting_ids is None:
            starting_ids = {name: 0 for name in model_scenes}

        for model_name, sdfs in model_scenes.items():
            start_id = starting_ids.get(model_name, 0)
            scene_ids = [start_id + idx for idx in range(len(sdfs))]
            print(f"[INFO] Training model '{model_name}' with {len(sdfs)} scenes…")

            for scene_id, sdf_func in zip(scene_ids, sdfs):
                print(f"  -> Training scene {scene_id:03d} for model '{model_name}'")
                TrainAShape.trainAShape(
                    model_name=model_name,
                    sdf_function=sdf_func,
                    scene_ids=[scene_id],
                    resume=resume,
                )

    @staticmethod
    def _batch_sdf(queries, sdfs, scene_ids):
        """
        Return SDF values for a batch of queries across multiple scenes.
        Only supports one scene at a time for now (picks first scene).
        """
        return sdfs[0](queries)

    # ------------------------
    # Visualize a model (single or all scenes)
    # ------------------------
    def visualize_model(self, model_name: str, all_scenes: bool = False):
        """
        Visualize one or all latents (scenes) of a model.

        Args:
            model_name (str): Name of the model folder.
            all_scenes (bool): If True, visualize all scenes. If False, only the first.

        Returns:
            List[trimesh.Trimesh]: Generated meshes
        """
        print(f"[INFO] Visualizing model '{model_name}' …")

        latent_dir = os.path.join(self.base_dir, model_name, "LatentCodes")
        if not os.path.exists(latent_dir):
            raise FileNotFoundError(
                f"[ERROR] LatentCodes directory missing for {model_name} at {latent_dir}"
            )

        # List all per-scene latent files (e.g., mymodel_000.pth)
        latent_files = sorted(
            f
            for f in os.listdir(latent_dir)
            if f.endswith(".pth") and "_" in f and f.split("_")[-1].split(".")[0].isdigit()
        )

        if not latent_files:
            raise RuntimeError(f"No per-scene latent files found for model '{model_name}'.")

        meshes = []

        if all_scenes:
            for latent_file in latent_files:
                scene_id = int(latent_file.split("_")[-1].split(".")[0])
                print(f"  -> Visualizing scene {scene_id:03d}")
                mesh = VisualizeAShape.visualize_a_shape(model_name, scene_id=scene_id)
                meshes.append(mesh)
        else:
            # Visualize only the first scene latent
            first_latent = latent_files[0]
            scene_id = int(first_latent.split("_")[-1].split(".")[0])
            print(f"  -> Visualizing first scene {scene_id:03d}")
            mesh = VisualizeAShape.visualize_a_shape(model_name, scene_id=scene_id)
            meshes.append(mesh)

        print(f"[INFO] Done visualizing model '{model_name}'.")
        return meshes
