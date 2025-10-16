import os
from typing import Dict, List, Callable, Optional
import TrainAShape
import VisualizeAShape

class DeepSDFTrainer:
    """
    Wrapper to train and visualize multiple DeepSDF models with multiple scenes.
    Each scene now has its own folder:
        trained_models/
            model_name/
                ModelParameters/
                Scenes/
                    000/
                        LatentCodes/
                            1.pth, 5.pth, latest.pth
                        model_000.npz
                    001/
                        LatentCodes/
                            ...
    """

    def __init__(self, base_dir: str = "trained_models"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    # ------------------------
    # Train multiple models/scenes
    # ------------------------
    def train_models(
        self,
        model_scenes: Dict[str, List[Callable]],
        starting_ids: Optional[Dict[str, int]] = None,
        resume: bool = True,
        sdf_parameters: Optional[List] = None,
        latentDim: int = 1,
        FORCE_ONLY_FINAL_SNAPSHOT: bool = False,
        domainRadius: float = 1.0,
    ):
        """
        Train multiple models, each with multiple scenes.

        Args:
            model_scenes (Dict[str, List[Callable]]): Dict of model names -> list of SDF functions (one per scene)
            starting_ids (Dict[str, int], optional): Starting scene_id per model (default 0)
            resume (bool): If True, continue training from latest checkpoints
            sdf_parameters (List[Tuple[float, float]], optional): List of (low, high) parameter ranges for SDF conditioning
            latentDim (int): Latent vector dimension
            FORCE_ONLY_FINAL_SNAPSHOT (bool): If True, only save the last snapshot
            domainRadius (float): Sampling domain radius
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
                    sdf_parameters=sdf_parameters,   
                    latentDim=latentDim,
                    resume=resume,
                    FORCE_ONLY_FINAL_SNAPSHOT=FORCE_ONLY_FINAL_SNAPSHOT,
                    domainRadius=domainRadius,
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
        Visualize one or all scenes for a given model using the scene-located latest.pth latents.

        Args:
            model_name (str): Name of the model folder.
            all_scenes (bool): If True, visualize all scenes. If False, only the first.

        Returns:
            List[trimesh.Trimesh]: Generated meshes.
        """
        print(f"[INFO] Visualizing model '{model_name}' …")

        scenes_root = os.path.join(self.base_dir, model_name, "Scenes")
        if not os.path.exists(scenes_root):
            raise FileNotFoundError(f"[ERROR] Scenes folder missing for {model_name} at {scenes_root}")

        scene_folders = sorted([f for f in os.listdir(scenes_root) if f.isdigit()])
        if not scene_folders:
            raise RuntimeError(f"No scene folders found in {scenes_root}")

        meshes = []

        if all_scenes:
            for scene_folder in scene_folders:
                scene_id = int(scene_folder)
                print(f"  -> Visualizing scene {scene_id:03d}")
                mesh = VisualizeAShape.visualize_a_shape(model_name, scene_id=scene_id)
                meshes.append(mesh)
        else:
            # Visualize only the first scene
            scene_id = int(scene_folders[0])
            print(f"  -> Visualizing first scene {scene_id:03d}")
            mesh = VisualizeAShape.visualize_a_shape(model_name, scene_id=scene_id)
            meshes.append(mesh)

        print(f"[INFO] Done visualizing model '{model_name}'.")
        return meshes
