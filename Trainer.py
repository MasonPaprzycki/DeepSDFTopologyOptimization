import os
from typing import Dict, List, Callable, Optional
import TrainAShape

class DeepSDFTrainer:
    """
    Wrapper to train multiple DeepSDF models with multiple scenes.
    Each scene has its own folder:
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
        sdf_parameters: Optional[Dict[str, List[dict]]] = None,
        starting_ids: Optional[Dict[str, int]] = None,
        resume: bool = True,
        latentDim: int = 1,
        FORCE_ONLY_FINAL_SNAPSHOT: bool = False,
        domainRadius: float = 1.0,
    ):
        """
        Train multiple models, each with multiple scenes.

        Parameters
        ----------
        model_scenes : dict[str, list[Callable]]
            Dictionary mapping model names → list of SDF functions correlating to scenes.
        sdf_parameters : dict[str, list[dict]], optional
            Dictionary mapping model names → list of dicts per scene.
            Each dict specifies parameter ranges: {param_name: (low, high)}.
            Only parameters to be sampled should be included.
        starting_ids : dict[str, int], optional
            Optional starting scene ID per model.
        resume : bool
            Whether to resume from existing checkpoints.
        latentDim : int
            Latent vector dimension.
        FORCE_ONLY_FINAL_SNAPSHOT : bool
            Only save the final epoch snapshot if True.
        domainRadius : float
            Maximum absolute value for 3D query points.
        """
        if sdf_parameters is None:
            sdf_parameters = {}
        if starting_ids is None:
            starting_ids = {name: 0 for name in model_scenes}

        for model_name, sdfs in model_scenes.items():
            start_id = starting_ids.get(model_name, 0)
            scene_ids = [start_id + i for i in range(len(sdfs))]

            # Get per-scene parameter dicts
            model_sdf_params = sdf_parameters.get(model_name, [{}] * len(sdfs))

            print(f"[INFO] Training model '{model_name}' with {len(sdfs)} scenes…")

            for scene_id, sdf_func, params in zip(scene_ids, sdfs, model_sdf_params):
                print(f"  -> Training scene {scene_id:03d} for model '{model_name}'")

                # Wrap per-scene sdf_parameters in a list for trainAShape
                TrainAShape.trainAShape(
                    base_directory=self.base_dir,
                    model_name=model_name,
                    sdf_function=sdf_func,
                    scene_ids=[scene_id],
                    sdf_parameters=[params],
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
