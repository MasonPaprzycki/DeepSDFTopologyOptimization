import os
from typing import Dict, List, Callable, Optional, Tuple

import torch
import TrainModel

SDFCallable = Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]

class DeepSDFTrainer:
    """
    Wrapper to train multiple DeepSDF models with multiple scenes using
    the updated DeepSDFTopologyOptimization.trainAModel API.

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

    def train_models(
        self,
        model_scenes: Dict[str, List[SDFCallable]],
        sdf_parameters: Optional[Dict[str, List[dict]]] = None,
        starting_ids: Optional[Dict[str, int]] = None,
        resume: bool = True,
        latentDim: int = 1,
        FORCE_ONLY_FINAL_SNAPSHOT: bool = False,
        domainRadius: float = 1.0
    ):
        """
        Train multiple models with multiple scenes, using the new DeepSDFStruct.trainAModel.
        """

        sdf_parameters = sdf_parameters or {}
        starting_ids = starting_ids or {name: 0 for name in model_scenes}

        for model_name, sdfs in model_scenes.items():
            start_id = starting_ids.get(model_name, 0)
            scene_ids = [start_id + i for i in range(len(sdfs))]

            model_sdf_params = sdf_parameters.get(model_name, [{}] * len(sdfs))
            print(f"[INFO] Training model '{model_name}' with {len(sdfs)} scenes...")

            # Build scenes dictionary expected by trainAModel
            scenes_dict = {}
            for scene_id, sdf_func, params in zip(scene_ids, sdfs, model_sdf_params):
                # convert dict param ranges to list of tuples
                param_ranges = [(v[0], v[1]) for v in params.values()] if params else []
                scenes_dict[scene_id] = (sdf_func, param_ranges)

            TrainModel.trainModel(
                base_directory=self.base_dir,
                model_name=model_name,
                scenes={model_name.lower(): scenes_dict},
                latentDim=latentDim,
                resume=resume,
                FORCE_ONLY_FINAL_SNAPSHOT=FORCE_ONLY_FINAL_SNAPSHOT,
                domainRadius=domainRadius,
            )

@staticmethod
def _batch_sdf(
    queries: torch.Tensor,
    scenes: Dict[int, Tuple[SDFCallable, List[Tuple[float, float]]]]
) -> Dict[int, torch.Tensor]:
    """
    Evaluate SDF for a batch of queries across multiple scenes.

    Parameters
    ----------
    queries : torch.Tensor
        Tensor of shape (N, 3) containing query points.
    scenes : Dict[int, Tuple[SDFCallable, List[Tuple[float, float]]]]
        Dictionary mapping scene_id -> (SDFCallable, list of parameter ranges)

    Returns
    -------
    Dict[int, torch.Tensor]
        Dictionary mapping scene_id -> SDF values (one tensor per scene).
    """
    results: Dict[int, torch.Tensor] = {}
    for scene_id, (sdf_fn, _) in scenes.items():
        results[scene_id] = sdf_fn(queries, None)  # use None for operator parameters

    return results
