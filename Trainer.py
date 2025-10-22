import os
from typing import Dict, List, Callable, Optional, Tuple

import torch
import TrainModel

SDFCallable = TrainModel.SDFCallable
Scenes = TrainModel.Scenes
Models  = TrainModel.Models

class Trainer:
    """
    Wrapper to train multiple DeepSDF models with multiple scenes using
    the updated trainModel API and new data types:
    
    SDFCallable = Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]
    Scenes = Dict[str, Dict[int, Tuple[SDFCallable, List[Tuple[float, float]]]]]
    Models = Dict[str, Scenes]
    """

    def __init__(self, base_dir: str = "trained_models"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def train_models(
        self,
        models: Models,
        resume: bool = True,
        latentDim: int = 1,
        FORCE_ONLY_FINAL_SNAPSHOT: bool = False,
        domainRadius: float = 1.0
    ):
        """
        Train multiple DeepSDF models with multiple scenes using trainModel.
        """
        for model_name, scenes_dict in models.items():
            print(f"[INFO] Training model '{model_name}' with {len(scenes_dict)} scenes...")
            TrainModel.trainModel(
                base_directory=self.base_dir,
                model_name=model_name,
                scenes=scenes_dict,
                latentDim=latentDim,
                resume=resume,
                FORCE_ONLY_FINAL_SNAPSHOT=FORCE_ONLY_FINAL_SNAPSHOT,
                domainRadius=domainRadius,
            )

    @staticmethod
    def batch_sdf(
        queries: torch.Tensor,
        scenes: Scenes
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Evaluate SDF for a batch of queries across multiple models and scenes.

        Parameters
        ----------
        queries : torch.Tensor
            Tensor of shape (N, 3) containing query points.
        scenes : Scenes
            Dictionary mapping model_name -> scene_id -> (SDFCallable, list of param ranges)

        Returns
        -------
        Dict[str, Dict[int, torch.Tensor]]
            Dictionary mapping model_name -> scene_id -> SDF values
        """
        results: Dict[str, Dict[int, torch.Tensor]] = {}
        for model_name, model_scenes in scenes.items():
            results[model_name] = {}
            for scene_id, (sdf_fn, _) in model_scenes.items():
                # Pass None for operator parameters for now
                results[model_name][scene_id] = sdf_fn(queries, None)
        return results
