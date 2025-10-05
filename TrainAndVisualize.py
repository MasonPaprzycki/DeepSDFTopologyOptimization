import Trainer
import DeepSDFStruct.sdf_primitives as sdf_primitives

# ---------------------------------------
# Example usage
# ---------------------------------------
if __name__ == "__main__":
    trainer = Trainer.DeepSDFTrainer()

    # ---------------------------
    # Define SDF functions per scene
    # ---------------------------
    # Each list element corresponds to one scene. scene_id will be assigned sequentially.
    model_scenes = {
        "Sphere": [
            sdf_primitives.SphereSDF(center=[0,0,0], radius=r)._compute
            for r in [0.1, 0.2, 0.3]  # small number for testing
        ],
        "CornerSphere": [
            sdf_primitives.CornerSpheresSDF(radius=r)._compute
            for r in [0.05, 0.1, 0.15]
        ],
        "Cylinder": [
            sdf_primitives.CylinderSDF(point=[0,0,0], axis="y", radius=r)._compute
            for r in [0.1, 0.2]
        ],
        "Torus": [
            sdf_primitives.TorusSDF(center=[0,0,0], R=R, r=r)._compute
            for R in [0.1, 0.2]
            for r in [0.05, 0.1]
        ]
    }

    # ---------------------------
    # Train all models/scenes
    # ---------------------------
    trainer.train_models(model_scenes, resume=True)

    # ---------------------------
    # Visualize models
    # ---------------------------
    for model_name in ["Sphere", "Cylinder", "Torus"]:
        print(f"\nVisualizing model: {model_name}")
        trainer.visualize_model(model_name, all_scenes=True)


