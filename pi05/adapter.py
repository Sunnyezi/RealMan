"""OpenPI data mapping, also testable without installing the CUDA stack."""

from dataclasses import dataclass

import numpy as np


def parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError("Expected a three-dimensional image")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] != 3:
        raise ValueError("Expected RGB image")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Float training images must be in [0, 1]")
        image = np.rint(image * 255).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError("Inference images must be uint8")
    return image


@dataclass(frozen=True)
class RealManInputs:
    def __call__(self, data: dict) -> dict:
        base = parse_image(data["observation/image"])
        wrist = parse_image(data["observation/wrist_image"])
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (12,) or not np.isfinite(state).all():
            raise ValueError("Expected 6 arm radians + 6 hand actuator positions")
        result = {
            "state": state,
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(base),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != 12 or not np.isfinite(actions).all():
                raise ValueError("Expected twelve finite absolute action channels")
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclass(frozen=True)
class RealManOutputs:
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < 12:
            raise ValueError("Expected an action chunk with at least twelve channels")
        actions = actions[..., :12]
        if not np.isfinite(actions).all():
            raise ValueError("Nonfinite policy output")
        return {"actions": actions}


def training_config(repo_id: str, experiment: str, batch_size: int = 1, steps: int = 30000):
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config, optimizer, weight_loaders

    @dataclass(frozen=True)
    class RealManDataConfig(config.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            from dataclasses import replace

            return replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=transforms.Group(
                    inputs=[
                        transforms.RepackTransform(
                            {
                                "observation/image": "image",
                                "observation/wrist_image": "wrist_image",
                                "observation/state": "state",
                                "actions": "actions",
                                "prompt": "prompt",
                            }
                        )
                    ]
                ),
                data_transforms=transforms.Group(inputs=[RealManInputs()], outputs=[RealManOutputs()]),
                model_transforms=config.ModelTransformFactory()(model_config),
            )

    model = Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=10,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    return config.TrainConfig(
        name="pi05_realman_absolute12",
        exp_name=experiment,
        model=model,
        data=RealManDataConfig(repo_id=repo_id, base_config=config.DataConfig(prompt_from_task=True)),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=100, peak_lr=2e-5, decay_steps=30000, decay_lr=2e-6
        ),
        batch_size=batch_size,
        num_workers=0,
        num_train_steps=steps,
        wandb_enabled=False,
        save_interval=1000,
        keep_period=1000,
        policy_metadata={
            "robot": "RM65-6F-V+RH56DFTP-2R",
            "action_dim": 12,
            "control_dt": 0.05,
            "simulation_only": True,
            "task": "pick up and hold the blue cube; not pick and place",
            "action_units": "radians",
            "hand_control": "six active joints; fixed grasp demonstrations",
        },
    )
