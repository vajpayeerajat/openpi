import dataclasses

import einops
import numpy as np

from openpi import transforms

# Number of real (non-padding) dimensions in the Unitree G1 upper-body state and action
# vectors: 7 arm joints + 7 hand joints per side. The model runs with a padded
# action_dim of 32, so the trailing 4 dimensions are zero padding added by
# `PadStatesAndActions` and stripped back off in `G1Outputs`.
STATE_DIM = 28
ACTION_DIM = 28


def make_g1_example() -> dict:
    """Creates a random input example for the G1 policy."""
    return {
        "observation/state": np.random.rand(STATE_DIM),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the object and place it in the box",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class G1Inputs(transforms.DataTransformFn):
    """Converts G1 observations into the format the model expects.

    Used for both training and inference, so the keys here define the wire protocol that a
    policy client must send. The dataset side is mapped onto these keys by the repack
    transform in `LeRobotG1DataConfig`.
    """

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            # ego_view / ego_left / ego_right are all real cameras, so nothing is masked out.
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class G1Outputs(transforms.DataTransformFn):
    """Strips the model's padding back off the predicted actions. Inference only."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}
