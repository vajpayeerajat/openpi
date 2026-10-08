"""Policy transforms for the G1 + Dex3 `pick_and_place-300` dataset.

Copy to `src/openpi/policies/g1_dex3_policy.py` in the openpi checkout.

Dataset facts this file encodes (verified against the dataset, 2026-08-25):

* `observation.state` / `action` are **43-dim** whole-body vectors:
      0..11   legs   (WBC-controlled, not teleoperated -> dropped)
      12..14  waist  (`enable_waist: false` during collection -> dropped)
      15..42  upper body, the 28 dims we train on
* The 28 upper-body dims are ordered `[L_arm 7, L_hand 7, R_arm 7, R_hand 7]`,
  **not** `[arm 14, hand 14]`. Anything that groups arms vs hands must interleave.
* Cameras are `ego_view` (head/chest, sees the table) and `ego_left` / `ego_right`
  (wrist-mounted), h264 640x480 @ 20 fps.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms

# Upper body = state/action indices 15..42 inclusive. Contiguous, so a slice is enough.
UPPER_BODY = slice(15, 43)
UPPER_BODY_DIM = 28

# Names of the 28 trained dims, in order. The robot-side control loop must apply the
# policy's 28 outputs to these joints in exactly this order.
UPPER_BODY_JOINTS = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # CHW -> HWC
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _upper_body(x: np.ndarray, name: str) -> np.ndarray:
    """Take the 28 trained dims, accepting either the full 43-dim vector or 28 already.

    Accepting both matters at serve time: the robot reads all 43 joints, but the runbook's
    client example sends 28. A silent mismatch here trains and serves different joints, so
    fail loudly on anything else.
    """
    x = np.asarray(x)
    if x.shape[-1] == 43:
        return x[..., UPPER_BODY]
    if x.shape[-1] == UPPER_BODY_DIM:
        return x
    raise ValueError(
        f"{name}: expected last dim 43 (whole body) or {UPPER_BODY_DIM} (upper body), got {x.shape[-1]}"
    )


@dataclasses.dataclass(frozen=True)
class G1Dex3Inputs(transforms.DataTransformFn):
    action_dim: int  # model action dim; 32 for this config (28 real + 4 pad)

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(_upper_body(data["state"], "state"), self.action_dim)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": _parse_image(data["images"]["base_0_rgb"]),
                "left_wrist_0_rgb": _parse_image(data["images"]["left_wrist_0_rgb"]),
                "right_wrist_0_rgb": _parse_image(data["images"]["right_wrist_0_rgb"]),
            },
            # All three cameras are present for every frame of every episode (verified:
            # 1047/1047 videos, frame counts exactly matching the parquet row counts).
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(
                _upper_body(data["actions"], "actions"), self.action_dim
            )
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class G1Dex3Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Keep the 28 real joints, drop the 4 padding dims. Order == UPPER_BODY_JOINTS.
        return {"actions": np.asarray(data["actions"][:, :UPPER_BODY_DIM])}
