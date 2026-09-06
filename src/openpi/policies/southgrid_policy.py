"""SouthGrid g1_omnipicker: absolute 18-D EEF targets and two RGB cameras."""

import dataclasses

import numpy as np

from openpi import transforms


def _rgb(image) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected CHW or HWC image, got {image.shape}")
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected three RGB channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Floating point images must be finite RGB in [0, 1]")
        image = np.rint(image * 255).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 or normalized floating point RGB, got {image.dtype}")
    return np.ascontiguousarray(image)


@dataclasses.dataclass(frozen=True)
class SouthGridInputs(transforms.DataTransformFn):
    """Accept either flat LeRobot frames or SouthGrid websocket observations.

    The left wrist is intentionally unused for this two-camera baseline. Physical
    actions stay absolute: no joint-delta or ALOHA gripper transforms are applied.
    Standard ModelTransformFactory pads state/actions from 18 to 32 after normalization.
    """

    def __call__(self, data: dict) -> dict:
        if "observation.state" in data:
            state = data["observation.state"]
            head = data["observation.images.cam_head"]
            wrist = data["observation.images.cam_wrist_r"]
        else:
            state = data["state"]
            head = data["images"]["cam_head"]
            wrist = data["images"]["cam_wrist_r"]
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (18,) or not np.isfinite(state).all():
            raise ValueError(f"Expected finite g1_omnipicker state (18,), got {state.shape}")
        head, wrist = _rgb(head), _rgb(wrist)
        result = {
            "state": state,
            "image": {
                "base_0_rgb": head,
                "left_wrist_0_rgb": np.zeros_like(head),
                "right_wrist_0_rgb": wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        if "action" in data:
            actions = np.asarray(data["action"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 18 or not np.isfinite(actions).all():
                raise ValueError(f"Expected finite action chunk (H, 18), got {actions.shape}")
            result["actions"] = actions
        return result


@dataclasses.dataclass(frozen=True)
class SouthGridOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :18]}
