"""RM65/Xense: six joint angles (rad), one finger closure (m), two RGB cameras."""

import dataclasses

import numpy as np

from openpi import transforms

CHANNELS = tuple(f"joint_{i}" for i in range(1, 7)) + ("slider_left",)
UNITS = ("rad",) * 6 + ("m",)
ACTION_SEMANTICS = "absolute_joint_reference_and_single_finger_closure"


def rgb_image(value):
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError("A single RGB image is required")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] != 3:
        raise ValueError("RGB channels are required; depth/BGR need explicit conversion")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Floating RGB must be in [0, 1]")
        image = np.rint(image * 255).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError("RGB must be uint8 or normalized float")
    return image


@dataclasses.dataclass(frozen=True)
class StiffInputs(transforms.DataTransformFn):
    def __call__(self, data):
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("Expected six joint radians and one finger closure in metres")
        external = rgb_image(data["observation/image"])
        wrist = rgb_image(data["observation/wrist_image"])
        result = {
            "state": state.copy(),
            "image": {"base_0_rgb": external, "left_wrist_0_rgb": wrist,
                      "right_wrist_0_rgb": np.zeros_like(wrist)},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_,
                           "right_wrist_0_rgb": np.False_},
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 7 or not np.isfinite(actions).all():
                raise ValueError("Expected [horizon, 7] absolute joint/finger targets")
            result["actions"] = actions.copy()
        for key in ("prompt", "advantage_indicator", "use_advantage", "is_human_intervention"):
            if key in data:
                result[key] = data[key]
        return result


@dataclasses.dataclass(frozen=True)
class StiffOutputs(transforms.DataTransformFn):
    def __call__(self, data):
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < 7 or not np.isfinite(actions[:, :7]).all():
            raise ValueError("Expected finite RM65 joint/finger actions")
        return {"actions": actions[:, :7].copy()}


def data_transforms():
    # Relative joint targets are anchored at the observed state at the chunk start.
    # Finger closure remains absolute; it is neither aperture nor a [0, 1] command.
    mask = transforms.make_bool_mask(6, -1)
    return transforms.Group(
        inputs=[StiffInputs(), transforms.DeltaActions(mask)],
        outputs=[transforms.AbsoluteActions(mask), StiffOutputs()],
    )


def repack_transform():
    return transforms.Group(inputs=[transforms.RepackTransform({
        "observation/state": "observation.state",
        "observation/image": "observation.images.external",
        "observation/wrist_image": "observation.images.wrist",
        "actions": "action", "prompt": "prompt",
    })])
