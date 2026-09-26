"""Verify an existing single-arm Stiff recording against the actual pi05 adapter, offline."""

import argparse
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np

from openpi.policies import stiff_policy
from openpi.training.stiff_data import SEMANTICS, UNITS, sha256, validate_dataset_contract
from openpi import transforms


def check_camera(camera, time):
    if not np.isfinite(camera["time_s"]) or abs(camera["time_s"] - time) > 1e-7:
        raise ValueError("RGB and pre-action state timestamps differ")
    k = np.asarray(camera["K"])
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("Invalid camera intrinsics")
    np.testing.assert_allclose(k[2], [0, 0, 1], atol=1e-7)
    if "T_world_camera" in camera or "T_world_camera_optical" not in camera:
        raise ValueError("Explicit optical-to-world calibration required, not OpenGL camera pose")
    t = np.asarray(camera["T_world_camera_optical"])
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("Invalid camera extrinsics")
    np.testing.assert_allclose(t[3], [0, 0, 0, 1], atol=1e-7)
    np.testing.assert_allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    provenance, info = validate_dataset_contract(args.dataset)
    episode = json.loads(args.episode.read_text())
    if episode.get("schema") != "stiff_lab_episode_v2" or episode.get("robot") != "rm65_xense":
        raise ValueError("Expected a Stiff RM65 episode")
    if episode.get("units") != UNITS or episode.get("action_semantics") != SEMANTICS:
        raise ValueError("Episode units/action semantics differ")
    if info["total_episodes"] != 1 or len(provenance["episodes"]) != 1:
        raise ValueError("Inspect each source episode separately before combining data")
    source = provenance["episodes"][0]
    if source.get("manifest_sha256") != sha256(args.episode):
        raise ValueError("Dataset does not reference this episode manifest")
    if source["source"] != episode["trajectory_source_id"]:
        raise ValueError("Physical trajectory source mismatch")
    if sha256(episode["trajectory"]) != episode["trajectory_sha256"]:
        raise ValueError("Changed trajectory")
    scene_hash = sha256(args.scene)
    if episode["source_metadata"].get("source_scene_sha256") != scene_hash:
        raise ValueError("This recording belongs to a different robot/camera assembly")
    scene = json.loads(args.scene.read_text())
    with np.load(episode["trajectory"], allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    n = len(data["time"])
    if n != info["total_frames"] or n != len(episode["cameras"]):
        raise ValueError("Frame count mismatch")
    for key in ("state", "action", "next_state"):
        if data[key].shape != (n, 7) or not np.isfinite(data[key]).all():
            raise ValueError("Seven finite state/action channels required")
    np.testing.assert_allclose(data["next_time"] - data["time"], 1 / info["fps"], atol=1e-7, rtol=0)
    np.testing.assert_allclose(data["next_time"][:-1], data["time"][1:], atol=1e-7, rtol=0)
    np.testing.assert_allclose(data["next_state"][:-1], data["state"][1:], atol=1e-7, rtol=0)
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset("local/stiff_rm65", root=args.dataset.resolve(),
        delta_timestamps={"action": [i / info["fps"] for i in range(50)]}, video_backend="pyav")
    group = stiff_policy.data_transforms()
    repack = stiff_policy.repack_transform().inputs[0]
    for i in range(n):
        row = dataset[i]
        np.testing.assert_allclose(row["observation.state"], data["state"][i], atol=1e-6, rtol=0)
        indices = np.minimum(np.arange(i, i + 50), n - 1)
        np.testing.assert_allclose(row["action"], data["action"][indices], atol=1e-6, rtol=0)
        if not np.array_equal(np.asarray(row["action_is_pad"]), np.arange(i, i + 50) >= n):
            raise ValueError("Action chunk crosses the source episode")
        loaded_time = np.asarray(row["simulation_time"])
        # LeRobot's tensor formatter converts numeric columns to float32, even
        # when the stored simulation_time feature is float64. Compare exactly
        # at that representation; raw camera/transition clocks stay float64.
        np.testing.assert_array_equal(loaded_time, np.asarray([data["time"][i]], dtype=loaded_time.dtype))
        mapped = repack({**row, "prompt": row["task"]})
        sample = transforms.compose(group.inputs)(mapped)
        for camera, slot in (("external", "base_0_rgb"), ("wrist", "left_wrist_0_rgb")):
            np.testing.assert_array_equal(sample["image"][slot], data["rgb_" + camera][i])
            check_camera(episode["cameras"][i][camera], data["time"][i])
        np.testing.assert_allclose(sample["actions"][:, :6], data["action"][indices, :6] - data["state"][i, :6], atol=1e-6)
        np.testing.assert_allclose(sample["actions"][:, 6], data["action"][indices, 6], atol=1e-7)
        padded = transforms.PadStatesAndActions(32)(sample)
        assert padded["state"].shape == (32,) and padded["actions"].shape == (50, 32)
        assert not padded["state"][7:].any() and not padded["actions"][:, 7:].any()
        restored = transforms.compose(group.outputs)({"state": padded["state"], "actions": padded["actions"].copy()})
        np.testing.assert_allclose(restored["actions"], data["action"][indices], atol=1e-6, rtol=0)
        assert not sample["image_mask"]["right_wrist_0_rgb"]
    args.output.mkdir(parents=True)
    binding = {
        "schema": "pi05_stiff_rm65_binding_v1", "dataset_root": str(args.dataset.resolve()),
        "episode_manifest": str(args.episode.resolve()), "scene": str(args.scene.resolve()),
        "scene_sha256": scene_hash, "source_id": episode["trajectory_source_id"],
        "channels": stiff_policy.CHANNELS, "units": UNITS, "raw_action_semantics": SEMANTICS,
        "policy_action_semantics": "joint_delta_from_chunk_start_and_absolute_single_finger_closure",
        "camera_slots": {"external": "base_0_rgb", "wrist": "left_wrist_0_rgb", "unused": "right_wrist_0_rgb"},
        "camera_masks": [True, True, False], "fps": info["fps"],
        "image_time_semantics": "offline_rendered_pre_action_simulation_time_not_hardware_capture_rate",
        "calibration": episode["cameras"], "scene_camera_mounts": scene["cameras"],
        "source_metadata": episode["source_metadata"], "training_ready": provenance.get("training_ready", False),
        "diagnostic": episode["diagnostic"], "full_success": episode["full_success"],
    }
    (args.output / "binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    report = {"alignment_passed": True, "frames_checked": n, "action_horizon": 50,
        "state_shape": [32], "action_shape": [50, 32], "camera_rgb_exact": True,
        "timestamps_and_calibration_verified": True, "joint_delta_roundtrip_verified": True,
        "finger_closure_unchanged": True, "episode_tail_padding_verified": True,
        "source_files_modified": False, "model_loaded": False, "training_ready": binding["training_ready"],
        "scene_sha256": scene_hash, "dataset_provenance_sha256": sha256(args.dataset / "provenance.json")}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
