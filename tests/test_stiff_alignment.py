"""Offline regression tests for units, camera identity, and absolute/delta action round trips."""

import json

import numpy as np
import pytest

from openpi import transforms
from openpi.policies import stiff_policy
from openpi.training.stiff_data import SEMANTICS, UNITS, sha256, validate_dataset_contract


def observation():
    return {
        "observation/state": np.array([1, 2, 3, 4, 5, 6, 0.01], dtype=np.float32),
        "observation/image": np.full((8, 10, 3), 17, dtype=np.uint8),
        "observation/wrist_image": np.full((3, 8, 10), 231 / 255, dtype=np.float32),
        "actions": np.array([[1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.025]], dtype=np.float32),
        "advantage_indicator": True, "use_advantage": True,
    }


def test_joint_deltas_do_not_change_finger_closure_or_source():
    raw = observation()
    original = raw["actions"].copy()
    group = stiff_policy.data_transforms()
    data = transforms.compose(group.inputs)(raw)
    np.testing.assert_allclose(data["actions"][0, :6], [.1, .2, .3, .4, .5, .6], atol=1e-6)
    np.testing.assert_allclose(data["actions"][0, 6], .025)
    np.testing.assert_array_equal(raw["actions"], original)
    restored = transforms.compose(group.outputs)(transforms.PadStatesAndActions(32)(data))
    np.testing.assert_allclose(restored["actions"], original, atol=1e-6)


def test_camera_identity_mask_and_conditioning_survive():
    data = stiff_policy.StiffInputs()(observation())
    assert (data["image"]["base_0_rgb"] == 17).all()
    assert (data["image"]["left_wrist_0_rgb"] == 231).all()
    assert not data["image"]["right_wrist_0_rgb"].any()
    assert data["image_mask"] == {"base_0_rgb": True, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False}
    assert data["advantage_indicator"] and data["use_advantage"]


def test_invalid_state_and_image_are_rejected():
    raw = observation()
    raw["observation/state"][0] = np.nan
    with pytest.raises(ValueError):
        stiff_policy.StiffInputs()(raw)
    with pytest.raises(ValueError):
        stiff_policy.rgb_image(np.ones((3, 8, 10)) * 255)


def test_configs_register_and_keep_local_dataset_root(tmp_path, monkeypatch):
    import dataclasses
    from openpi.training import config

    monkeypatch.setattr(config.ReCAPModelTransformFactory, "__call__", lambda self, model: transforms.Group())
    for name in ("pi05_stiff_rm65", "pi05_stiff_rm65_recap_lora"):
        cfg = config.get_config(name)
        factory = dataclasses.replace(cfg.data, local_root=str(tmp_path))
        data = factory.create(tmp_path / "assets", cfg.model)
        assert data.local_root == str(tmp_path) and data.stiff_contract
        assert data.action_sequence_keys == ("action",) and data.prompt_from_task
    assert config.SimpleDataConfig().data_transforms(cfg.model) == transforms.Group()


def test_diagnostic_and_wrong_robot_contract_cannot_enter_training(tmp_path):
    (tmp_path / "meta").mkdir()
    info = {"robot_type": "rm65_xense", "fps": 50, "features": {
        "observation.state": {"shape": [7]}, "action": {"shape": [7]},
        **{"observation.images." + c: {"shape": [8, 10, 3]} for c in ("external", "wrist")},
    }}
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    provenance = {"schema": "stiff_lab_dataset_v1", "contract": {
        "robot": "rm65_xense", "fps": 50, "units": UNITS, "action_semantics": SEMANTICS},
        "episodes": [{"source": "actual_source", "split": "train", "diagnostic": True}],
        "dataset_files": {"meta/info.json": sha256(tmp_path / "meta/info.json")},
        "training_ready": False, "readback_passed": True,
    }
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(provenance))
    validate_dataset_contract(tmp_path)
    with pytest.raises(ValueError, match="not qualified"):
        validate_dataset_contract(tmp_path, require_training=True)
    provenance["contract"]["units"] = ["m"] * 7
    path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="radians/metres"):
        validate_dataset_contract(tmp_path)
