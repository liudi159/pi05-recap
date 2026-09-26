"""Offline Stiff dataset identity checks; no simulator or hardware imports."""

import hashlib
import json
from pathlib import Path

UNITS = ["rad"] * 6 + ["m"]
SEMANTICS = "absolute_joint_reference_and_single_finger_closure"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset_contract(root, *, require_training=False):
    root = Path(root).resolve()
    provenance = json.loads((root / "provenance.json").read_text())
    info = json.loads((root / "meta/info.json").read_text())
    contract = provenance["contract"]
    if provenance.get("schema") != "stiff_lab_dataset_v1":
        raise ValueError("Unknown Stiff dataset schema")
    if contract.get("robot") != "rm65_xense" or contract.get("units") != UNITS:
        raise ValueError("RM65/Xense radians/metres contract required; coupon sliders are incompatible")
    if contract.get("action_semantics") != SEMANTICS:
        raise ValueError("Absolute joint reference + single-finger closure required")
    if info["robot_type"] != contract["robot"] or info["fps"] != contract["fps"] or info["fps"] <= 0:
        raise ValueError("Robot identity or sample rate mismatch")
    for key in ("observation.state", "action"):
        if info["features"][key]["shape"] != [7]:
            raise ValueError("Seven channels required")
    for camera in ("external", "wrist"):
        feature = info["features"]["observation.images." + camera]
        if len(feature["shape"]) != 3 or feature["shape"][-1] != 3:
            raise ValueError("Actual HWC RGB camera feature required")
    files = provenance.get("dataset_files")
    if not files:
        raise ValueError("Hashed source dataset inventory required")
    for name, digest in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise ValueError(f"Changed or unsafe dataset file: {name}")
    groups = {}
    for episode in provenance["episodes"]:
        source, split = episode["source"], episode["split"]
        if groups.setdefault(source, split) != split:
            raise ValueError("Source trajectory leaks across dataset splits")
    if require_training:
        if provenance.get("training_ready") is not True or provenance.get("readback_passed") is not True:
            raise ValueError("Stiff source is not qualified for training; use offline alignment inspection only")
        if any(e.get("diagnostic") or e["split"] != "train" for e in provenance["episodes"]):
            raise ValueError("Training input must be a qualified train-only dataset")
    return provenance, info
