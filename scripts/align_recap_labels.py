"""Match labels to LeRobot frame IDs, rejecting missing/extra/duplicate frames."""

import argparse
import json
from pathlib import Path

import numpy as np


def align(records, episode_map, episode_indices, frame_indices):
    if len(set(episode_map.values())) != len(episode_map):
        raise ValueError("Episode mapping must be one-to-one.")
    keyed = {}
    for r in records:
        key = (int(episode_map[r["episode_id"]]), int(r["t"]))
        if key in keyed:
            raise ValueError(f"Duplicate label: {key}")
        keyed[key] = r
    keys = list(zip(map(int, episode_indices), map(int, frame_indices), strict=True))
    if len(set(keys)) != len(keys) or set(keys) != set(keyed):
        raise ValueError("Dataset frames and label frames differ; use a train-only LeRobot dataset matching the JSON.")
    fields = {
        name: np.asarray([keyed[k][name] for k in keys], dtype=bool)
        for name in ("advantage_indicator", "use_advantage", "is_human_intervention")
    }
    fields.update(
        episode_index=np.asarray(episode_indices, dtype=np.int64), frame_index=np.asarray(frame_indices, dtype=np.int64)
    )
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--episode-map", type=Path, required=True, help='JSON: {"ep001": 0, ...}')
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if args.dataset_root is not None and not (args.dataset_root / "meta/info.json").is_file():
        raise FileNotFoundError(args.dataset_root)
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, video_backend="pyav")
    records = [json.loads(line) for line in args.labels.read_text().splitlines() if line.strip()]
    fields = align(
        records,
        json.loads(args.episode_map.read_text()),
        dataset.hf_dataset["episode_index"],
        dataset.hf_dataset["frame_index"],
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **fields)


if __name__ == "__main__":
    main()
