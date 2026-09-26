"""Offline iteration orchestration. Default only prints a plan; --execute launches stages.

Each manifest iteration supplies cumulative training episodes with cached VLM features,
a separate held-out episode JSON, matching train-only LeRobot repo and explicit ID map.
Robot/simulator rollout collection is external; this is not an online HITL loop.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def commands(manifest, output):
    base = manifest["base_params"]
    config = manifest.get("config", "pi05_libero_recap_lora")
    if not manifest["iterations"]:
        raise ValueError("At least one offline iteration is required.")
    result = []
    for index, iteration in enumerate(manifest["iterations"]):
        target = output / f"iter_{index:03d}"
        value = target / "value"
        prefix = [sys.executable]
        result.append(
            prefix
            + [
                "scripts/learn_recap_value.py",
                "--episodes",
                iteration["episodes"],
                "--eval-episodes",
                iteration["eval_episodes"],
                "--output",
                str(value),
                "--steps",
                str(manifest.get("value_steps", 500)),
            ]
            + (["--state-only"] if manifest.get("state_only") else [])
        )
        result.append(
            prefix
            + [
                "scripts/align_recap_labels.py",
                "--labels",
                str(value / "labels.jsonl"),
                "--episode-map",
                iteration["episode_map"],
                "--repo-id",
                iteration["repo_id"],
                "--output",
                str(target / "fields.npz"),
            ]
        )
        result.append(
            prefix + ["scripts/compute_norm_stats.py", "--config-name", config, "--repo-id", iteration["repo_id"]]
        )
        result.append(
            prefix
            + [
                "scripts/train_recap_stage.py",
                "--config",
                config,
                "--repo-id",
                iteration["repo_id"],
                "--fields",
                str(target / "fields.npz"),
                "--base-params",
                base,
                "--exp-name",
                f"{manifest['experiment']}_iter{index:03d}",
                "--steps",
                str(manifest.get("policy_steps", 30000)),
            ]
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/recap"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    stages = commands(manifest, args.output)
    if not args.execute:
        print(json.dumps({"status": "plan_only", "commands": stages}, indent=2))
        return
    # Validate all files and train/eval isolation before launching any stage.
    from openpi.training.recap_episode_io import load_recap_episodes
    from openpi.training.recap_learned_value import validate_episodes

    previous = set()
    heldout = None
    for iteration in manifest["iterations"]:
        train = load_recap_episodes(iteration["episodes"])
        evaluation = load_recap_episodes(iteration["eval_episodes"])
        validate_episodes(train)
        validate_episodes(evaluation)
        train_ids = {e.episode_id for e in train}
        eval_ids = {e.episode_id for e in evaluation}
        if train_ids & eval_ids or not previous <= train_ids:
            raise ValueError("Iterations must accumulate data and keep evaluation episodes separate.")
        if heldout is not None and heldout != eval_ids:
            raise ValueError("Keep a fixed evaluation set across offline iterations.")
        previous, heldout = train_ids, eval_ids
        json.loads(Path(iteration["episode_map"]).read_text())
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    for number, command in enumerate(stages):
        with (args.output / f"stage_{number:02d}.log").open("w") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    (args.output / "status.json").write_text(
        json.dumps(
            {
                "status": "offline_training_complete",
                "policy_rollout_evaluated": False,
                "base_params": manifest["base_params"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
