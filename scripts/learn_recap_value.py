"""Explicitly invoked offline value training; never called by deployment."""

import argparse
import json
from pathlib import Path

from openpi.training.recap_episode_io import load_recap_episodes
from openpi.training.recap_learned_value import fit_critic, label_with_critic, write_keyed_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--eval-episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--positive-fraction", type=float, default=0.4)
    parser.add_argument("--n-step", type=int, default=50)
    parser.add_argument("--state-only", action="store_true", help="Explicit state-only ablation; no visual inputs.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    episodes = load_recap_episodes(args.episodes)
    evaluation = load_recap_episodes(args.eval_episodes)
    critic, report = fit_critic(
        episodes,
        evaluation,
        steps=args.steps,
        seed=args.seed,
        feature_key="state" if args.state_only else "value_features",
    )
    records, thresholds = label_with_critic(
        episodes, critic, positive_fraction=args.positive_fraction, n_step=args.n_step
    )
    args.output.mkdir(parents=True)
    critic.save(args.output / "critic.npz")
    write_keyed_labels(records, args.output / "labels.jsonl")
    report.update(
        task_thresholds=thresholds,
        n_step=args.n_step,
        positive_fraction_actual=sum(r["advantage_indicator"] for r in records) / len(records),
        policy_performance_evaluated=False,
    )
    (args.output / "value_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
