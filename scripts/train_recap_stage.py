"""Train one offline RECAP policy stage from a fixed base checkpoint."""

import argparse
import dataclasses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_libero_recap_lora")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--fields", required=True)
    parser.add_argument("--base-params", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--steps", type=int, default=30000)
    args = parser.parse_args()
    from openpi.training import config, weight_loaders
    import train

    cfg = config.get_config(args.config)
    cfg = dataclasses.replace(
        cfg,
        data=dataclasses.replace(cfg.data, repo_id=args.repo_id, recap_fields_path=args.fields),
        weight_loader=weight_loaders.CheckpointWeightLoader(args.base_params),
        exp_name=args.exp_name,
        num_train_steps=args.steps,
        overwrite=False,
        resume=False,
        wandb_enabled=False,
    )
    train.main(cfg)


if __name__ == "__main__":
    main()
