"""Serve a trained RECAP policy with positive advantage conditioning. Not auto-started."""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="RECAP-trained checkpoint directory, including assets.")
    parser.add_argument("--repo-id", required=True, help="Same dataset/normalization asset ID used for training.")
    parser.add_argument("--config", default="pi05_libero_recap_lora")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8016)
    args = parser.parse_args()
    import dataclasses
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config

    cfg = config.get_config(args.config)
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=args.repo_id))
    policy = policy_config.create_trained_policy(cfg, args.checkpoint)

    class PositivePolicy:
        def infer(self, obs):
            return policy.infer({**obs, "advantage_indicator": np.asarray(True), "use_advantage": np.asarray(True)})

    WebsocketPolicyServer(
        PositivePolicy(),
        host=args.host,
        port=args.port,
        metadata={**policy.metadata, "advantage_condition": "positive"},
    ).serve_forever()


if __name__ == "__main__":
    main()
