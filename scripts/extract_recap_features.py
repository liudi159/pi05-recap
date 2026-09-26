"""Cache frozen pi0.5 image/state/language features for the distributional critic.

JSON observations contain state, image_path, wrist_image_path. Image paths are
relative to --image-root. Output retains those paths and adds value_features.
Run separately from policy training so the VLM does not remain resident in GPU RAM.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.policies import policy_config
from openpi.training import config
from openpi.training.recap_episode_io import load_recap_episodes, save_recap_episodes
from openpi.training.recap_learned_value import validate_episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True, help="Frozen pi05_libero checkpoint, including assets.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    episodes = load_recap_episodes(args.episodes)
    validate_episodes(episodes)
    policy = policy_config.create_trained_policy(config.get_config("pi05_libero"), args.checkpoint)
    if policy._is_pytorch_model:
        raise ValueError("Feature extraction requires the JAX/Orbax pi05_libero checkpoint.")

    def encode(model, observation):
        tokens, mask, ar = model.embed_prefix(observation)
        (features, _), _cache = model.PaliGemma.llm(
            [tokens, None],
            mask=make_attn_mask(mask, ar),
            positions=jnp.cumsum(mask, axis=1) - 1,
        )
        return jnp.sum(features.astype(jnp.float32) * mask[..., None], axis=1) / jnp.maximum(
            mask.sum(1, keepdims=True), 1
        )

    # Compile a frozen inference pass; no value-backbone optimization is performed.
    import flax.nnx as nnx

    encode_jit = nnx.jit(encode)
    for episode in episodes:
        for frame in episode.frames:
            obs = frame.observation

            def read_image(key):
                with Image.open(args.image_root / obs[key]) as im:
                    return np.asarray(im.convert("RGB"))

            inputs = {
                "observation/state": np.asarray(obs["state"], np.float32),
                "observation/image": read_image("image_path"),
                "observation/wrist_image": read_image("wrist_image_path"),
                "prompt": episode.task,
            }
            transformed = policy._input_transform(inputs)
            batched = jax.tree.map(lambda x: jnp.asarray(x)[None], transformed)
            value = encode_jit(policy._model, model_lib.Observation.from_dict(batched))
            obs["value_features"] = np.asarray(value[0], dtype=np.float32)
    save_recap_episodes(episodes, args.output)


if __name__ == "__main__":
    main()
