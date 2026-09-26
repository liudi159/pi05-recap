"""Trainable 201-bin critic on frozen VLM features (or explicit state-only ablation).

This is a resource-limited RECAP adaptation, not the paper's end-to-end value VLM.
No time index, episode length, success label, or future observation enters the critic.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from openpi.training.recap_value_proxy import (
    compute_empirical_returns,
    compute_n_step_advantages,
    compute_proxy_rewards,
)


def validate_episodes(episodes):
    if not episodes:
        raise ValueError("At least one episode is required.")
    ids = [e.episode_id for e in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError("Episode IDs must be unique across all iterations.")
    for e in episodes:
        if not e.frames or not e.task:
            raise ValueError(f"Episode {e.episode_id} must have frames and a task.")
        if e.max_episode_length is None or e.max_episode_length < len(e.frames):
            raise ValueError(f"Episode {e.episode_id} needs a valid task horizon.")
        if e.timeout and e.success:
            raise ValueError("An episode cannot be both successful and timed out.")
        if [f.t for f in e.frames] != list(range(len(e.frames))):
            raise ValueError("Frames must be contiguous and start at zero.")


def feature_matrix(episodes, tasks, feature_key):
    rows = []
    for e in episodes:
        if e.task not in tasks:
            raise ValueError(f"Unseen task: {e.task}")
        task_vector = np.eye(len(tasks), dtype=np.float32)[tasks.index(e.task)]
        for frame in e.frames:
            if feature_key not in frame.observation:
                raise ValueError(f"Missing {feature_key} in {e.episode_id}/{frame.t}; extract VLM features first.")
            feature = np.asarray(frame.observation[feature_key], dtype=np.float32)
            if feature.ndim != 1 or not len(feature) or not np.isfinite(feature).all():
                raise ValueError("Critic features must be finite, nonempty vectors.")
            rows.append(np.concatenate((feature, task_vector)))
    return np.stack(rows)


@dataclasses.dataclass
class DistributionalCritic:
    weight: np.ndarray
    bias: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    tasks: list[str]
    feature_key: str
    num_bins: int = 201

    def probabilities(self, x):
        z = ((x - self.mean) / self.std) @ self.weight + self.bias
        z -= z.max(axis=-1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=-1, keepdims=True)

    def values(self, episodes):
        x = feature_matrix(episodes, self.tasks, self.feature_key)
        return self.probabilities(x) @ np.linspace(-1, 0, self.num_bins, dtype=np.float32)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            weight=self.weight,
            bias=self.bias,
            mean=self.mean,
            std=self.std,
            tasks=np.asarray(self.tasks),
            feature_key=np.asarray(self.feature_key),
            num_bins=self.num_bins,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            return cls(
                d["weight"],
                d["bias"],
                d["mean"],
                d["std"],
                d["tasks"].tolist(),
                str(d["feature_key"]),
                int(d["num_bins"]),
            )


def fit_critic(
    train, evaluation=(), *, steps=500, batch_size=256, learning_rate=0.01, seed=0, feature_key="value_features"
):
    validate_episodes(train)
    if evaluation:
        validate_episodes(evaluation)
    if {e.episode_id for e in train} & {e.episode_id for e in evaluation}:
        raise ValueError("Value train/evaluation episodes overlap.")
    if steps < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Training steps, batch size, and learning rate must be positive.")
    tasks = sorted({e.task for e in train})
    x = feature_matrix(train, tasks, feature_key)
    returns = np.concatenate([compute_empirical_returns(compute_proxy_rewards(e)) for e in train])
    targets = np.rint((returns + 1) * 200).astype(np.int64).clip(0, 200)
    critic = DistributionalCritic(
        np.zeros((x.shape[1], 201), np.float32),
        np.zeros(201, np.float32),
        x.mean(0),
        np.maximum(x.std(0), 1e-3),
        tasks,
        feature_key,
    )
    normalized = (x - critic.mean) / critic.std
    rng = np.random.default_rng(seed)
    moments = [np.zeros_like(critic.weight), np.zeros_like(critic.bias)]
    variances = [np.zeros_like(critic.weight), np.zeros_like(critic.bias)]
    losses = []
    for step in range(1, steps + 1):
        indices = rng.integers(len(x), size=min(batch_size, len(x)))
        p = critic.probabilities(x[indices])
        losses.append(float(-np.log(np.maximum(p[np.arange(len(indices)), targets[indices]], 1e-12)).mean()))
        p[np.arange(len(indices)), targets[indices]] -= 1
        p /= len(indices)
        grads = [normalized[indices].T @ p, p.sum(0)]
        for i, (parameter, gradient) in enumerate(zip((critic.weight, critic.bias), grads, strict=True)):
            moments[i] = 0.9 * moments[i] + 0.1 * gradient
            variances[i] = 0.999 * variances[i] + 0.001 * gradient**2
            parameter -= (
                learning_rate * (moments[i] / (1 - 0.9**step)) / (np.sqrt(variances[i] / (1 - 0.999**step)) + 1e-8)
            )
    report = {
        "steps": steps,
        "initial_batch_ce": losses[0],
        "final_batch_ce": losses[-1],
        "feature_key": feature_key,
        "train_episodes": [e.episode_id for e in train],
        "eval_episodes": [e.episode_id for e in evaluation],
        "num_bins": 201,
    }
    for name, episodes in (("train", train), ("eval", evaluation)):
        if episodes:
            target = np.concatenate([compute_empirical_returns(compute_proxy_rewards(e)) for e in episodes])
            report[f"{name}_value_mae"] = float(np.abs(critic.values(episodes) - target).mean())
    return critic, report


def label_with_critic(episodes, critic, *, positive_fraction=0.4, n_step=50):
    validate_episodes(episodes)
    if not 0 < positive_fraction <= 1:
        raise ValueError("positive_fraction must be in (0, 1].")
    predicted = critic.values(episodes)
    advantages, values = [], []
    offset = 0
    for e in episodes:
        value = predicted[offset : offset + len(e.frames)]
        values.append(value)
        advantages.append(compute_n_step_advantages(compute_proxy_rewards(e), value, n_step_lookahead=n_step))
        offset += len(e.frames)
    # Paper Eq. 3 uses task-dependent thresholds and strict greater-than.
    thresholds = {}
    for task in critic.tasks:
        arrays = [a for e, a in zip(episodes, advantages, strict=True) if e.task == task]
        if arrays:
            thresholds[task] = float(np.quantile(np.concatenate(arrays), 1 - positive_fraction))
    records = []
    for e, adv, value in zip(episodes, advantages, values, strict=True):
        for f, a, v in zip(e.frames, adv, value, strict=True):
            indicator = bool(f.is_human_intervention or positive_fraction == 1 or a > thresholds[e.task])
            records.append(
                {
                    "episode_id": e.episode_id,
                    "t": f.t,
                    "task": e.task,
                    "value": float(v),
                    "advantage": float(a),
                    "advantage_indicator": indicator,
                    "use_advantage": True,
                    "is_human_intervention": bool(f.is_human_intervention),
                }
            )
    return records, thresholds


def write_keyed_labels(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
