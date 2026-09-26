"""Regression tests supplied for later execution; deployment does not run them."""

import numpy as np
import pytest

from openpi.training.recap_episode_io import ReCAPFrame, ReCAPOfflineEpisode
from openpi.training.recap_learned_value import DistributionalCritic, fit_critic, label_with_critic
from openpi.training.recap_value_proxy import compute_n_step_advantages
from scripts.align_recap_labels import align


def episode(identifier, task="pick", success=True):
    return ReCAPOfflineEpisode(
        identifier,
        task,
        success,
        [ReCAPFrame(i, {"state": np.array([i / 2], np.float32)}, np.zeros(7)) for i in range(3)],
        max_episode_length=10,
    )


def test_failed_terminal_has_zero_bootstrap():
    result = compute_n_step_advantages(np.array([-0.1, -1]), np.array([-0.5, -0.7]), n_step_lookahead=1)
    np.testing.assert_allclose(result, [-0.3, -0.3], atol=1e-6)


def test_alignment_reorders_and_rejects_missing_frames():
    rows = [
        dict(episode_id="ep", t=i, advantage_indicator=bool(i), use_advantage=True, is_human_intervention=False)
        for i in range(2)
    ]
    data = align(rows, {"ep": 5}, [5, 5], [1, 0])
    assert data["advantage_indicator"].tolist() == [True, False]
    with pytest.raises(ValueError, match="differ"):
        align(rows, {"ep": 5}, [5], [0])


def test_critic_never_uses_episode_outcome_as_input_and_roundtrips(tmp_path):
    train = [episode("a"), episode("b", success=False)]
    critic, _ = fit_critic(train, steps=20, feature_key="state")
    # Both episodes have identical observations and task despite different outcomes.
    np.testing.assert_allclose(critic.values(train)[:3], critic.values(train)[3:])
    target = tmp_path / "critic.npz"
    critic.save(target)
    np.testing.assert_allclose(DistributionalCritic.load(target).values(train), critic.values(train))


def test_critic_rejects_train_eval_leakage():
    with pytest.raises(ValueError, match="overlap"):
        fit_critic([episode("same")], [episode("same")], steps=1, feature_key="state")


def test_task_thresholds_and_human_override():
    first, second = episode("a", "pick"), episode("b", "place")
    first.frames[0] = ReCAPFrame(0, {"state": np.array([0.0])}, np.zeros(7), True)
    critic, _ = fit_critic([first, second], steps=1, feature_key="state")
    rows, thresholds = label_with_critic([first, second], critic, n_step=1)
    assert set(thresholds) == {"pick", "place"}
    assert rows[0]["advantage_indicator"] is True
