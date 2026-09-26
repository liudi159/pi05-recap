# Sources and adaptation

- Physical Intelligence `openpi`: https://github.com/Physical-Intelligence/openpi
  Inspected official revision: `215abfb217dbac7d5f1273282331b9b1866c0479`.
- Source base for this repository: https://github.com/hzm8341/pi0.6
  Revision: `4337f1e9abf6f079e9ed41150902b6cd0c33745a` (Apache-2.0).
  Model/training code was copied from this reference, excluding its dataset, Git metadata,
  editor settings and GitHub workflows. It contains pi0.5 and experimental MEM/RECAP changes.
- Algorithm reference: https://arxiv.org/abs/2511.14759
  Paper details were checked against v1: https://arxiv.org/html/2511.14759v1.
- LeRobot source revision is fixed in `pyproject.toml` and `uv.lock`.

Local adaptation (2026-09-26):

1. Linux x86_64 / Python 3.11 dependency lock; PyAV 14.2 binary wheel; PyTorch 2.7.1 CUDA 12.8.
2. LIBERO RECAP and LoRA configs; preserve labels through repacking and LIBERO observation transforms.
3. RECAP loss uses one flow-matching pass with indicator dropout (paper V-B), honors input masks,
   forces human correction labels positive, and rejects silently missing conditioning.
4. Zero bootstrap after episode termination; keyed sidecars validate episode/frame correspondence.
5. An experimental learned 201-bin linear classifier over frozen VLM features, with explicit
   state-only ablation, task-dependent thresholds, checkpoint IO and separate evaluation episodes.
6. Frozen pi0.5 feature extraction; explicit offline-stage subprocess orchestration from a fixed base;
   positive-conditioned serving entry point; deployment scripts do not start any job.

These changes retain the pi0.5 backbone. The frozen-encoder critic is an approximation, not
the paper's separately trained smaller VLM. No Gemma 3 4B/860M action expert, KI joint
pretraining, real-robot HITL collector or paper performance claims are provided.

The source modification list is not a statement of runtime validation. Deployment-only status
and validation limits are recorded in `DEPLOYMENT.md`.
