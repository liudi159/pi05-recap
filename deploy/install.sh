#!/usr/bin/env bash
# Dependency installation and static verification only. Never starts ML jobs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "Install uv before running this script." >&2; exit 1; }
export GIT_LFS_SKIP_SMUDGE=1
# LeRobot source checkout does not need LFS dataset assets or a system git-lfs binary.
export GIT_CONFIG_COUNT=4
export GIT_CONFIG_KEY_0=filter.lfs.process GIT_CONFIG_VALUE_0=""
export GIT_CONFIG_KEY_1=filter.lfs.smudge GIT_CONFIG_VALUE_1=cat
export GIT_CONFIG_KEY_2=filter.lfs.clean GIT_CONFIG_VALUE_2=cat
export GIT_CONFIG_KEY_3=filter.lfs.required GIT_CONFIG_VALUE_3=false
if [[ -d .cache/git/lerobot.git ]]; then
  export GIT_CONFIG_COUNT=5
  export GIT_CONFIG_KEY_4="url.file://$PWD/.cache/git/lerobot.git.insteadOf"
  export GIT_CONFIG_VALUE_4="https://github.com/huggingface/lerobot"
fi
# Optional offline wheel fallback. The exact upstream lock hash is mandatory.
RECAP_TORCH_WHEEL=".cache/wheels/torch-2.7.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl"
if [[ -f "$RECAP_TORCH_WHEEL" ]]; then
  [[ -x .venv/bin/python ]] || uv venv --python 3.11 .venv
  .venv/bin/python - "$RECAP_TORCH_WHEEL" <<'VERIFY'
import hashlib
from pathlib import Path
import sys
import tomllib
from urllib.parse import unquote, urlparse
wheel = Path(sys.argv[1])
lock = tomllib.loads(Path("uv.lock").read_text())
package = next(p for p in lock["package"] if p["name"] == "torch")
record = next(w for w in package["wheels"] if Path(unquote(urlparse(w["url"]).path)).name == wheel.name)
with wheel.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
if record["hash"] != "sha256:" + digest:
    raise SystemExit("Cached torch wheel does not match uv.lock")
print("Cached PyTorch wheel SHA256 matches upstream lock.")
VERIFY
  uv sync --frozen --no-install-package torch
  uv pip install --python .venv/bin/python --no-deps --no-index "$RECAP_TORCH_WHEEL"
else
  uv sync --frozen
fi
uv pip check --python .venv/bin/python
.venv/bin/python - <<'PY'
import ast
import importlib.metadata as metadata
from pathlib import Path
for directory in ("src", "scripts", "packages"):
    for path in Path(directory).rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
for name in ("openpi", "jax", "jaxlib", "torch", "torchvision", "flax", "av", "lerobot"):
    print(f"{name}=={metadata.version(name)}")
print("Installation and static syntax verification complete. No model was loaded or run.")
PY
