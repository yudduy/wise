"""Provisioning helper for the live WISE reproduction environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .mrsteve import DEFAULT_TASK, mrsteve_smoke, stats_command, steve1_smoke


ENV_TEMPLATE = """# Fill these in after cloning MrSteve and running prepare_models.sh.
WISE_MRSTEVE_ROOT={root}
WISE_MINECLIP_CHECKPOINT={root}/downloads/weights/mineclip/attn.pth
WISE_STEVE1_WEIGHTS={root}/downloads/weights/steve1/steve1.weights
WISE_VPT_MODEL={root}/downloads/weights/vpt/2x.model
WISE_VPT_NAV_CHECKPOINT={root}/downloads/weights/vpt/vpt_nav.weights
WISE_OPENAI_MODEL=gpt-4o
# OPENAI_API_KEY=...
"""


def plan(root: str) -> dict[str, object]:
    return {
        "clone": f"git clone https://github.com/frechele/MrSteve {root}",
        "setup": [
            f"cd {root}",
            "uv sync",
            "uv run -m pip install git+https://github.com/MineDojo/MineCLIP",
            "uv run -m pip install gym==0.21.0",
            "uv run bash prepare_models.sh",
        ],
        "smoke": [
            steve1_smoke(root, task=DEFAULT_TASK).display(),
            mrsteve_smoke(root, task=DEFAULT_TASK).display(),
            stats_command(root, f"outputs/{DEFAULT_TASK}/mrsteve/*").display(),
        ],
        "readiness": "uv run python -m wise.readiness",
    }


def write_env(path: Path, root: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(ENV_TEMPLATE.format(root=root), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mrsteve-root", default="../MrSteve")
    parser.add_argument("--env-file", default=".env.wise")
    parser.add_argument("--write-env", action="store_true")
    args = parser.parse_args(argv)

    if args.write_env:
        write_env(Path(args.env_file), args.mrsteve_root)
    print(json.dumps(plan(args.mrsteve_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
