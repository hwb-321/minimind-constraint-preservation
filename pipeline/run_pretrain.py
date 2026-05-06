from __future__ import annotations

import argparse
from pathlib import Path

from common import PIPELINE_ROOT, run_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pretraining stage.")
    parser.add_argument("--config", default=str(PIPELINE_ROOT / "stages" / "pretrain.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_stage(Path(args.config), args.dry_run)


if __name__ == "__main__":
    main()
