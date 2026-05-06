from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from common import PIPELINE_ROOT, load_yaml, parse_accuracy, run_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calc eval datasets.")
    parser.add_argument("--config", default=str(PIPELINE_ROOT / "stages" / "eval.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    base_args = dict(cfg.get("args", {}) or {})
    base_args["weight"] = cfg["weight"]

    results = []
    for name, data_path in (cfg.get("datasets", {}) or {}).items():
        stage = {
            "script": cfg["script"],
            "distributed": False,
            "runtime": cfg.get("runtime", {}) or {},
            "args": {
                **base_args,
                "data_path": data_path,
                "results_path": f"./results/{cfg['weight']}_{name}.jsonl",
            },
        }
        tmp_path = PIPELINE_ROOT / ".eval_stage.tmp.yaml"
        tmp_path.write_text(yaml.safe_dump(stage, sort_keys=False), encoding="utf-8")
        try:
            stdout = run_stage(tmp_path, args.dry_run)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        results.append((name, parse_accuracy(stdout)))

    if results and not args.dry_run:
        print("\n| Dataset | Accuracy |")
        print("|---|---:|")
        for name, accuracy in results:
            print(f"| {name} | {accuracy or 'N/A'} |")


if __name__ == "__main__":
    main()
