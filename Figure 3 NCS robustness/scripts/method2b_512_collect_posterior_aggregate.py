from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


MN_ROOT = Path(r"H:\My Drive\MN_simulation")
MN_SCRIPTS = MN_ROOT / "scripts"
if str(MN_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MN_SCRIPTS))

from rspi_03_eval_posterior_metrics import aggregate_posterior  # noqa: E402


DEFAULT_RUN_ROOT = MN_ROOT / "outputs" / "rspi_joint_052026" / "method2b_512_five_seed_ood_20260601"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)
        handle.write("\n")


def infer_seed(path: Path) -> int | None:
    for part in path.parts[::-1]:
        match = re.match(r"seed_(\d+)$", part)
        if match:
            return int(match.group(1))
    return None


def collect(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else run_root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_paths = sorted(run_root.glob(args.posterior_glob))
    if not metrics_paths:
        raise RuntimeError(f"No posterior metrics files matched {args.posterior_glob} under {run_root}")

    frames: list[pd.DataFrame] = []
    for path in metrics_paths:
        frame = pd.read_csv(path)
        seed = infer_seed(path)
        if seed is not None and "seed" not in frame.columns:
            frame.insert(0, "seed", seed)
        frame.insert(0, "source_metrics_csv", str(path))
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    aggregate = aggregate_posterior(metrics)

    metrics_path = out_dir / "method2b_512_five_seed_posterior_metrics.csv"
    aggregate_path = out_dir / "method2b_512_five_seed_posterior_aggregate.csv"
    metrics.to_csv(metrics_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)

    best = aggregate.sort_values(["dataset", "split_mode", "rmse_norm_mean"]).groupby(["dataset", "split_mode"]).head(1).to_dict("records")
    summary = {
        "step": "method2b_512_collect_posterior_aggregate",
        "decision": "METHOD2B_512_FIVE_SEED_POSTERIOR_AGGREGATED",
        "run_root": str(run_root),
        "metrics_file_count": int(len(metrics_paths)),
        "row_count": int(len(metrics)),
        "seeds": sorted(int(x) for x in metrics["seed"].dropna().unique()) if "seed" in metrics.columns else [],
        "split_modes": sorted(str(x) for x in metrics["split_mode"].dropna().unique()) if "split_mode" in metrics.columns else [],
        "models": sorted(str(x) for x in metrics["model"].dropna().unique()) if "model" in metrics.columns else [],
        "best_by_split": best,
        "outputs": {
            "metrics_csv": str(metrics_path),
            "aggregate_csv": str(aggregate_path),
            "summary_json": str(out_dir / "method2b_512_five_seed_posterior_aggregate_summary.json"),
        },
    }
    write_json(out_dir / "method2b_512_five_seed_posterior_aggregate_summary.json", summary)
    lines = [
        "# Method2b 512 Five-Seed Posterior Aggregate",
        "",
        f"- Metrics files: `{len(metrics_paths)}`",
        f"- Rows: `{len(metrics)}`",
        f"- Seeds: `{', '.join(map(str, summary['seeds']))}`",
        f"- Aggregate: `{aggregate_path}`",
        "",
        "## Best By Split",
    ]
    for row in best:
        lines.append(
            f"- `{row['split_mode']}` best `{row['model']}`: rmse_norm `{row.get('rmse_norm_mean')}`, "
            f"cov90 `{row.get('cov90_mean')}`, width90 `{row.get('width90_norm_mean')}`"
        )
    (out_dir / "method2b_512_five_seed_posterior_aggregate_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect Method2b 512 five-seed OOD posterior metrics.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--posterior-glob", default="seed_*/posterior/rspi_posterior_metrics.csv")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
