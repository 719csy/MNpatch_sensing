from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


WORKSPACE = Path(r"H:\My Drive\Manuscript\Sparse reconstruction")
MN = Path(r"H:\My Drive\MN_simulation")
DEFAULT_RUN_ROOT = (
    MN
    / "outputs"
    / "rspi_joint_052026"
    / "method2b_512_five_seed_ood_full512_20260611"
)
DEFAULT_STATS_DIR = (
    MN
    / "outputs"
    / "rspi_joint_052026"
    / "fig3_posterior_paired_stats_full512_20260611"
)
REPORT_JSON = WORKSPACE / "reports" / "figure3_full512_finalization_status.json"
REPORT_MD = WORKSPACE / "reports" / "figure3_full512_finalization_status.md"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def run(cmd: list[str], *, timeout: int | None = None) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(WORKSPACE), check=True, timeout=timeout)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# Figure 3 Full512 Finalization Status",
        "",
        f"- Status: `{payload['status']}`.",
        f"- Reason: {payload['reason']}",
        f"- Updated: `{payload['updated_at']}`",
        "",
        "## Required Outputs",
        f"- Metrics: `{payload['paths']['metrics_csv']}`",
        f"- Aggregate: `{payload['paths']['aggregate_csv']}`",
        f"- Paired stats: `{payload['paths']['paired_stats_csv']}`",
        f"- Strict audit: `{payload['paths']['audit_summary_json']}`",
        "",
        "## Current Counts",
        f"- Seed metrics files: `{payload['seed_metrics_file_count']}`",
        f"- Audit gate: `{payload.get('audit_gate', 'UNKNOWN')}`",
    ]
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Finalize Figure 3 after full512 five-seed/OOD posterior outputs arrive."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--render-on-pass", action="store_true", default=True)
    parser.add_argument("--no-render-on-pass", dest="render_on_pass", action="store_false")
    args = parser.parse_args()

    run_root = args.run_root
    aggregate_dir = run_root / "aggregate"
    metrics_csv = aggregate_dir / "method2b_512_five_seed_posterior_metrics.csv"
    aggregate_csv = aggregate_dir / "method2b_512_five_seed_posterior_aggregate.csv"
    stats_csv = args.stats_dir / "fig3_posterior_paired_stats.csv"
    audit_json = (
        MN
        / "outputs"
        / "rspi_joint_052026"
        / "fig3_ncs_robustness_audit_20260531"
        / "fig3_ncs_robustness_audit_summary.json"
    )
    seed_metrics = sorted(run_root.glob("seed_*/posterior/rspi_posterior_metrics.csv"))

    if not metrics_csv.exists() or not aggregate_csv.exists():
        if len(seed_metrics) < 5:
            payload = {
                "status": "BLOCKED_WAITING_FOR_COLAB",
                "reason": "full512 seed posterior metrics are not complete yet",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "seed_metrics_file_count": len(seed_metrics),
                "paths": {
                    "run_root": str(run_root),
                    "metrics_csv": str(metrics_csv),
                    "aggregate_csv": str(aggregate_csv),
                    "paired_stats_csv": str(stats_csv),
                    "audit_summary_json": str(audit_json),
                },
            }
            write_json(REPORT_JSON, payload)
            write_report(payload)
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 2
        run(
            [
                sys.executable,
                str(WORKSPACE / "scripts" / "method2b_512_collect_posterior_aggregate.py"),
                "--run-root",
                str(run_root),
            ],
            timeout=600,
        )

    run(
        [
            sys.executable,
            str(WORKSPACE / "scripts" / "fig3_posterior_paired_stats.py"),
            "--metrics",
            str(metrics_csv),
            "--out-dir",
            str(args.stats_dir),
            "--bootstrap",
            "10000",
        ],
        timeout=900,
    )
    run(
        [sys.executable, str(WORKSPACE / "scripts" / "fig3_ncs_robustness_audit.py")],
        timeout=300,
    )
    audit = read_json(audit_json)
    gate = str(audit.get("ncs_gate_status", "UNKNOWN"))
    if gate == "PASS" and args.render_on_pass:
        run(
            [sys.executable, str(WORKSPACE / "scripts" / "fig3_render_realdata_dashboard.py")],
            timeout=900,
        )
        run(
            [sys.executable, str(WORKSPACE / "scripts" / "fig3_build_hybrid_realdata_on_v3.py")],
            timeout=900,
        )

    payload = {
        "status": "PASS" if gate == "PASS" else "NOT_YET",
        "reason": audit.get("ncs_gate_reason", "audit did not provide a reason"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "seed_metrics_file_count": len(seed_metrics),
        "audit_gate": gate,
        "paths": {
            "run_root": str(run_root),
            "metrics_csv": str(metrics_csv),
            "aggregate_csv": str(aggregate_csv),
            "paired_stats_csv": str(stats_csv),
            "audit_summary_json": str(audit_json),
            "final_png": str(WORKSPACE / "figures" / "figure3" / "figure3_final_locked_overlay.png"),
            "final_pdf": str(WORKSPACE / "figures" / "figure3" / "figure3_final_locked_overlay.pdf"),
        },
    }
    write_json(REPORT_JSON, payload)
    write_report(payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if gate == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
