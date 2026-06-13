from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


MN = Path(r"H:\My Drive\MN_simulation")
OUT = MN / "outputs/rspi_joint_052026/fig3_ncs_robustness_audit_20260531"
PAIRED = MN / "outputs/rspi_joint_052026/fig3_paired_device_benchmark_20260531/fig3_paired_device_benchmark_manifest_method1_mmp.csv"
M2B = MN / "outputs/method2b/fig3_ncs_robustness_20260531_n512"
M2B_EXPECTED = M2B / "targeted_comsol/method2b_targeted_expected_exports.csv"
M2B_PLAN_SUMMARY = M2B / "fig3_domain_randomized_3d_plan_summary.json"
PIEZO_ALL = MN / "piezo_passive_benchmark/outputs/n7_matched_20260525/full_comsol/piezo_full_comsol_all_manifest.csv"
US_ALL = MN / "ultrasound_strain_benchmark/outputs/n7_matched_20260525/full_comsol/us_full_comsol_all_manifest.csv"
FULL512_AGG = (
    MN
    / "outputs/rspi_joint_052026/method2b_512_five_seed_ood_full512_20260611/aggregate/method2b_512_five_seed_posterior_aggregate.csv"
)
FULL512_METRICS = (
    MN
    / "outputs/rspi_joint_052026/method2b_512_five_seed_ood_full512_20260611/aggregate/method2b_512_five_seed_posterior_metrics.csv"
)
FULL512_COMMAND_PLAN = (
    MN
    / "outputs/rspi_joint_052026/method2b_512_five_seed_ood_full512_20260611/method2b_512_five_seed_ood_command_plan.json"
)
FULL512_STATS_SUMMARY = (
    MN
    / "outputs/rspi_joint_052026/fig3_posterior_paired_stats_full512_20260611/fig3_posterior_paired_stats_summary.json"
)
FULL512_STATS_CSV = (
    MN
    / "outputs/rspi_joint_052026/fig3_posterior_paired_stats_full512_20260611/fig3_posterior_paired_stats.csv"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def path_exists(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    return Path(str(value).replace("/", "\\")).exists()


def readout_exists(value: object, min_bytes: int = 1024) -> bool:
    if value is None or pd.isna(value):
        return False
    path = Path(str(value).replace("/", "\\"))
    return path.exists() and path.is_file() and path.stat().st_size >= int(min_bytes)


def completion_from_expected(expected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in expected.itertuples(index=False):
        ok = (
            path_exists(item.expected_stl)
            and path_exists(item.expected_Bz)
            and path_exists(item.expected_Bx)
            and path_exists(item.expected_By)
            and path_exists(item.expected_Bmag)
        )
        rows.append({"sample_id": str(item.sample_id), "strain": float(item.strain), "complete": ok})
    status = pd.DataFrame(rows)
    return (
        status.groupby("sample_id", as_index=False)
        .agg(expected_strain_rows=("strain", "size"), complete_strain_rows=("complete", "sum"))
        .assign(complete_condition=lambda d: d["expected_strain_rows"].eq(d["complete_strain_rows"]))
    )


def paired_gaps(paired: pd.DataFrame) -> dict:
    mmp_missing = (
        paired.loc[~paired["mmp_bz_exists"].fillna(False).astype(bool), ["condition_id", "sample_id", "family_id", "state_name", "u_load"]]
        .drop_duplicates()
        .sort_values(["condition_id", "u_load"])
    )
    piezo_missing = paired.loc[
        ~(paired["piezo_piezo_full_model_exists"].fillna(False).astype(bool) & paired["piezo_piezo_primary_readout_exists"].fillna(False).astype(bool))
    ].copy()
    us_missing = paired.loc[
        ~(paired["us_us_full_model_exists"].fillna(False).astype(bool) & paired["us_us_primary_readout_exists"].fillna(False).astype(bool))
    ].copy()
    return {
        "mmp_missing_unique": mmp_missing,
        "piezo_missing": piezo_missing,
        "us_missing": us_missing,
    }


def posterior_aggregate_summary() -> list[dict]:
    if not FULL512_AGG.exists():
        return []
    df = pd.read_csv(FULL512_AGG)
    keep_cols = [
        c
        for c in [
            "dataset",
            "split_mode",
            "model",
            "n_folds",
            "rmse_norm_mean",
            "rmse_kpa_mean",
            "coverage90_mean",
            "coverage90_kpa_mean",
            "cov90_mean",
            "width90_norm_mean",
        ]
        if c in df.columns
    ]
    if not keep_cols:
        return []
    return df[keep_cols].sort_values([c for c in ["split_mode", "rmse_kpa_mean", "rmse_norm_mean", "model"] if c in keep_cols]).to_dict(orient="records")


def write_markdown(summary: dict, md_path: Path) -> None:
    lines = [
        "# Figure 3 NCS Robustness Audit",
        "",
        "## Verdict",
        f"- Manuscript robustness gate: `{summary['ncs_gate_status']}`.",
        f"- Reason: {summary['ncs_gate_reason']}",
        "",
        "## High-fidelity domain-randomized MMP library",
        f"- Planned cases: `{summary['method2b_512']['planned_cases']}`.",
        f"- Complete cases: `{summary['method2b_512']['complete_cases']}`.",
        f"- Complete strain rows: `{summary['method2b_512']['complete_strain_rows']}` / `{summary['method2b_512']['expected_strain_rows']}`.",
        f"- Observed smoke-case COMSOL time: `{summary['method2b_512']['observed_seconds_per_case']}` s/case; estimated remaining wall time: `{summary['method2b_512']['estimated_remaining_hours']:.1f}` h if run serially.",
        "",
        "## Paired device benchmark",
        f"- Paired rows: `{summary['paired_device']['rows']}` across `{summary['paired_device']['conditions']}` conditions.",
        f"- MMP full-field rows complete: `{summary['paired_device']['mmp_complete_rows']}` / `{summary['paired_device']['rows']}`.",
        f"- Piezo full-COMSOL rows complete: `{summary['paired_device']['piezo_complete_rows']}` / `{summary['paired_device']['rows']}`.",
        f"- US full-COMSOL rows complete: `{summary['paired_device']['us_complete_rows']}` / `{summary['paired_device']['rows']}`.",
        f"- All three complete on same paired row: `{summary['paired_device']['all_three_complete_rows']}` / `{summary['paired_device']['rows']}`.",
        "",
        "## Missing queues",
        f"- MMP exact readout queue: `{summary['outputs']['mmp_missing_queue']}`.",
        f"- Piezo full-COMSOL queue: `{summary['outputs']['piezo_missing_queue']}`.",
        f"- Piezo runner-ready manifest: `{summary['outputs']['piezo_runner_manifest']}`.",
        f"- US full-COMSOL queue: `{summary['outputs']['us_missing_queue']}`.",
        f"- US runner-ready manifest: `{summary['outputs']['us_runner_manifest']}`.",
        f"- Method2b 512 completion queue: `{summary['outputs']['method2b_queue']}`.",
        "",
        "## Colab posterior aggregate",
        f"- Required aggregate: `{summary['posterior']['aggregate_csv']}`.",
        f"- Required metrics: `{summary['posterior']['metrics_csv']}`.",
        f"- Existing aggregate rows: `{summary['posterior']['aggregate_rows']}`.",
        f"- Command plan: `{summary['posterior']['command_plan_json']}`.",
        f"- Paired statistics: `{summary['statistics']['stats_csv']}`.",
        "",
        "## Reviewer gate",
        "- Do not claim Nature Computational Science-level robustness until the full512 five-seed/OOD posterior aggregate exists and paired bootstrap/statistical tests are recomputed from that final aggregate.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(PAIRED)
    gaps = paired_gaps(paired)

    gaps["mmp_missing_unique"].to_csv(OUT / "fig3_missing_method1_mmp_readout_queue.csv", index=False)
    gaps["piezo_missing"].to_csv(OUT / "fig3_missing_piezo_full_comsol_queue.csv", index=False)
    gaps["us_missing"].to_csv(OUT / "fig3_missing_us_full_comsol_queue.csv", index=False)
    piezo_all = pd.read_csv(PIEZO_ALL)
    piezo_runner_missing = piezo_all.loc[
        ~(piezo_all["full_comsol_model_out"].map(lambda p: readout_exists(p, 1024)) & piezo_all["full_comsol_Voc_csv"].map(readout_exists))
    ].copy()
    piezo_runner_missing.to_csv(OUT / "fig3_missing_piezo_runner_manifest.csv", index=False)
    us_all = pd.read_csv(US_ALL)
    us_runner_missing = us_all.loc[
        ~(us_all["full_comsol_model_out"].map(lambda p: readout_exists(p, 1024)) & us_all["full_comsol_us_array_csv"].map(readout_exists))
    ].copy()
    us_runner_missing.to_csv(OUT / "fig3_missing_us_runner_manifest.csv", index=False)

    expected = pd.read_csv(M2B_EXPECTED)
    method2b_status = completion_from_expected(expected)
    method2b_status.to_csv(OUT / "fig3_method2b_512_completion_queue.csv", index=False)
    planned_summary = read_json(M2B_PLAN_SUMMARY)
    complete_cases = int(method2b_status["complete_condition"].sum())
    expected_rows = int(len(expected))
    complete_rows = int(method2b_status["complete_strain_rows"].sum())
    observed_seconds_per_case = 1546 + 197
    remaining_cases = int(len(method2b_status) - complete_cases)

    piezo_complete = paired["piezo_piezo_full_model_exists"].fillna(False).astype(bool) & paired["piezo_piezo_primary_readout_exists"].fillna(False).astype(bool)
    us_complete = paired["us_us_full_model_exists"].fillna(False).astype(bool) & paired["us_us_primary_readout_exists"].fillna(False).astype(bool)
    mmp_complete = paired["mmp_bz_exists"].fillna(False).astype(bool)
    all_complete = mmp_complete & piezo_complete & us_complete
    posterior_rows = posterior_aggregate_summary()
    metrics_ready = FULL512_METRICS.exists()
    if metrics_ready:
        metrics = pd.read_csv(FULL512_METRICS)
        seed_count = int(metrics["seed"].nunique()) if "seed" in metrics.columns else 0
    else:
        seed_count = 0
    stats_ready = False
    stats_comparisons = 0
    if FULL512_STATS_SUMMARY.exists() and FULL512_STATS_CSV.exists():
        stats_summary = read_json(FULL512_STATS_SUMMARY)
        stats_comparisons = int(stats_summary.get("n_comparisons", 0) or 0)
        stats_ready = stats_comparisons > 0

    gate_pass = (
        complete_cases >= 512
        and int(all_complete.sum()) >= 120
        and len(posterior_rows) > 0
        and seed_count >= 5
        and stats_ready
    )
    reason = (
        "blocked until full512 five-seed/OOD posterior aggregate and paired statistics are available"
        if not gate_pass
        else "minimum high-fidelity, paired benchmark and posterior aggregate gates passed"
    )
    summary = {
        "ncs_gate_status": "PASS" if gate_pass else "NOT_YET",
        "ncs_gate_reason": reason,
        "method2b_512": {
            "planned_cases": int(planned_summary.get("n_cases", len(method2b_status))),
            "complete_cases": complete_cases,
            "expected_strain_rows": expected_rows,
            "complete_strain_rows": complete_rows,
            "remaining_cases": remaining_cases,
            "observed_seconds_per_case": observed_seconds_per_case,
            "estimated_remaining_hours": remaining_cases * observed_seconds_per_case / 3600.0,
            "plan_summary": str(M2B_PLAN_SUMMARY),
        },
        "paired_device": {
            "manifest": str(PAIRED),
            "rows": int(len(paired)),
            "conditions": int(paired["condition_id"].nunique()),
            "mmp_complete_rows": int(mmp_complete.sum()),
            "piezo_complete_rows": int(piezo_complete.sum()),
            "us_complete_rows": int(us_complete.sum()),
            "all_three_complete_rows": int(all_complete.sum()),
            "mmp_missing_unique_condition_loads": int(len(gaps["mmp_missing_unique"])),
            "piezo_missing_rows": int(len(gaps["piezo_missing"])),
            "us_missing_rows": int(len(gaps["us_missing"])),
        },
        "posterior": {
            "aggregate_csv": str(FULL512_AGG),
            "metrics_csv": str(FULL512_METRICS),
            "command_plan_json": str(FULL512_COMMAND_PLAN),
            "aggregate_rows": int(len(posterior_rows)),
            "metrics_ready": bool(metrics_ready),
            "seed_count": int(seed_count),
            "aggregate_preview": posterior_rows[:20],
        },
        "statistics": {
            "stats_csv": str(FULL512_STATS_CSV),
            "summary_json": str(FULL512_STATS_SUMMARY),
            "stats_ready": bool(stats_ready),
            "n_comparisons": int(stats_comparisons),
        },
        "outputs": {
            "mmp_missing_queue": str(OUT / "fig3_missing_method1_mmp_readout_queue.csv"),
            "piezo_missing_queue": str(OUT / "fig3_missing_piezo_full_comsol_queue.csv"),
            "us_missing_queue": str(OUT / "fig3_missing_us_full_comsol_queue.csv"),
            "piezo_runner_manifest": str(OUT / "fig3_missing_piezo_runner_manifest.csv"),
            "us_runner_manifest": str(OUT / "fig3_missing_us_runner_manifest.csv"),
            "method2b_queue": str(OUT / "fig3_method2b_512_completion_queue.csv"),
            "summary_json": str(OUT / "fig3_ncs_robustness_audit_summary.json"),
            "report_md": str(OUT / "Figure3_NCS_robustness_audit.md"),
        },
    }
    (OUT / "fig3_ncs_robustness_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, OUT / "Figure3_NCS_robustness_audit.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
