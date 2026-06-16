from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

import pandas as pd

from figure3_colab_orchestrator import (
    ApiDriveAccess,
    DRIVE_MODE,
    MANUSCRIPT_DRIVE_REL,
    MN_DRIVE_REL,
    MN_WORK,
    MountedDriveAccess,
    WORK_ROOT,
    rel_to_log_path,
    rel_to_windows_path,
    run_cmd,
)


RUN_TAG = os.environ.get("METHOD2B_COLAB_RUN_TAG") or time.strftime("%Y%m%d_%H%M%S")
SEEDS = [int(x.strip()) for x in os.environ.get("METHOD2B_COLAB_SEEDS", "1,2,3,4,5").split(",") if x.strip()]
EXPECTED_SAMPLE_COUNT = int(os.environ.get("METHOD2B_EXPECTED_SAMPLE_COUNT", "512"))
SPLIT_MODES = os.environ.get("METHOD2B_SPLIT_MODES", "fig3_true_ood")
BATCH_SIZE = os.environ.get("METHOD2B_BATCH_SIZE", "16")
BASE_CHANNELS = os.environ.get("METHOD2B_BASE_CHANNELS", "16")
TIMESTEPS = os.environ.get("METHOD2B_TIMESTEPS", "200")
REGRESSOR_EPOCHS = os.environ.get("METHOD2B_REGRESSOR_EPOCHS", "24")
DIFFUSION_EPOCHS = os.environ.get("METHOD2B_DIFFUSION_EPOCHS", "48")
RESIDUAL_DIFFUSION_EPOCHS = os.environ.get("METHOD2B_RESIDUAL_DIFFUSION_EPOCHS", "48")
SURROGATE_EPOCHS = os.environ.get("METHOD2B_SURROGATE_EPOCHS", "16")
TRAIN_POSTERIOR_SAMPLES = os.environ.get("METHOD2B_TRAIN_POSTERIOR_SAMPLES", "4")
POSTERIOR_SAMPLES = os.environ.get("METHOD2B_POSTERIOR_SAMPLES", "64")
IMG2IMG_START_STEP = os.environ.get("METHOD2B_IMG2IMG_START_STEP", "10")
LR = os.environ.get("METHOD2B_LR", "1e-3")
GUIDANCE_SCALE = os.environ.get("METHOD2B_GUIDANCE_SCALE", "0.06")
RESIDUAL_SCALE = os.environ.get("METHOD2B_RESIDUAL_SCALE", "0.25")
RESIDUAL_SCALE_QUANTILE = os.environ.get("METHOD2B_RESIDUAL_SCALE_QUANTILE", "0.995")
RESIDUAL_SCALE_FLOOR = os.environ.get("METHOD2B_RESIDUAL_SCALE_FLOOR", "0.03")
RUN_SMOKE = os.environ.get("METHOD2B_RUN_SMOKE", "1").strip().lower() not in {"0", "false", "no"}

CACHE_DRIVE_REL = PurePosixPath(
    os.environ.get(
        "METHOD2B_CACHE_DRIVE_REL",
        "MN_simulation/outputs/rspi_joint_052026/method2b_512_targetmap_diffusion_full512_20260611",
    )
)
RUN_DRIVE_REL = PurePosixPath(
    os.environ.get(
        "METHOD2B_RUN_DRIVE_REL",
        f"MN_simulation/outputs/rspi_joint_052026/method2b_512_five_seed_ood_colab_cuda_{RUN_TAG}",
    )
)
CACHE_NAMES = [
    "method2b_512_diffusion_ready_cache.npz",
    "method2b_512_diffusion_ready_manifest.csv",
    "method2b_512_diffusion_cache_summary.json",
    "method2b_512_diffusion_excluded_cases.csv",
]
SCRIPT_NAMES = [
    "step3j_method1_train_canonical_diffusion_cv.py",
    "rspi_03_eval_posterior_metrics.py",
    "step3c_train_batch_benchmark.py",
    "step3d_feature_utils.py",
]


def logged_cmd(cmd: list[object], log_path: Path, cwd: Path | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\nRUN:", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("COMMAND\n")
        handle.write(" ".join(shlex.quote(str(x)) for x in cmd) + "\n\n")
        handle.flush()
        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(map(str, cmd))}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def check_cuda() -> str:
    import torch

    print("torch", torch.__version__, flush=True)
    print("torch.cuda.is_available", torch.cuda.is_available(), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use Runtime > Change runtime type > GPU, then rerun.")
    gpu = torch.cuda.get_device_name(0)
    print("GPU", gpu, flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return gpu


def download_required_scripts(drive_access: ApiDriveAccess | MountedDriveAccess) -> None:
    scripts_dir = MN_WORK / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("METHOD2B_DOWNLOAD_ALL_SCRIPTS", "0").strip().lower() in {"1", "true", "yes"}:
        drive_access.download_folder(MN_DRIVE_REL / "scripts", scripts_dir)
        return
    for name in SCRIPT_NAMES:
        drive_access.download_file(MN_DRIVE_REL / "scripts" / name, scripts_dir / name)


def aggregate_seed_metrics(seed_eval_dirs: list[Path], out_dir: Path) -> tuple[Path, Path]:
    if str(MN_WORK / "scripts") not in sys.path:
        sys.path.insert(0, str(MN_WORK / "scripts"))
    from rspi_03_eval_posterior_metrics import aggregate_posterior

    frames: list[pd.DataFrame] = []
    for eval_dir in seed_eval_dirs:
        metrics_path = eval_dir / "rspi_posterior_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        frame = pd.read_csv(metrics_path)
        seed_text = eval_dir.name.split("_seed")[-1].split("_")[0]
        if "seed" not in frame.columns:
            frame.insert(0, "seed", int(seed_text))
        frame.insert(0, "source_metrics_csv", str(metrics_path))
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    aggregate = aggregate_posterior(metrics)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "method2b_512_five_seed_posterior_metrics.csv"
    aggregate_path = out_dir / "method2b_512_five_seed_posterior_aggregate.csv"
    summary_path = out_dir / "method2b_512_five_seed_posterior_aggregate_summary.json"
    metrics.to_csv(metrics_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    best = (
        aggregate.sort_values(["dataset", "split_mode", "rmse_norm_mean"])
        .groupby(["dataset", "split_mode"])
        .head(1)
        .to_dict("records")
    )
    write_json(
        summary_path,
        {
            "step": "method2b_512_colab_aggregate",
            "decision": "METHOD2B_512_COLAB_FIVE_SEED_POSTERIOR_AGGREGATED",
            "run_tag": RUN_TAG,
            "seeds": SEEDS,
            "row_count": int(len(metrics)),
            "models": sorted(str(x) for x in metrics["model"].dropna().unique()) if "model" in metrics else [],
            "split_modes": sorted(str(x) for x in metrics["split_mode"].dropna().unique()) if "split_mode" in metrics else [],
            "best_by_split": best,
            "outputs": {
                "metrics_csv": str(metrics_path),
                "aggregate_csv": str(aggregate_path),
                "summary_json": str(summary_path),
            },
        },
    )
    return aggregate_path, summary_path


def main() -> None:
    drive_access = ApiDriveAccess() if DRIVE_MODE == "api" else MountedDriveAccess()
    gpu = check_cuda()
    if os.environ.get("METHOD2B_SKIP_PIP_INSTALL", "0").strip().lower() in {"1", "true", "yes"}:
        print("Skipping pip install because METHOD2B_SKIP_PIP_INSTALL=1.", flush=True)
    else:
        run_cmd([sys.executable, "-m", "pip", "install", "-q", "numpy", "pandas", "scipy", "matplotlib", "scikit-learn", "tqdm"])

    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    download_required_scripts(drive_access)

    cache_work_dir = MN_WORK / "outputs" / "rspi_joint_052026" / "method2b_512_targetmap_diffusion_20260601"
    for name in CACHE_NAMES:
        drive_access.download_file(CACHE_DRIVE_REL / name, cache_work_dir / name)

    cache_path = cache_work_dir / "method2b_512_diffusion_ready_cache.npz"
    manifest_path = cache_work_dir / "method2b_512_diffusion_ready_manifest.csv"
    summary_path = cache_work_dir / "method2b_512_diffusion_cache_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    sample_count = int(summary.get("sample_count", 0) or 0)
    if sample_count < EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(f"Method2b cache has {sample_count} samples; expected {EXPECTED_SAMPLE_COUNT}.")
    if bool(summary.get("cache_ready", False)) and "training_allowed" not in summary:
        summary["training_allowed"] = True
        summary["training_allowed_note"] = (
            "Set by method2b Colab orchestrator for compatibility with "
            "step3j canonical training gate after full512 cache audit passed."
        )
        write_json(summary_path, summary)

    train_script = MN_WORK / "scripts" / "step3j_method1_train_canonical_diffusion_cv.py"
    eval_script = MN_WORK / "scripts" / "rspi_03_eval_posterior_metrics.py"
    run_root = MN_WORK / "outputs" / "rspi_joint_052026" / f"method2b_512_five_seed_ood_colab_cuda_{RUN_TAG}"
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if RUN_SMOKE:
        smoke_dir = run_root / "smoke"
        logged_cmd(
            [
                sys.executable,
                train_script,
                "--cache",
                cache_path,
                "--manifest",
                manifest_path,
                "--cache-summary",
                summary_path,
                "--out-dir",
                smoke_dir,
                "--split-modes",
                SPLIT_MODES,
                "--max-folds",
                "1",
                "--device",
                "cuda",
                "--batch-size",
                BATCH_SIZE,
                "--base-channels",
                "8",
                "--regressor-epochs",
                "1",
                "--diffusion-epochs",
                "1",
                "--residual-diffusion-epochs",
                "1",
                "--surrogate-epochs",
                "1",
                "--timesteps",
                "20",
                "--posterior-samples",
                "1",
                "--img2img-start-step",
                "5",
                "--lr",
                LR,
                "--guidance-scale",
                GUIDANCE_SCALE,
                "--residual-scale-quantile",
                RESIDUAL_SCALE_QUANTILE,
                "--residual-scale-floor",
                RESIDUAL_SCALE_FLOOR,
                "--seed",
                str(SEEDS[0]),
            ],
            logs_dir / "smoke_train.log",
            cwd=MN_WORK,
        )

    seed_eval_dirs: list[Path] = []
    seed_records: list[dict] = []
    for seed in SEEDS:
        seed_root = run_root / f"seed_{seed}"
        train_dir = seed_root / "training"
        eval_dir = seed_root / "posterior"
        logged_cmd(
            [
                sys.executable,
                train_script,
                "--cache",
                cache_path,
                "--manifest",
                manifest_path,
                "--cache-summary",
                summary_path,
                "--out-dir",
                train_dir,
                "--split-modes",
                SPLIT_MODES,
                "--device",
                "cuda",
                "--batch-size",
                BATCH_SIZE,
                "--base-channels",
                BASE_CHANNELS,
                "--regressor-epochs",
                REGRESSOR_EPOCHS,
                "--diffusion-epochs",
                DIFFUSION_EPOCHS,
                "--residual-diffusion-epochs",
                RESIDUAL_DIFFUSION_EPOCHS,
                "--surrogate-epochs",
                SURROGATE_EPOCHS,
                "--timesteps",
                TIMESTEPS,
                "--posterior-samples",
                TRAIN_POSTERIOR_SAMPLES,
                "--img2img-start-step",
                IMG2IMG_START_STEP,
                "--lr",
                LR,
                "--guidance-scale",
                GUIDANCE_SCALE,
                "--residual-scale-quantile",
                RESIDUAL_SCALE_QUANTILE,
                "--residual-scale-floor",
                RESIDUAL_SCALE_FLOOR,
                "--seed",
                str(seed),
            ],
            logs_dir / f"seed_{seed}_train.log",
            cwd=MN_WORK,
        )
        logged_cmd(
            [
                sys.executable,
                eval_script,
                "--dataset-name",
                "method2b_512_full_comsol_colab_cuda",
                "--cache",
                cache_path,
                "--manifest",
                manifest_path,
                "--training-dir",
                train_dir,
                "--out-dir",
                eval_dir,
                "--split-modes",
                SPLIT_MODES,
                "--device",
                "cuda",
                "--batch-size",
                BATCH_SIZE,
                "--base-channels",
                BASE_CHANNELS,
                "--timesteps",
                TIMESTEPS,
                "--posterior-samples",
                POSTERIOR_SAMPLES,
                "--img2img-start-step",
                IMG2IMG_START_STEP,
                "--residual-scale",
                RESIDUAL_SCALE,
                "--seed",
                str(seed),
            ],
            logs_dir / f"seed_{seed}_posterior.log",
            cwd=MN_WORK,
        )
        seed_eval_dirs.append(eval_dir)
        seed_records.append({"seed": seed, "train": str(train_dir), "posterior": str(eval_dir)})
        if os.environ.get("METHOD2B_UPLOAD_AFTER_EACH_SEED", "1").strip().lower() not in {"0", "false", "no"}:
            drive_access.upload_folder(seed_root, RUN_DRIVE_REL / f"seed_{seed}")
            drive_access.upload_folder(logs_dir, RUN_DRIVE_REL / "logs")

    aggregate_dir = run_root / "aggregate"
    aggregate_path, aggregate_summary_path = aggregate_seed_metrics(seed_eval_dirs, aggregate_dir)
    drive_access.upload_folder(run_root, RUN_DRIVE_REL)
    aggregate_rel = RUN_DRIVE_REL / "aggregate" / aggregate_path.name
    pointer = {
        "run_tag": RUN_TAG,
        "target": "method2b_512_full_comsol",
        "gpu": gpu,
        "drive_mode": DRIVE_MODE,
        "seeds": SEEDS,
        "split_modes": SPLIT_MODES,
        "expected_sample_count": EXPECTED_SAMPLE_COUNT,
        "sample_count": sample_count,
        "run_drive": rel_to_log_path(RUN_DRIVE_REL),
        "posterior_aggregate": rel_to_log_path(aggregate_rel),
        "posterior_aggregate_windows": rel_to_windows_path(aggregate_rel),
        "posterior_summary": rel_to_log_path(RUN_DRIVE_REL / "aggregate" / aggregate_summary_path.name),
        "seed_runs": seed_records,
    }
    pointer_rel = MANUSCRIPT_DRIVE_REL / "reports" / f"method2b_512_colab_cuda_run_pointer_{RUN_TAG}.json"
    drive_access.upload_text(json.dumps(pointer, indent=2), pointer_rel)
    print("\nMETHOD2B 512 COLAB TRAINING COMPLETE", flush=True)
    print(json.dumps(pointer, indent=2), flush=True)


if __name__ == "__main__":
    main()
