from __future__ import annotations

import json
import io
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath


RUN_TAG = os.environ.get("FIG3_COLAB_RUN_TAG") or time.strftime("%Y%m%d_%H%M%S")
DRIVE_MODE = os.environ.get("FIG3_COLAB_DRIVE_MODE", "api").strip().lower()
SEEDS = [int(x.strip()) for x in os.environ.get("FIG3_COLAB_SEEDS", "23,101,202,303,404").split(",") if x.strip()]
BASE_SPLIT_MODES = [x.strip() for x in os.environ.get("FIG3_COLAB_SPLIT_MODES", "leave_prior_group,leave_disease_family").split(",") if x.strip()]
OOD_COLUMNS = [x.strip() for x in os.environ.get("FIG3_COLAB_OOD_COLUMNS", "ood_family,shape_family,depth_bin,layer_modulus_family,operator_noise_family").split(",") if x.strip()]
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
DRIVE_ROOT = Path("/content/drive/MyDrive")
MN_DRIVE = DRIVE_ROOT / "MN_simulation"
MANUSCRIPT_DRIVE = DRIVE_ROOT / "Manuscript" / "Sparse reconstruction"
MN_DRIVE_REL = PurePosixPath("MN_simulation")
MANUSCRIPT_DRIVE_REL = PurePosixPath("Manuscript") / "Sparse reconstruction"
WORK_ROOT = Path("/content/fig3_colab_work")
MN_WORK = WORK_ROOT / "MN_simulation"


def run_cmd(cmd: list[object], cwd: Path | None = None) -> None:
    print("\nRUN:", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
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
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(map(str, cmd))}")


def copytree_clean(src: Path, dst: Path, ignore_names: set[str] | None = None) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    ignore_names = ignore_names or {"__pycache__", ".pytest_cache"}
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*sorted(ignore_names)))


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copytree_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"Copied {src} -> {dst}", flush=True)


def rel_to_windows_path(rel: PurePosixPath) -> str:
    return "H:\\My Drive\\" + "\\".join(rel.parts)


def rel_to_log_path(rel: PurePosixPath) -> str:
    if DRIVE_MODE == "mount":
        return str(DRIVE_ROOT / rel.as_posix())
    return "gdrive://MyDrive/" + rel.as_posix()


def get_drive_service():
    print("Authenticating Google Drive API with full-drive scope...", flush=True)
    from google.colab import auth

    auth.authenticate_user()

    import google.auth
    from googleapiclient.discovery import build

    try:
        creds, _ = google.auth.default(scopes=DRIVE_SCOPES)
    except TypeError:
        creds, _ = google.auth.default()
        if getattr(creds, "requires_scopes", False):
            creds = creds.with_scopes(DRIVE_SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


class MountedDriveAccess:
    def __init__(self) -> None:
        if not DRIVE_ROOT.exists():
            from google.colab import drive

            drive.mount("/content/drive", force_remount=False)

    def path(self, rel: PurePosixPath) -> Path:
        return DRIVE_ROOT / rel.as_posix()

    def exists(self, rel: PurePosixPath) -> bool:
        return self.path(rel).exists()

    def download_folder(self, rel: PurePosixPath, dst: Path, ignore_names: set[str] | None = None) -> None:
        copytree_clean(self.path(rel), dst, ignore_names=ignore_names)

    def download_file(self, rel: PurePosixPath, dst: Path) -> None:
        copy_file(self.path(rel), dst)

    def upload_folder(self, src: Path, rel: PurePosixPath) -> None:
        copytree_replace(src, self.path(rel))

    def upload_text(self, text: str, rel: PurePosixPath) -> None:
        dst = self.path(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")
        print(f"Wrote {dst}", flush=True)


class ApiDriveAccess:
    FOLDER_MIME = "application/vnd.google-apps.folder"

    def __init__(self) -> None:
        self.service = get_drive_service()

    @staticmethod
    def _parts(rel: PurePosixPath) -> list[str]:
        return [part for part in rel.parts if part not in ("", ".")]

    @staticmethod
    def _escape_query(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    def _find_child(self, parent_id: str, name: str, mime_type: str | None = None) -> dict | None:
        query = f"'{parent_id}' in parents and name = '{self._escape_query(name)}' and trashed = false"
        if mime_type:
            query += f" and mimeType = '{mime_type}'"
        response = (
            self.service.files()
            .list(
                q=query,
                spaces="drive",
                fields="files(id,name,mimeType,size,modifiedTime)",
                pageSize=10,
            )
            .execute()
        )
        files = response.get("files", [])
        return files[0] if files else None

    def _list_children(self, parent_id: str) -> list[dict]:
        items: list[dict] = []
        page_token = None
        while True:
            response = (
                self.service.files()
                .list(
                    q=f"'{parent_id}' in parents and trashed = false",
                    spaces="drive",
                    fields="nextPageToken, files(id,name,mimeType,size,modifiedTime)",
                    pageSize=1000,
                    pageToken=page_token,
                )
                .execute()
            )
            items.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                return items

    def resolve(self, rel: PurePosixPath, expect_folder: bool | None = None) -> dict:
        parent_id = "root"
        current: dict | None = None
        for part in self._parts(rel):
            current = self._find_child(parent_id, part)
            if current is None:
                raise FileNotFoundError(f"Google Drive path not found: MyDrive/{rel.as_posix()}")
            parent_id = current["id"]
        if current is None:
            current = {"id": "root", "name": "MyDrive", "mimeType": self.FOLDER_MIME}
        is_folder = current.get("mimeType") == self.FOLDER_MIME
        if expect_folder is True and not is_folder:
            raise NotADirectoryError(f"Expected Drive folder: MyDrive/{rel.as_posix()}")
        if expect_folder is False and is_folder:
            raise IsADirectoryError(f"Expected Drive file: MyDrive/{rel.as_posix()}")
        return current

    def exists(self, rel: PurePosixPath) -> bool:
        try:
            self.resolve(rel)
            return True
        except FileNotFoundError:
            return False

    def ensure_folder(self, rel: PurePosixPath) -> str:
        parent_id = "root"
        for part in self._parts(rel):
            current = self._find_child(parent_id, part, self.FOLDER_MIME)
            if current is None:
                current = (
                    self.service.files()
                    .create(
                        body={"name": part, "mimeType": self.FOLDER_MIME, "parents": [parent_id]},
                        fields="id,name,mimeType",
                    )
                    .execute()
                )
            parent_id = current["id"]
        return parent_id

    def download_file(self, rel: PurePosixPath, dst: Path) -> None:
        from googleapiclient.http import MediaIoBaseDownload

        item = self.resolve(rel, expect_folder=False)
        dst.parent.mkdir(parents=True, exist_ok=True)
        request = self.service.files().get_media(fileId=item["id"])
        with dst.open("wb") as handle:
            downloader = MediaIoBaseDownload(handle, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        print(f"Downloaded {rel_to_log_path(rel)} -> {dst}", flush=True)

    def download_folder(self, rel: PurePosixPath, dst: Path, ignore_names: set[str] | None = None) -> None:
        item = self.resolve(rel, expect_folder=True)
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        self._download_folder_by_id(item["id"], dst, ignore_names or {"__pycache__", ".pytest_cache"})
        print(f"Downloaded {rel_to_log_path(rel)} -> {dst}", flush=True)

    def _download_folder_by_id(self, folder_id: str, dst: Path, ignore_names: set[str]) -> None:
        for item in self._list_children(folder_id):
            name = item["name"]
            if name in ignore_names:
                continue
            target = dst / name
            mime_type = item.get("mimeType")
            if mime_type == self.FOLDER_MIME:
                target.mkdir(parents=True, exist_ok=True)
                self._download_folder_by_id(item["id"], target, ignore_names)
            elif str(mime_type).startswith("application/vnd.google-apps."):
                print(f"Skipping Google-native file during download: {name}", flush=True)
            else:
                self._download_file_by_id(item["id"], target)

    def _download_file_by_id(self, file_id: str, dst: Path) -> None:
        from googleapiclient.http import MediaIoBaseDownload

        dst.parent.mkdir(parents=True, exist_ok=True)
        request = self.service.files().get_media(fileId=file_id)
        with dst.open("wb") as handle:
            downloader = MediaIoBaseDownload(handle, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

    def upload_folder(self, src: Path, rel: PurePosixPath) -> None:
        parent_id = self.ensure_folder(rel.parent)
        existing = self._find_child(parent_id, rel.name, self.FOLDER_MIME)
        if existing:
            self.service.files().update(fileId=existing["id"], body={"trashed": True}).execute()
        folder = (
            self.service.files()
            .create(body={"name": rel.name, "mimeType": self.FOLDER_MIME, "parents": [parent_id]}, fields="id,name")
            .execute()
        )
        self._upload_folder_contents(src, folder["id"])
        print(f"Uploaded {src} -> {rel_to_log_path(rel)}", flush=True)

    def _upload_folder_contents(self, src: Path, parent_id: str) -> None:
        for child in sorted(src.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if child.name in {"__pycache__", ".pytest_cache"}:
                continue
            if child.is_dir():
                folder = (
                    self.service.files()
                    .create(
                        body={"name": child.name, "mimeType": self.FOLDER_MIME, "parents": [parent_id]},
                        fields="id,name",
                    )
                    .execute()
                )
                self._upload_folder_contents(child, folder["id"])
            else:
                self._upload_file_to_parent(child, parent_id)

    def _upload_file_to_parent(self, src: Path, parent_id: str) -> None:
        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(str(src), resumable=True)
        self.service.files().create(body={"name": src.name, "parents": [parent_id]}, media_body=media, fields="id").execute()

    def upload_text(self, text: str, rel: PurePosixPath) -> None:
        from googleapiclient.http import MediaIoBaseUpload

        parent_id = self.ensure_folder(rel.parent)
        existing = self._find_child(parent_id, rel.name)
        mime_type = "application/json" if rel.suffix.lower() == ".json" else "text/plain"
        media = MediaIoBaseUpload(io.BytesIO(text.encode("utf-8")), mimetype=mime_type, resumable=False)
        if existing:
            self.service.files().update(fileId=existing["id"], media_body=media, fields="id").execute()
        else:
            self.service.files().create(body={"name": rel.name, "parents": [parent_id]}, media_body=media, fields="id").execute()
        print(f"Uploaded text -> {rel_to_log_path(rel)}", flush=True)


def build_drive_access() -> MountedDriveAccess | ApiDriveAccess:
    if DRIVE_MODE == "mount":
        print("Using mounted Google Drive filesystem access.", flush=True)
        return MountedDriveAccess()
    if DRIVE_MODE != "api":
        raise ValueError("FIG3_COLAB_DRIVE_MODE must be 'api' or 'mount'.")
    print("Using Google Drive API access; no /content/drive mount is required.", flush=True)
    return ApiDriveAccess()


def split_modes_for_manifest(manifest_path: Path) -> str:
    import pandas as pd

    manifest = pd.read_csv(manifest_path)
    modes = list(BASE_SPLIT_MODES)
    for column in OOD_COLUMNS:
        if column in manifest.columns and manifest[column].astype(str).nunique() >= 3:
            modes.append(f"leave_column__{column}")
    deduped = list(dict.fromkeys(modes))
    print("FIG3_COLAB_SPLIT_MODES_RESOLVED", ",".join(deduped), flush=True)
    return ",".join(deduped)


def bootstrap_ci(values, rng, n_boot: int = 2000) -> tuple[float, float]:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    draws = rng.choice(arr, size=(int(n_boot), arr.size), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def aggregate_multiseed_eval(seed_eval_dirs: list[Path], out_dir: Path) -> tuple[Path, Path]:
    import numpy as np
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    metric_frames = []
    aggregate_frames = []
    for eval_dir in seed_eval_dirs:
        seed_match = eval_dir.name.split("_seed")[-1].split("_", 1)[0] if "_seed" in eval_dir.name else ""
        metrics_path = eval_dir / "rspi_posterior_metrics.csv"
        aggregate_path = eval_dir / "rspi_posterior_aggregate.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            metrics["seed"] = int(seed_match) if seed_match.isdigit() else seed_match
            metric_frames.append(metrics)
        if aggregate_path.exists():
            aggregate = pd.read_csv(aggregate_path)
            aggregate["seed"] = int(seed_match) if seed_match.isdigit() else seed_match
            aggregate_frames.append(aggregate)

    if not metric_frames:
        raise RuntimeError("No seed-level rspi_posterior_metrics.csv files were produced.")
    metrics_all = pd.concat(metric_frames, ignore_index=True)
    metrics_all.to_csv(out_dir / "rspi_posterior_metrics_all_seeds.csv", index=False)
    if aggregate_frames:
        pd.concat(aggregate_frames, ignore_index=True).to_csv(out_dir / "rspi_posterior_aggregate_by_seed.csv", index=False)

    group_cols = [c for c in ["dataset", "split_mode", "model"] if c in metrics_all.columns]
    numeric_cols = [
        c
        for c in metrics_all.columns
        if c not in set(group_cols + ["fold", "seed"])
        and pd.api.types.is_numeric_dtype(metrics_all[c])
    ]
    rng = np.random.default_rng(53126)
    rows = []
    for keys, sub in metrics_all.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_seed_folds"] = int(len(sub))
        row["n_seeds"] = int(sub["seed"].nunique()) if "seed" in sub.columns else 0
        for col in numeric_cols:
            vals = sub[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            lo, hi = bootstrap_ci(vals, rng)
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_sd"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            row[f"{col}_boot95_low"] = lo
            row[f"{col}_boot95_high"] = hi
        rows.append(row)
    multiseed = pd.DataFrame(rows).sort_values(group_cols + [c for c in ["rmse_kpa_mean", "rmse_norm_mean"] if c in rows[0]], na_position="last")
    aggregate_path = out_dir / "rspi_posterior_aggregate_multiseed_bootstrap.csv"
    summary_path = out_dir / "rspi_posterior_multiseed_summary.json"
    multiseed.to_csv(aggregate_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "run_tag": RUN_TAG,
                "seed_eval_dirs": [str(p) for p in seed_eval_dirs],
                "aggregate_path": str(aggregate_path),
                "metrics_all_seeds": str(out_dir / "rspi_posterior_metrics_all_seeds.csv"),
                "n_seed_folds": int(len(metrics_all)),
                "n_rows": int(len(multiseed)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return aggregate_path, summary_path


def main() -> None:
    print("FIG3_COLAB_RUN_TAG", RUN_TAG, flush=True)
    print("FIG3_COLAB_DRIVE_MODE", DRIVE_MODE, flush=True)
    print("FIG3_COLAB_SEEDS", SEEDS, flush=True)
    drive_access = build_drive_access()
    run_cmd(["nvidia-smi"])
    import torch

    print("torch", torch.__version__, flush=True)
    print("torch.version.cuda", torch.version.cuda, flush=True)
    print("torch.cuda.is_available", torch.cuda.is_available(), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use Runtime > Change runtime type > GPU.")
    print("GPU", torch.cuda.get_device_name(0), flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_cmd([sys.executable, "-m", "pip", "install", "-q", "numpy", "pandas", "scipy", "matplotlib", "scikit-learn", "tqdm"])

    if not drive_access.exists(MN_DRIVE_REL):
        raise FileNotFoundError(f"MN_simulation not found in Drive: {rel_to_log_path(MN_DRIVE_REL)}")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    drive_access.download_folder(MN_DRIVE_REL / "scripts", MN_WORK / "scripts")

    src_cache_dir = MN_DRIVE_REL / "outputs" / "method2c" / "targetmap_diffusion"
    dst_cache_dir = MN_WORK / "outputs" / "method2c" / "targetmap_diffusion"
    for name in [
        "method2c_targetmap_diffusion_ready_cache.npz",
        "method2c_targetmap_diffusion_ready_manifest.csv",
        "method2c_targetmap_diffusion_cache_summary.json",
        "method2c_targetmap_diffusion_cache_CN.md",
        "method2c_targetmap_diffusion_excluded_cases.csv",
    ]:
        drive_access.download_file(src_cache_dir / name, dst_cache_dir / name)

    cache_path = dst_cache_dir / "method2c_targetmap_diffusion_ready_cache.npz"
    manifest_path = dst_cache_dir / "method2c_targetmap_diffusion_ready_manifest.csv"
    summary_path = dst_cache_dir / "method2c_targetmap_diffusion_cache_summary.json"
    train_script = MN_WORK / "scripts" / "step3j_method1_train_canonical_diffusion_cv.py"
    eval_script = MN_WORK / "scripts" / "rspi_03_eval_posterior_metrics.py"
    split_modes = split_modes_for_manifest(manifest_path)
    first_seed = int(SEEDS[0])

    smoke_dir = dst_cache_dir / f"colab_smoke_cuda_{RUN_TAG}"
    run_cmd(
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
            "leave_prior_group",
            "--max-folds",
            "1",
            "--device",
            "cuda",
            "--batch-size",
            "16",
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
            "2e-3",
            "--guidance-scale",
            "0.06",
            "--residual-scale-quantile",
            "0.95",
            "--residual-scale-floor",
            "0.03",
            "--seed",
            str(first_seed),
        ],
        cwd=MN_WORK,
    )

    seed_records: list[dict[str, object]] = []
    seed_eval_dirs: list[Path] = []
    for seed in SEEDS:
        train_dir = dst_cache_dir / f"colab_cuda_t200_img2img_s10_q95_seed{seed}_{RUN_TAG}"
        run_cmd(
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
                split_modes,
                "--max-folds",
                "0",
                "--device",
                "cuda",
                "--batch-size",
                "16",
                "--base-channels",
                "16",
                "--regressor-epochs",
                "20",
                "--diffusion-epochs",
                "80",
                "--residual-diffusion-epochs",
                "80",
                "--surrogate-epochs",
                "20",
                "--timesteps",
                "200",
                "--posterior-samples",
                "4",
                "--img2img-start-step",
                "10",
                "--lr",
                "1e-3",
                "--guidance-scale",
                "0.06",
                "--residual-scale-quantile",
                "0.95",
                "--residual-scale-floor",
                "0.03",
                "--seed",
                str(seed),
            ],
            cwd=MN_WORK,
        )

        eval_dir = MN_WORK / "outputs" / "rspi_joint_052026" / f"method2c_targetmap_posterior64_colab_cuda_seed{seed}_{RUN_TAG}"
        run_cmd(
            [
                sys.executable,
                eval_script,
                "--dataset-name",
                "method2c_targetmap_colab_cuda_multiseed",
                "--cache",
                cache_path,
                "--manifest",
                manifest_path,
                "--training-dir",
                train_dir,
                "--out-dir",
                eval_dir,
                "--split-modes",
                split_modes,
                "--max-folds",
                "0",
                "--device",
                "cuda",
                "--batch-size",
                "16",
                "--base-channels",
                "16",
                "--timesteps",
                "200",
                "--posterior-samples",
                "64",
                "--img2img-start-step",
                "10",
                "--residual-scale",
                "0.25",
                "--seed",
                str(seed),
            ],
            cwd=MN_WORK,
        )
        train_drive_rel = MN_DRIVE_REL / "outputs" / "method2c" / "targetmap_diffusion" / train_dir.name
        eval_drive_rel = MN_DRIVE_REL / "outputs" / "rspi_joint_052026" / eval_dir.name
        drive_access.upload_folder(train_dir, train_drive_rel)
        drive_access.upload_folder(eval_dir, eval_drive_rel)
        seed_eval_dirs.append(eval_dir)
        seed_records.append(
            {
                "seed": seed,
                "train_drive": rel_to_log_path(train_drive_rel),
                "eval_drive": rel_to_log_path(eval_drive_rel),
                "posterior_aggregate": rel_to_log_path(eval_drive_rel / "rspi_posterior_aggregate.csv"),
            }
        )

    combined_eval_dir = MN_WORK / "outputs" / "rspi_joint_052026" / f"method2c_targetmap_posterior64_colab_cuda_multiseed_{RUN_TAG}"
    multiseed_aggregate_path, multiseed_summary_path = aggregate_multiseed_eval(seed_eval_dirs, combined_eval_dir)
    combined_eval_drive_rel = MN_DRIVE_REL / "outputs" / "rspi_joint_052026" / combined_eval_dir.name
    drive_access.upload_folder(combined_eval_dir, combined_eval_drive_rel)
    posterior_aggregate_rel = combined_eval_drive_rel / multiseed_aggregate_path.name

    pointer = {
        "run_tag": RUN_TAG,
        "gpu": torch.cuda.get_device_name(0),
        "drive_mode": DRIVE_MODE,
        "drive_scope": DRIVE_SCOPES[0],
        "seeds": SEEDS,
        "split_modes": split_modes,
        "seed_runs": seed_records,
        "combined_eval_drive": rel_to_log_path(combined_eval_drive_rel),
        "posterior_aggregate": rel_to_log_path(posterior_aggregate_rel),
        "posterior_multiseed_summary": rel_to_log_path(combined_eval_drive_rel / multiseed_summary_path.name),
        "windows_fig3_command": (
            "$env:FIG3_MMP_POSTERIOR='"
            + rel_to_windows_path(posterior_aggregate_rel)
            + "'; "
            + "py -3.12 scripts\\fig3_data_pipeline.py metrics; "
            + "py -3.12 scripts\\fig3_data_pipeline.py render; "
            + "py -3.12 scripts\\fig3_data_pipeline.py composite; "
            + "py -3.12 scripts\\fig3_data_pipeline.py qa"
        ),
    }
    pointer_rel = MANUSCRIPT_DRIVE_REL / "reports" / f"colab_cuda_run_pointer_{RUN_TAG}.json"
    drive_access.upload_text(json.dumps(pointer, indent=2), pointer_rel)
    print("\nCOLAB FIGURE 3 TRAINING COMPLETE", flush=True)
    print(json.dumps(pointer, indent=2), flush=True)


if __name__ == "__main__":
    main()
