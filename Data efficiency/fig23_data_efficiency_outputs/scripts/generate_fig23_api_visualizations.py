from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(r"H:\My Drive\Manuscript\Sparse reconstruction")
RESULTS = ROOT / "Data" / "Data efficiency" / "data_efficiency_results_20260604"
OUT = RESULTS / "api_generated_visualizations"
OUT.mkdir(parents=True, exist_ok=True)


def request_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int = 180) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:4000]}") from exc


def save_b64_image(b64_data: str, path: Path) -> None:
    path.write_bytes(base64.b64decode(b64_data))


def compact_prompt() -> str:
    checklist = (RESULTS / "TARGET_COMPLETION_CHECKLIST.md").read_text(encoding="utf-8")
    audit = (RESULTS / "OPENAI_GPT5_NCS_AUDIT_VALIDATED.md").read_text(encoding="utf-8")
    summary = pd.read_csv(RESULTS / "fig23_N90_AULC_DER_summary.csv")
    coverage = pd.read_csv(RESULTS / "model_device_coverage_matrix.csv")

    f2 = coverage[coverage["figure"] == "Figure 2"]
    f3 = coverage[coverage["figure"] == "Figure 3"]
    claims = summary[
        [
            "figure",
            "device_label",
            "model_label",
            "N_to_operator_quality",
            "DER_common_quality_vs_operator",
            "common_quality_reached",
        ]
    ].copy()
    claims = claims.sort_values(["figure", "device_label", "DER_common_quality_vs_operator"], ascending=[True, True, False])
    claim_lines = []
    for _, row in claims.iterrows():
        claim_lines.append(
            f"{row['figure']} {row['device_label']} {row['model_label']}: "
            f"Ntarget {float(row['N_to_operator_quality']):.0f}, "
            f"DER {float(row['DER_common_quality_vs_operator']):.2f}, "
            f"reached {bool(row['common_quality_reached'])}"
        )

    return (
        "Create a clean scientific dashboard for Figure 2 and Figure 3 data-efficiency results. "
        "This is a style exploration image only; exact numeric values are in local CSV files. "
        "Use a Nature Computational Science-like visual tone: white background, black/gray text, restrained accent colors, "
        "three-device columns for soft MMP, sheet piezo, and ultrasound. "
        "Show two rows: Figure 2 2D Eapp reconstruction and Figure 3 3D z-bottom reconstruction. "
        "Show model families as compact color-coded line glyphs or bars: SRCNN, U-Net or 3D U-Net, vanilla diffusion, "
        "DPS inverse diffusion, Eapp-only diffusion where applicable, kernel GP where applicable, and operator-conditioned diffusion. "
        "Emphasize that operator-conditioned diffusion is the shared target comparator and that censored DER values need full retraining. "
        "Avoid tiny unreadable text. Do not invent extra devices, models, or numeric values. "
        f"Coverage: Figure 2 has {len(f2)} device-model cells; Figure 3 has {len(f3)} device-model cells. "
        "Key result lines: " + " | ".join(claim_lines[:39]) + ". "
        "Checklist summary: " + checklist[:1800].replace("\n", " ") + ". "
        "Validated audit summary: " + audit[:1800].replace("\n", " ") + "."
    )


def generate_openai(prompt: str, log: list[dict[str, Any]]) -> Path | None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        log.append({"provider": "openai", "status": "skipped", "reason": "OPENAI_API_KEY not available"})
        return None
    models = ["gpt-image-1.5", "gpt-image-1", "gpt-image-1-mini"]
    last_error = None
    for model in models:
        payload = {
            "model": model,
            "prompt": prompt,
            "size": "1536x1024",
            "quality": "high",
            "n": 1,
            "output_format": "png",
        }
        try:
            started = time.time()
            response = request_json(
                "https://api.openai.com/v1/images/generations",
                payload,
                {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                timeout=240,
            )
            item = (response.get("data") or [{}])[0]
            out = OUT / f"openai_{model}_fig23_data_efficiency.png"
            if item.get("b64_json"):
                save_b64_image(item["b64_json"], out)
            elif item.get("url"):
                urllib.request.urlretrieve(item["url"], out)
            else:
                raise RuntimeError(f"No image content in response keys={list(item.keys())}")
            log.append({"provider": "openai", "status": "success", "model": model, "elapsed_sec": round(time.time() - started, 2), "output": str(out)})
            return out
        except Exception as exc:
            last_error = str(exc)
            log.append({"provider": "openai", "status": "failed_attempt", "model": model, "error": last_error[:1000]})
    log.append({"provider": "openai", "status": "failed", "error": (last_error or "unknown")[:2000]})
    return None


def extract_gemini_image(response: dict[str, Any]) -> tuple[str | None, str | None]:
    for cand in response.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return inline.get("data"), inline.get("mimeType") or inline.get("mime_type")
    return None, None


def generate_gemini(prompt: str, log: list[dict[str, Any]]) -> Path | None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_GENAI_API_KEY")
    if not key:
        log.append({"provider": "gemini_nanobanana", "status": "skipped", "reason": "Gemini API key not available"})
        return None
    models = ["gemini-2.5-flash-image", "gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"]
    last_error = None
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(model)}:generateContent?key={urllib.parse.quote(key)}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
        }
        try:
            started = time.time()
            response = request_json(url, payload, {"Content-Type": "application/json"}, timeout=240)
            b64_data, mime = extract_gemini_image(response)
            if not b64_data:
                raise RuntimeError("No inline image in Gemini response")
            ext = mimetypes.guess_extension(mime or "image/png") or ".png"
            if ext == ".jpe":
                ext = ".jpg"
            out = OUT / f"nanobanana_{model}_fig23_data_efficiency{ext}"
            save_b64_image(b64_data, out)
            log.append({"provider": "gemini_nanobanana", "status": "success", "model": model, "elapsed_sec": round(time.time() - started, 2), "mime": mime, "output": str(out)})
            return out
        except Exception as exc:
            last_error = str(exc)
            log.append({"provider": "gemini_nanobanana", "status": "failed_attempt", "model": model, "error": last_error[:1000]})
    log.append({"provider": "gemini_nanobanana", "status": "failed", "error": (last_error or "unknown")[:2000]})
    return None


def main() -> None:
    log: list[dict[str, Any]] = []
    prompt = compact_prompt()
    (OUT / "fig23_api_visualization_prompt.txt").write_text(prompt, encoding="utf-8")
    openai_path = generate_openai(prompt, log)
    gemini_path = generate_gemini(prompt, log)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "openai_output": str(openai_path) if openai_path else None,
        "nanobanana_output": str(gemini_path) if gemini_path else None,
        "log": log,
    }
    (OUT / "fig23_api_generation_log.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not (openai_path or gemini_path):
        sys.exit(2)


if __name__ == "__main__":
    main()
