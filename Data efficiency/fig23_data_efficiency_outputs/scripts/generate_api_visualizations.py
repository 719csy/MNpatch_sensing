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
AUDIT = ROOT / "Data" / "Data efficiency" / "ncs_audit_20260604"
OUT = AUDIT / "api_generated_visualizations"
OUT.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_prompt() -> str:
    summary = read_json(AUDIT / "ncs_audit_summary.json")
    gates = pd.read_csv(AUDIT / "ncs_gate_decisions.csv")
    axis = pd.read_csv(AUDIT / "ncs_category_axis_summary.csv")
    metrics = pd.read_csv(AUDIT / "ncs_primary_metric_snapshot.csv")

    gate_lines = []
    for _, r in gates.iterrows():
        gate_lines.append(f"{r['gate']} ({r['figure']}): {r['status']} - {r['current_value']}")

    axis_lines = []
    for _, r in axis.iterrows():
        axis_lines.append(
            f"{r['figure']} {r['axis']}: {int(r['n_classes'])} classes, status {r['overall_status']}, "
            f"min n {r['min_current_n']}, max n {r['max_current_n']}"
        )

    metric_lines = []
    for _, r in metrics.iterrows():
        if r["figure"] == "Figure 2":
            metric_lines.append(
                f"Figure 2 op-conditioned diffusion + mask: MAE {float(r['MAE_kPa']):.3f} kPa, "
                f"SSIM {float(r['SSIM']):.3f}, Dice {float(r['Dice']):.3f}, Boundary-F1 {float(r['Boundary_F1']):.3f}"
            )
        else:
            metric_lines.append(
                f"{r['metric_family']}: MAE_zbottom {float(r['MAE_zbottom_mm']):.3f} mm, "
                f"Cov90 {float(r['Cov90']):.3f}, Width90 {float(r['Width90_mm']):.3f} mm, "
                f"IoU {float(r['Interval_IoU']):.3f}, AUROC_fail {float(r['AUROC_fail_1mm']):.3f}"
            )

    return (
        "Create a clean Nature Computational Science-style infographic dashboard for a manuscript data-efficiency audit. "
        "White background, restrained scientific palette, strong hierarchy, readable panel labels, no decorative stock imagery. "
        "Do not invent or change values. The visual should clearly distinguish exact local evidence from missing gates. "
        "Use three panels: (A) NCS readiness gates, (B) domain-randomization class coverage, (C) Figure 2/3 metric snapshot. "
        "Important headline: NOT YET READY for strong NCS data-efficiency/sim-to-real claim; ready only as simulation-grounded benchmark. "
        f"Gate status counts: {summary['gate_status_counts']}. "
        f"Figure 2: {summary['figure2_axes']} axes and {summary['figure2_axis_level_classes']} axis-level classes. "
        f"Figure 3: {summary['figure3_axes']} axes and {summary['figure3_axis_level_classes']} axis-level classes. "
        "Gate details: " + " | ".join(gate_lines) + ". "
        "Class summaries: " + " | ".join(axis_lines) + ". "
        "Metric snapshot: " + " | ".join(metric_lines) + ". "
        "Show blockers prominently: missing real MMP support envelope, missing learning curves/N90/DER, missing projection consistency, "
        "and Figure 2 posterior export not complete. Keep all text concise and professional."
    )


def request_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int = 180) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body[:4000]}") from e


def save_b64_image(b64_data: str, path: Path) -> None:
    path.write_bytes(base64.b64decode(b64_data))


def generate_openai(prompt: str, log: list[dict[str, Any]]) -> Path | None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        log.append({"provider": "openai", "status": "skipped", "reason": "OPENAI_API_KEY not available in process env"})
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
                {
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
            )
            elapsed = round(time.time() - started, 2)
            item = (response.get("data") or [{}])[0]
            if item.get("b64_json"):
                out = OUT / f"openai_{model}_ncs_data_efficiency.png"
                save_b64_image(item["b64_json"], out)
                log.append(
                    {
                        "provider": "openai",
                        "status": "success",
                        "model": model,
                        "elapsed_sec": elapsed,
                        "output": str(out),
                        "response_keys": list(item.keys()),
                    }
                )
                (OUT / f"openai_{model}_response_metadata.json").write_text(
                    json.dumps(
                        {
                            "created": response.get("created"),
                            "model": model,
                            "usage": response.get("usage"),
                            "data_keys": list(item.keys()),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return out
            if item.get("url"):
                out = OUT / f"openai_{model}_ncs_data_efficiency.png"
                urllib.request.urlretrieve(item["url"], out)
                log.append({"provider": "openai", "status": "success", "model": model, "elapsed_sec": elapsed, "output": str(out), "response_keys": list(item.keys())})
                return out
            raise RuntimeError(f"No b64_json or url in OpenAI response: keys={list(item.keys())}")
        except Exception as exc:
            last_error = str(exc)
            log.append({"provider": "openai", "status": "failed_attempt", "model": model, "error": last_error[:1000]})
    responses_path = generate_openai_responses(prompt, key, log)
    if responses_path:
        return responses_path
    log.append({"provider": "openai", "status": "failed", "error": (last_error or "unknown")[:2000]})
    return None


def extract_openai_responses_image(response: dict[str, Any]) -> str | None:
    for item in response.get("output", []):
        if item.get("type") == "image_generation_call":
            if item.get("result"):
                return item["result"]
            if item.get("b64_json"):
                return item["b64_json"]
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_image", "image_generation_call"}:
                if content.get("result"):
                    return content["result"]
                if content.get("b64_json"):
                    return content["b64_json"]
    return None


def generate_openai_responses(prompt: str, key: str, log: list[dict[str, Any]]) -> Path | None:
    attempts = [
        ("gpt-5.2", [{"type": "image_generation", "model": "gpt-image-1.5", "size": "1536x1024", "quality": "high", "action": "generate"}]),
        ("gpt-5", [{"type": "image_generation", "model": "gpt-image-1.5", "size": "1536x1024", "quality": "high", "action": "generate"}]),
        ("gpt-4.1", [{"type": "image_generation", "model": "gpt-image-1.5", "size": "1536x1024", "quality": "high", "action": "generate"}]),
        ("gpt-4o-mini", [{"type": "image_generation"}]),
        ("gpt-4o", [{"type": "image_generation"}]),
        ("gpt-5.2", [{"type": "image_generation"}]),
        ("gpt-5", [{"type": "image_generation"}]),
    ]
    last_error = None
    for model, tools in attempts:
        payload = {
            "model": model,
            "input": prompt,
            "tools": tools,
        }
        try:
            started = time.time()
            response = request_json(
                "https://api.openai.com/v1/responses",
                payload,
                {
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                timeout=240,
            )
            elapsed = round(time.time() - started, 2)
            b64 = extract_openai_responses_image(response)
            if not b64:
                output_types = [item.get("type") for item in response.get("output", [])]
                raise RuntimeError(f"No image_generation result in Responses API output. output_types={output_types}")
            out = OUT / f"openai_responses_{model}_ncs_data_efficiency.png"
            save_b64_image(b64, out)
            log.append(
                {
                    "provider": "openai_responses",
                    "status": "success",
                    "model": model,
                    "elapsed_sec": elapsed,
                    "output": str(out),
                    "response_id": response.get("id"),
                }
            )
            (OUT / f"openai_responses_{model}_response_metadata.json").write_text(
                json.dumps(
                    {
                        "id": response.get("id"),
                        "model": response.get("model"),
                        "usage": response.get("usage"),
                        "output_types": [item.get("type") for item in response.get("output", [])],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return out
        except Exception as exc:
            last_error = str(exc)
            log.append({"provider": "openai_responses", "status": "failed_attempt", "model": model, "error": last_error[:1000]})
    log.append({"provider": "openai_responses", "status": "failed", "error": (last_error or "unknown")[:2000]})
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
        log.append({"provider": "gemini_nanobanana", "status": "skipped", "reason": "GEMINI/GOOGLE API key not available in process env"})
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
            response = request_json(url, payload, {"Content-Type": "application/json"}, timeout=180)
            elapsed = round(time.time() - started, 2)
            b64_data, mime = extract_gemini_image(response)
            if not b64_data:
                text_parts = []
                for cand in response.get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        if "text" in part:
                            text_parts.append(part["text"])
                raise RuntimeError("No inline image in response. Text response: " + " ".join(text_parts)[:1000])
            ext = mimetypes.guess_extension(mime or "image/png") or ".png"
            if ext == ".jpe":
                ext = ".jpg"
            out = OUT / f"nanobanana_{model}_ncs_data_efficiency{ext}"
            save_b64_image(b64_data, out)
            log.append(
                {
                    "provider": "gemini_nanobanana",
                    "status": "success",
                    "model": model,
                    "elapsed_sec": elapsed,
                    "mime": mime,
                    "output": str(out),
                }
            )
            (OUT / f"nanobanana_{model}_response_metadata.json").write_text(
                json.dumps(
                    {
                        "model": model,
                        "candidate_count": len(response.get("candidates", [])),
                        "promptFeedback": response.get("promptFeedback"),
                        "usageMetadata": response.get("usageMetadata"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return out
        except Exception as exc:
            last_error = str(exc)
            log.append({"provider": "gemini_nanobanana", "status": "failed_attempt", "model": model, "error": last_error[:1000]})
    log.append({"provider": "gemini_nanobanana", "status": "failed", "error": (last_error or "unknown")[:2000]})
    return None


def main() -> None:
    log: list[dict[str, Any]] = []
    prompt = compact_prompt()
    (OUT / "api_visualization_prompt.txt").write_text(prompt, encoding="utf-8")
    openai_path = generate_openai(prompt, log)
    gemini_path = generate_gemini(prompt, log)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "openai_output": str(openai_path) if openai_path else None,
        "nanobanana_output": str(gemini_path) if gemini_path else None,
        "log": log,
    }
    (OUT / "api_generation_log.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not (openai_path or gemini_path):
        sys.exit(2)


if __name__ == "__main__":
    main()
