"""
Qualcomm AI Hub profiler for DistilBERT on Snapdragon 8 Elite (Hexagon NPU).

Workflow:
  1. Export DistilBERT to ONNX (FP32, opset 17, seq=128)
  2. Compile to QNN context binary targeting "Snapdragon 8 Elite QRD"
  3. Profile on real Hexagon hardware
  4. Parse results and compare with existing laptop ONNX Runtime numbers
"""

import json
import os
import sys
from pathlib import Path

import shutil

import numpy as np
import onnx
import qai_hub
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Reference numbers from the local ONNX Runtime laptop benchmark
# ---------------------------------------------------------------------------
LAPTOP_RESULTS = {
    "DEFAULT":          {"p99_ms": 3.866, "p50_ms": 3.040, "mean_ms": 3.087},
    "MEMORY_OPTIMIZED": {"p99_ms": 3.761, "p50_ms": 3.026, "mean_ms": 3.075},
    "PARALLEL":         {"p99_ms": 5.966, "p50_ms": 3.737, "mean_ms": 3.830},
    "platform": "Apple M-series CPU (ONNX Runtime 1.26, batch=1, seq=128)",
}

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
ONNX_PATH = MODELS_DIR / "distilbert.onnx"
RESULTS_JSON = RESULTS_DIR / "qai_hub_results.json"
RESULTS_TXT = RESULTS_DIR / "qai_hub_results.txt"

DEVICE_NAME = "Snapdragon 8 Elite QRD"
BATCH_SIZE = 1
SEQ_LEN = 128


# ---------------------------------------------------------------------------
# Step 0 — Configure AI Hub
# ---------------------------------------------------------------------------
def configure_hub() -> None:
    print("\n=== Qualcomm AI Hub Configuration ===")
    token = input("Enter your AI Hub API token: ").strip()
    if not token:
        print("ERROR: No token provided.")
        sys.exit(1)
    # Write ~/.qai_hub/client.ini — same format as `qai-hub configure`
    config_dir = Path.home() / ".qai_hub"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "client.ini"
    config_file.write_text(
        "[api]\n"
        "api_url = https://workbench.aihub.qualcomm.com\n"
        "web_url = https://workbench.aihub.qualcomm.com\n"
        f"api_token = {token}\n"
    )
    print("Token accepted.\n")


# ---------------------------------------------------------------------------
# Step 1 — Export DistilBERT to ONNX
# ---------------------------------------------------------------------------
def export_onnx() -> Path:
    MODELS_DIR.mkdir(exist_ok=True)

    if ONNX_PATH.exists():
        size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
        if size_mb > 50:
            print(f"[Step 1] ONNX model already present at {ONNX_PATH} ({size_mb:.1f} MB), skipping export.")
            return ONNX_PATH
        else:
            print(f"[Step 1] Existing ONNX at {ONNX_PATH} is only {size_mb:.1f} MB (weights missing), re-exporting …")
            ONNX_PATH.unlink()

    print("[Step 1] Downloading pre-converted DistilBERT ONNX from HuggingFace Hub …")
    print("         (optimum/distilbert-base-uncased-finetuned-sst-2-english — same transformer backbone)")
    downloaded = Path(hf_hub_download(
        repo_id="optimum/distilbert-base-uncased-finetuned-sst-2-english",
        filename="model.onnx",
    ))
    if downloaded.resolve() != ONNX_PATH.resolve():
        shutil.copy(str(downloaded), str(ONNX_PATH))

    size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
    print(f"[Step 1] ONNX ready: {ONNX_PATH} ({size_mb:.1f} MB)\n")
    return ONNX_PATH


# ---------------------------------------------------------------------------
# Step 2 — Compile for Hexagon via AI Hub
# ---------------------------------------------------------------------------
COMPILE_JOB_CACHE = RESULTS_DIR / ".compile_job_id"


def compile_for_hexagon(onnx_path: Path) -> qai_hub.CompileJob:
    # Resume from a previous successful compile job if available
    if COMPILE_JOB_CACHE.exists():
        job_id = COMPILE_JOB_CACHE.read_text().strip()
        print(f"[Step 2] Resuming from cached compile job {job_id} …")
        compile_job = qai_hub.get_job(job_id)
        if compile_job.get_status().success:
            print(f"[Step 2] Compile job already succeeded: {compile_job.url}\n")
            return compile_job
        print(f"[Step 2] Cached job did not succeed, re-compiling …")

    print(f"[Step 2] Uploading ONNX model to AI Hub …")
    model = qai_hub.upload_model(str(onnx_path))
    print(f"[Step 2] Model uploaded: {model}")

    device = qai_hub.Device(DEVICE_NAME)
    print(f"[Step 2] Submitting compile job → device: {DEVICE_NAME}, runtime: QNN context binary …")

    compile_job = qai_hub.submit_compile_job(
        model=model,
        device=device,
        name="distilbert-fp32-hexagon",
        options="--target_runtime qnn_context_binary --truncate_64bit_io",
        input_specs={
            "input_ids":      ((BATCH_SIZE, SEQ_LEN), "int64"),
            "attention_mask": ((BATCH_SIZE, SEQ_LEN), "int64"),
        },
    )
    print(f"[Step 2] Compile job submitted: {compile_job.url}\n")

    print("[Step 2] Waiting for compile job to complete (this may take a few minutes) …")
    compile_job.wait()

    status = compile_job.get_status()
    if not status.success:
        print(f"[Step 2] ERROR: Compile job failed — {status.message}")
        sys.exit(1)

    RESULTS_DIR.mkdir(exist_ok=True)
    COMPILE_JOB_CACHE.write_text(compile_job.job_id)
    print(f"[Step 2] Compile job SUCCEEDED: {compile_job.url}\n")
    return compile_job


# ---------------------------------------------------------------------------
# Step 3 — Profile on real Hexagon hardware
# ---------------------------------------------------------------------------
def profile_on_device(compile_job: qai_hub.CompileJob) -> qai_hub.ProfileJob:
    device = qai_hub.Device(DEVICE_NAME)
    print(f"[Step 3] Submitting profile job → device: {DEVICE_NAME} …")

    compiled_model = compile_job.get_target_model()
    profile_job = qai_hub.submit_profile_job(
        model=compiled_model,
        device=device,
        name="distilbert-fp32-hexagon-profile",
    )
    print(f"[Step 3] Profile job submitted: {profile_job.url}\n")

    print("[Step 3] Waiting for profile job to complete (this may take a few minutes) …")
    profile_job.wait()

    status = profile_job.get_status()
    if not status.success:
        print(f"[Step 3] ERROR: Profile job failed — {status.message}")
        sys.exit(1)

    print(f"[Step 3] Profile job SUCCEEDED: {profile_job.url}\n")
    return profile_job


# ---------------------------------------------------------------------------
# Step 4 — Parse results
# ---------------------------------------------------------------------------
def _safe(d: dict, *keys, default=None):
    """Safely traverse nested dict keys."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is default:
            return default
    return d


def parse_and_save(
    compile_job: qai_hub.CompileJob,
    profile_job: qai_hub.ProfileJob,
) -> dict:
    print("[Step 4] Downloading and parsing profile results …")
    profile = profile_job.download_profile()

    exec_summary = profile.get("execution_summary", {})
    exec_detail = profile.get("execution_detail", [])

    # --- Latency extraction ---
    # estimated_inference_time is in microseconds
    estimated_us = _safe(exec_summary, "estimated_inference_time", default=None)

    # Collect all timing samples if available
    inference_times_us: list = _safe(exec_summary, "inference_times", default=[])
    if not inference_times_us and estimated_us is not None:
        inference_times_us = [estimated_us]

    p50_us = float(np.percentile(inference_times_us, 50)) if inference_times_us else None
    p99_us = float(np.percentile(inference_times_us, 99)) if inference_times_us else None
    mean_us = float(np.mean(inference_times_us)) if inference_times_us else None

    # Cold start vs warm (first sample vs median of rest)
    cold_start_us = inference_times_us[0] if len(inference_times_us) >= 2 else None
    warm_median_us = float(np.median(inference_times_us[1:])) if len(inference_times_us) >= 2 else None

    # --- Memory ---
    peak_memory_bytes = _safe(exec_summary, "peak_memory_bytes", default=None)
    peak_memory_mb = peak_memory_bytes / 1024 / 1024 if peak_memory_bytes else None

    # --- Per-operator compute unit breakdown ---
    npu_count = cpu_count = gpu_count = 0
    op_breakdown: list[dict] = []

    for op in exec_detail:
        compute_unit = (op.get("compute_unit") or op.get("execution_target") or "").upper()
        op_name = op.get("op_type") or op.get("name") or "unknown"
        duration_us = op.get("execution_time") or op.get("duration_us") or 0

        entry = {
            "op": op_name,
            "compute_unit": compute_unit,
            "duration_us": duration_us,
        }
        op_breakdown.append(entry)

        if "NPU" in compute_unit or "HTP" in compute_unit or "DSP" in compute_unit:
            npu_count += 1
        elif "CPU" in compute_unit:
            cpu_count += 1
        elif "GPU" in compute_unit:
            gpu_count += 1

    total_ops = npu_count + cpu_count + gpu_count
    npu_pct = 100.0 * npu_count / total_ops if total_ops > 0 else 0.0

    # --- Comparison with laptop numbers ---
    def us_to_ms(us):
        return us / 1000.0 if us is not None else None

    laptop_best_p99_ms = min(v["p99_ms"] for v in LAPTOP_RESULTS.values() if isinstance(v, dict))
    hexagon_p99_ms = us_to_ms(p99_us)
    speedup = laptop_best_p99_ms / hexagon_p99_ms if hexagon_p99_ms else None

    # --- Build result dict ---
    results = {
        "model": "distilbert-base-uncased (FP32, ONNX opset 17)",
        "device": DEVICE_NAME,
        "runtime": "QNN context binary (Hexagon NPU)",
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "jobs": {
            "compile_url": compile_job.url,
            "profile_url": profile_job.url,
        },
        "latency_us": {
            "all_samples": inference_times_us,
            "p50": p50_us,
            "p99": p99_us,
            "mean": mean_us,
            "estimated": estimated_us,
            "cold_start": cold_start_us,
            "warm_median": warm_median_us,
        },
        "latency_ms": {
            "p50": us_to_ms(p50_us),
            "p99": us_to_ms(p99_us),
            "mean": us_to_ms(mean_us),
            "cold_start": us_to_ms(cold_start_us),
            "warm_median": us_to_ms(warm_median_us),
        },
        "memory": {
            "peak_bytes": peak_memory_bytes,
            "peak_mb": round(peak_memory_mb, 3) if peak_memory_mb else None,
        },
        "compute_unit_breakdown": {
            "npu_ops": npu_count,
            "cpu_ops": cpu_count,
            "gpu_ops": gpu_count,
            "total_ops": total_ops,
            "npu_coverage_pct": round(npu_pct, 1),
        },
        "op_breakdown": op_breakdown[:50],  # cap to first 50 ops in JSON
        "comparison_vs_laptop": {
            "laptop_platform": LAPTOP_RESULTS["platform"],
            "laptop_DEFAULT_p99_ms":           LAPTOP_RESULTS["DEFAULT"]["p99_ms"],
            "laptop_MEMORY_OPTIMIZED_p99_ms":  LAPTOP_RESULTS["MEMORY_OPTIMIZED"]["p99_ms"],
            "laptop_PARALLEL_p99_ms":          LAPTOP_RESULTS["PARALLEL"]["p99_ms"],
            "laptop_best_p99_ms":              laptop_best_p99_ms,
            "hexagon_p99_ms":                  hexagon_p99_ms,
            "speedup_vs_laptop_best":          round(speedup, 2) if speedup else None,
        },
        "raw_execution_summary": exec_summary,
    }

    # --- Save JSON ---
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[Step 4] JSON results saved → {RESULTS_JSON}")

    # --- Save human-readable TXT report ---
    _write_txt_report(results)
    print(f"[Step 4] Text report saved  → {RESULTS_TXT}\n")

    return results


def _write_txt_report(r: dict) -> None:
    lu = r["latency_us"]
    lm = r["latency_ms"]
    mem = r["memory"]
    cu = r["compute_unit_breakdown"]
    cmp = r["comparison_vs_laptop"]

    lines = [
        "=" * 70,
        "  DistilBERT FP32 — Qualcomm AI Hub Hexagon NPU Profile",
        "=" * 70,
        f"  Model   : {r['model']}",
        f"  Device  : {r['device']}",
        f"  Runtime : {r['runtime']}",
        f"  Input   : batch={r['batch_size']}, seq_len={r['seq_len']}",
        "",
        "  Job URLs",
        f"    Compile : {r['jobs']['compile_url']}",
        f"    Profile : {r['jobs']['profile_url']}",
        "",
        "  Latency (microseconds)",
        f"    Estimated   : {lu['estimated']} µs",
        f"    P50         : {lu['p50']} µs",
        f"    P99         : {lu['p99']} µs",
        f"    Mean        : {lu['mean']} µs",
        f"    Cold start  : {lu['cold_start']} µs",
        f"    Warm median : {lu['warm_median']} µs",
        "",
        "  Latency (milliseconds, for comparison)",
        f"    P50  : {lm['p50']} ms",
        f"    P99  : {lm['p99']} ms",
        f"    Mean : {lm['mean']} ms",
        "",
        "  Peak Memory",
        f"    {mem['peak_mb']} MB ({mem['peak_bytes']} bytes)",
        "",
        "  Compute Unit Breakdown (NPU vs CPU vs GPU)",
        f"    Total ops : {cu['total_ops']}",
        f"    NPU ops   : {cu['npu_ops']}  ({cu['npu_coverage_pct']}%)",
        f"    CPU ops   : {cu['cpu_ops']}",
        f"    GPU ops   : {cu['gpu_ops']}",
        "",
        "  Comparison vs Laptop (Apple M-series, ONNX Runtime 1.26)",
        f"    Laptop DEFAULT          p99: {cmp['laptop_DEFAULT_p99_ms']} ms",
        f"    Laptop MEMORY_OPTIMIZED p99: {cmp['laptop_MEMORY_OPTIMIZED_p99_ms']} ms",
        f"    Laptop PARALLEL         p99: {cmp['laptop_PARALLEL_p99_ms']} ms",
        f"    Laptop best p99         : {cmp['laptop_best_p99_ms']} ms",
        f"    Hexagon p99             : {cmp['hexagon_p99_ms']} ms",
        f"    Speedup vs laptop best  : {cmp['speedup_vs_laptop_best']}×",
        "",
        "  Key Question: Does DistilBERT get 100% NPU coverage?",
        f"    NPU coverage: {cu['npu_coverage_pct']}%  "
        f"({cu['npu_ops']} / {cu['total_ops']} ops on NPU)",
    ]

    if cu["cpu_ops"] > 0:
        lines.append(
            f"    WARNING: {cu['cpu_ops']} op(s) fell back to CPU "
            "(likely attention/layernorm/softmax)"
        )
    else:
        lines.append("    All ops executed on NPU — 100% NPU coverage achieved.")

    lines.append("=" * 70)

    with open(RESULTS_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Print summary to stdout
# ---------------------------------------------------------------------------
def print_summary(results: dict) -> None:
    cu = results["compute_unit_breakdown"]
    lm = results["latency_ms"]
    cmp = results["comparison_vs_laptop"]

    print("=" * 70)
    print("  SUMMARY — DistilBERT on Snapdragon 8 Elite (Hexagon NPU)")
    print("=" * 70)
    print(f"  Compile URL : {results['jobs']['compile_url']}")
    print(f"  Profile URL : {results['jobs']['profile_url']}")
    print()
    print(f"  Latency  → P50: {lm['p50']} ms  |  P99: {lm['p99']} ms  |  Mean: {lm['mean']} ms")
    print(f"  Memory   → Peak: {results['memory']['peak_mb']} MB")
    print()
    print(f"  NPU coverage : {cu['npu_coverage_pct']}%  ({cu['npu_ops']}/{cu['total_ops']} ops)")
    print(f"  CPU fallbacks: {cu['cpu_ops']} ops")
    print(f"  GPU ops      : {cu['gpu_ops']} ops")
    print()
    print(f"  Laptop best p99 : {cmp['laptop_best_p99_ms']} ms  (MEMORY_OPTIMIZED, Apple M-series)")
    print(f"  Hexagon p99     : {cmp['hexagon_p99_ms']} ms")
    print(f"  Speedup         : {cmp['speedup_vs_laptop_best']}×")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    configure_hub()

    onnx_path = export_onnx()
    compile_job = compile_for_hexagon(onnx_path)
    profile_job = profile_on_device(compile_job)
    results = parse_and_save(compile_job, profile_job)
    print_summary(results)


if __name__ == "__main__":
    main()
