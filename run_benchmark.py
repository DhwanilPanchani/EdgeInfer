import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.model_loader import load_model
from src.benchmarker import run_benchmark
from src.profiler import profile_memory, profile_cpu
from src.visualizer import generate_chart


def _parse_args():
    parser = argparse.ArgumentParser(description="EdgeInfer — ONNX inference benchmark")
    parser.add_argument("--runs", type=int, default=100, help="Number of inference passes per config (default: 100)")
    parser.add_argument("--model-path", type=str, default=None, help="Path to existing .onnx file (default: auto-download)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for results.json and chart (default: results/)")
    parser.add_argument("--no-chart", action="store_true", help="Skip generating benchmark_chart.png")
    return parser.parse_args()


def _strip_latencies(bench: dict) -> dict:
    """Return a copy of bench with raw latency arrays removed for JSON output."""
    cleaned = copy.deepcopy(bench)
    for cfg in cleaned.get("configs", {}).values():
        cfg.pop("latencies_ms", None)
    return cleaned


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=== EdgeInfer Benchmark ===\n")

    session, model_path = load_model(args.model_path)
    bench = run_benchmark(model_path, n_runs=args.runs)
    mem = profile_memory(model_path)
    cpu = profile_cpu(session, n_runs=20)

    if not args.no_chart:
        generate_chart(bench, output_dir)

    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": "distilbert-base-uncased (ONNX)",
        "runtime": "onnxruntime-cpu",
        **_strip_latencies(bench),
        **mem,
        **cpu,
    }

    out_path = output_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── summary print ────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"  Model       : {results['model']}")
    for name, r in results["configs"].items():
        print(f"  [{name}]  P50={r['p50_ms']:.3f} ms  P99={r['p99_ms']:.3f} ms  regressions={r['regression_count']}")
    print(f"  Peak Memory : {results['peak_memory_mb']} MB")
    print(f"  Avg CPU     : {results['avg_cpu_percent']}%")
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
