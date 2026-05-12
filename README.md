# EdgeInfer — Generative AI Inference Engine for Edge Targets

## Overview
EdgeInfer benchmarks transformer model inference on CPU using ONNX Runtime, targeting the performance characteristics of edge SoC hardware (Raspberry Pi, Jetson, mobile NPUs). It downloads DistilBERT in ONNX format, runs a configurable number of inference passes, and reports latency percentiles alongside memory and CPU profiles. The C++ stub demonstrates how the same ONNX Runtime C++ SDK would be integrated in a production edge deployment.

## Architecture
```
src/
├── model_loader.py   # Downloads/exports model to ONNX, returns InferenceSession
├── benchmarker.py    # Runs n passes across three session configs, computes percentiles + regressions
├── profiler.py       # Measures peak heap memory (tracemalloc) and CPU usage (psutil)
└── visualizer.py     # Generates latency histogram and P50/P99 bar chart via matplotlib
```
`run_benchmark.py` is the top-level entrypoint that orchestrates all modules and writes `results/results.json` and `results/benchmark_chart.png`.

## Benchmarking Methodology
- **Latency (P50/P90/P99):** `time.perf_counter` wraps each inference call. 100 runs after 5 warm-up passes remove JIT/cache cold-start noise. Percentiles are computed with NumPy to show tail latency under realistic load.
- **Session configs:** Three ONNX Runtime configurations are compared — DEFAULT (standard), MEMORY_OPTIMIZED (`enable_mem_pattern=True`, `inter_op_num_threads=1`), and PARALLEL (`intra_op_num_threads=4`, `ExecutionMode.ORT_PARALLEL`).
- **Regression detection:** Any individual run exceeding 2× the median latency for that config is flagged; `regression_count` and `regression_runs` (0-indexed run numbers) are included per config.
- **Memory (tracemalloc):** Python's built-in `tracemalloc` captures peak heap allocation starting before `InferenceSession.__init__`, reported in MB. Native C++ allocations are not visible to tracemalloc.
- **CPU (psutil):** `psutil.Process.cpu_percent` is sampled after each of 20 inference runs and normalised by logical core count to single-core-equivalent percentage.

## Sample Results
Results from `python run_benchmark.py --runs 100` on Apple M-series CPU, DistilBERT SST-2 (ONNX), batch=1, seq=128:

| Config | P50 (ms) | P90 (ms) | P99 (ms) | Mean (ms) | Regressions |
|---|---|---|---|---|---|
| DEFAULT | 3.040 | 3.186 | 3.866 | 3.087 | 0 |
| MEMORY_OPTIMIZED | 3.026 | 3.202 | 3.762 | 3.075 | 0 |
| PARALLEL | 3.737 | 4.331 | 5.966 | 3.830 | 1 |

Peak memory (tracemalloc): 0.002 MB · Avg CPU (normalised): 46.3%

> PARALLEL is slower on this model because the graph is too small to amortise thread-coordination overhead. MEMORY_OPTIMIZED achieves the lowest tail latency with no regressions.

See `results/results.json` and `results/benchmark_chart.png` for full output.

## Build

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the benchmark (downloads model on first run)
python run_benchmark.py --runs 100

# Options
python run_benchmark.py --runs 200 --output-dir my_results/
python run_benchmark.py --model-path /path/to/custom.onnx --no-chart
```

## CMake Build

```bash
mkdir build && cd build
cmake ..
cmake --build .
./edge_infer_stub
```

## Tech Stack
| Layer | Technology |
|-------|-----------|
| Inference runtime | ONNX Runtime 1.26 (CPU EP) |
| Model export | Hugging Face Optimum 2.1 |
| Model hub | huggingface-hub 1.14 |
| Tokenisation | Transformers 5.8 |
| Profiling | tracemalloc, psutil 7.2 |
| Numerics | NumPy 2.4 |
| Edge C++ runtime (stub) | ONNX Runtime C++ SDK / CMake 3.20 |
