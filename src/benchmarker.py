import time
import numpy as np
import onnxruntime as ort

_CONFIGS = {
    "DEFAULT": {},
    "MEMORY_OPTIMIZED": {"enable_mem_pattern": True, "inter_op_num_threads": 1},
    "PARALLEL": {"intra_op_num_threads": 4, "execution_mode": "parallel"},
}


def _build_session(model_path: str, config_name: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    cfg = _CONFIGS[config_name]
    if cfg.get("enable_mem_pattern"):
        opts.enable_mem_pattern = True
    if "inter_op_num_threads" in cfg:
        opts.inter_op_num_threads = cfg["inter_op_num_threads"]
    if "intra_op_num_threads" in cfg:
        opts.intra_op_num_threads = cfg["intra_op_num_threads"]
    if cfg.get("execution_mode") == "parallel":
        opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])


def _make_dummy_inputs(session: ort.InferenceSession) -> dict:
    inputs = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        if len(shape) == 2 and shape[1] in (0, None, "sequence_length"):
            shape[1] = 128
        if "int" in inp.type:
            inputs[inp.name] = np.ones(shape, dtype=np.int64)
        else:
            inputs[inp.name] = np.ones(shape, dtype=np.float32)
    return inputs


def _run_config(model_path: str, config_name: str, n_runs: int) -> dict:
    session = _build_session(model_path, config_name)
    dummy_inputs = _make_dummy_inputs(session)
    output_names = [o.name for o in session.get_outputs()]

    for _ in range(5):
        session.run(output_names, dummy_inputs)

    latencies_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(output_names, dummy_inputs)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms)
    median = float(np.median(arr))
    threshold = 2.0 * median
    regression_runs = [i for i, v in enumerate(latencies_ms) if v > threshold]

    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "mean_ms": float(arr.mean()),
        "regression_count": len(regression_runs),
        "regression_runs": regression_runs,
        # kept in-memory for the visualizer; stripped before JSON serialization
        "latencies_ms": latencies_ms,
    }


def run_benchmark(model_path, n_runs: int = 100) -> dict:
    configs_results = {}
    for name in _CONFIGS:
        print(f"[{name}] running {n_runs} passes...")
        configs_results[name] = _run_config(str(model_path), name, n_runs)

    _print_comparison(configs_results, n_runs)
    return {"n_runs": n_runs, "configs": configs_results}


def _print_comparison(configs_results: dict, n_runs: int):
    print(f"\n{'Config':<20} {'P50':>8} {'P90':>8} {'P99':>8} {'Mean':>8} {'Regressions':>12}")
    print("-" * 68)
    for name, r in configs_results.items():
        print(f"{name:<20} {r['p50_ms']:>8.3f} {r['p90_ms']:>8.3f} {r['p99_ms']:>8.3f} {r['mean_ms']:>8.3f} {r['regression_count']:>12}")
    print("-" * 68)
    print("(times in ms)\n")
