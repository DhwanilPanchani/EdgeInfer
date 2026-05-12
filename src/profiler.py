import tracemalloc
import numpy as np
import psutil
import os


def _make_dummy_inputs(session):
    inputs = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        if len(shape) == 2 and shape[1] in (0, None):
            shape[1] = 128
        if "int" in inp.type:
            inputs[inp.name] = np.ones(shape, dtype=np.int64)
        else:
            inputs[inp.name] = np.ones(shape, dtype=np.float32)
    return inputs


def profile_memory(model_path: str) -> dict:
    import onnxruntime as ort

    # tracemalloc must start before InferenceSession.__init__ so that model
    # weight allocation is captured inside the measurement window.
    tracemalloc.start()
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    dummy_inputs = _make_dummy_inputs(session)
    output_names = [o.name for o in session.get_outputs()]
    session.run(output_names, dummy_inputs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    print(f"Peak memory (tracemalloc): {peak_mb:.2f} MB")
    return {"peak_memory_mb": round(peak_mb, 3)}


def profile_cpu(session, n_runs: int = 20) -> dict:
    dummy_inputs = _make_dummy_inputs(session)
    output_names = [o.name for o in session.get_outputs()]
    proc = psutil.Process(os.getpid())
    core_count = psutil.cpu_count(logical=True) or 1

    cpu_samples = []
    for _ in range(n_runs):
        proc.cpu_percent(interval=None)  # reset interval
        session.run(output_names, dummy_inputs)
        cpu_samples.append(proc.cpu_percent(interval=None))

    # Normalise to single-core equivalent (psutil reports aggregate across all cores)
    avg_cpu_raw = float(np.mean(cpu_samples)) if cpu_samples else 0.0
    avg_cpu = round(avg_cpu_raw / core_count, 2)
    print(f"Avg CPU during inference ({n_runs} runs): {avg_cpu:.1f}% (normalised, {core_count} logical cores)")
    return {"avg_cpu_percent": avg_cpu}
