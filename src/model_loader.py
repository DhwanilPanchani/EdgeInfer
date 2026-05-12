import os
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"


def load_model(model_path=None):
    import onnxruntime as ort

    MODELS_DIR.mkdir(exist_ok=True)
    if model_path is None:
        model_path = MODELS_DIR / "model.onnx"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        _download_model(model_path)

    print(f"Loading ONNX model from {model_path}")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    print("\nModel inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: shape={inp.shape}, dtype={inp.type}")

    print("Model outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: shape={out.shape}, dtype={out.type}")

    return session, model_path


def _download_model(model_path: Path):
    # Try optimum export first
    try:
        _export_via_optimum(model_path)
        return
    except Exception as e:
        print(f"Optimum export failed ({e}), falling back to preconverted ONNX download...")

    # Fallback: download preconverted ONNX from HuggingFace Hub
    _download_preconverted(model_path)


def _export_via_optimum(model_path: Path):
    from optimum.onnxruntime import ORTModelForSequenceClassification

    print("Exporting distilbert-base-uncased via optimum...")
    export_dir = model_path.parent / "optimum_export"
    model = ORTModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        export=True,
        cache_dir=str(model_path.parent),
    )
    model.save_pretrained(str(export_dir))

    exported = export_dir / "model.onnx"
    if not exported.exists():
        raise FileNotFoundError("Optimum did not produce model.onnx")

    import shutil
    shutil.copy(str(exported), str(model_path))
    print("Optimum export succeeded.")


def _download_preconverted(model_path: Path):
    import shutil
    from huggingface_hub import hf_hub_download

    print("Downloading preconverted ONNX model from HuggingFace Hub...")
    # optimum-maintained ONNX distilbert (sequence classification, ~67 MB)
    downloaded = Path(hf_hub_download(
        repo_id="optimum/distilbert-base-uncased-finetuned-sst-2-english",
        filename="model.onnx",
    ))
    if downloaded.resolve() != model_path.resolve():
        shutil.copy(str(downloaded), str(model_path))
    print(f"Model ready at {model_path}")
