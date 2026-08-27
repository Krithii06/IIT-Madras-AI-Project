"""Export a trained run to the self-contained bundle the backend serves.

The backend runs onnxruntime instead of torch. That keeps the deployed image around
150 MB rather than close to a gigabyte, which matters on a free CPU host with a
cold start budget. This script writes models/export/ with the graph, the class
mapping and the preprocessing settings needed to reproduce the eval transform.

    python -m src.inference.export --run-name mobilenet_leaf
"""

import argparse
import json
import shutil

import numpy as np
import torch

from src import config
from src.training.models import create_model


def _inline_external_data(onnx_path):
    """Rewrite an ONNX file so its weights live inside it, and drop the sidecars."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save(model, str(onnx_path), save_as_external_data=False)

    for sidecar in onnx_path.parent.iterdir():
        if sidecar.name.startswith(onnx_path.name) and sidecar.name != onnx_path.name:
            sidecar.unlink()


def export(run_name, out_dir=None, confidence_threshold=0.70, opset=None):
    run_dir = config.MODELS_DIR / run_name
    out_dir = out_dir or (config.MODELS_DIR / "export")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = json.load(open(run_dir / "train_config.json", encoding="utf-8"))
    mapping = json.load(open(run_dir / "class_mapping.json", encoding="utf-8"))

    model = create_model(train_cfg["arch"], len(mapping["classes"]), pretrained=False)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    onnx_path = out_dir / "model.onnx"

    # opset is left to the exporter (currently 20, which onnxruntime 1.20 loads).
    # Asking for an older one makes torch attempt a version downgrade that fails on
    # this graph and prints an alarming - but harmless - traceback before falling
    # back to the native opset anyway. Pin it only if a host needs an older runtime.
    extra = {"opset_version": opset} if opset else {}
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        # Batch stays dynamic so the same graph serves one upload or a batched test run.
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        **extra,
    )

    # torch's exporter writes the weights to a sidecar model.onnx.data once they pass
    # its size threshold. That loads fine as long as both files travel together, which
    # is exactly the assumption that breaks when someone copies the .onnx on its own.
    # Fold the weights back in so the served artefact is a single self-contained file.
    _inline_external_data(onnx_path)

    shutil.copy(run_dir / "class_mapping.json", out_dir / "class_mapping.json")

    preprocess = {
        "arch": train_cfg["arch"],
        "image_size": config.IMAGE_SIZE,
        "resize_before_crop": config.RESIZE_BEFORE_CROP,
        "normalize_mean": list(config.IMAGENET_MEAN),
        "normalize_std": list(config.IMAGENET_STD),
        "confidence_threshold": confidence_threshold,
        "intra_op_threads": 1,
        "source_run": run_name,
    }
    with open(out_dir / "preprocess.json", "w", encoding="utf-8") as fh:
        json.dump(preprocess, fh, indent=2)

    # Parity check: a silently diverging export would be very hard to notice later.
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    probe = torch.randn(4, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    with torch.no_grad():
        torch_out = model(probe).numpy()
    onnx_out = session.run(None, {"input": probe.numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"exported {train_cfg['arch']} -> {onnx_path} ({size_mb:.1f} MB)")
    print(f"max |torch - onnx| logit difference on random input: {max_diff:.2e}")
    if max_diff > 1e-3:
        raise SystemExit("ONNX export diverges from the torch model; refusing to ship it")

    latency_ms = _measure_latency(onnx_path)
    print(f"single-image CPU inference: {latency_ms:.1f} ms "
          f"(1 intra-op thread, as configured for the free tier)")

    info = {
        "arch": train_cfg["arch"],
        "source_run": run_name,
        "onnx_size_mb": round(size_mb, 2),
        "torch_checkpoint_mb": round((run_dir / "best_model.pt").stat().st_size / 1e6, 2),
        "max_logit_diff": max_diff,
        "single_image_cpu_ms": round(latency_ms, 1),
        "image_size": config.IMAGE_SIZE,
    }
    with open(out_dir / "export_info.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)

    return out_dir, info


def _measure_latency(onnx_path, runs=25):
    """Time one image through the graph the way the API will run it: one thread."""
    import time

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), options,
                                   providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    sample = np.random.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).astype(np.float32)

    for _ in range(5):
        session.run(None, {name: sample})
    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, {name: sample})
    return (time.perf_counter() - start) / runs * 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    args = parser.parse_args()
    export(args.run_name, confidence_threshold=args.confidence_threshold)
