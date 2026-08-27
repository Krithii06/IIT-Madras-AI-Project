"""Inference used by the backend and by the tests.

Deliberately depends on onnxruntime, Pillow and numpy only - not torch. Shipping
the training stack to a free CPU host would add hundreds of megabytes to the image
and slow cold starts for no benefit at inference time.

Because torchvision is not available here, the evaluation transform is
reimplemented with PIL and numpy. tests/test_inference.py checks that this
reimplementation matches the torchvision pipeline it replaces.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "export"


def resize_shorter_side(image, target):
    """Match torchvision Resize(int): scale so the shorter edge equals target."""
    width, height = image.size
    if width == height == target:
        return image
    if width < height:
        new_w = target
        new_h = max(1, round(height * target / width))
    else:
        new_h = target
        new_w = max(1, round(width * target / height))
    return image.resize((new_w, new_h), Image.BILINEAR)


def center_crop(image, size):
    width, height = image.size
    left = int(round((width - size) / 2.0))
    top = int(round((height - size) / 2.0))
    return image.crop((left, top, left + size, top + size))


class LeafPredictor:
    """Loads an exported ONNX model plus its class mapping and preprocessing config."""

    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)

        mapping = json.loads((self.model_dir / "class_mapping.json").read_text(encoding="utf-8"))
        self.classes = mapping["classes"]
        self.positive_class = mapping.get("positive_class")
        self.source_classes = mapping.get("source_classes", [])

        preprocess = json.loads((self.model_dir / "preprocess.json").read_text(encoding="utf-8"))
        self.image_size = preprocess["image_size"]
        self.resize = preprocess["resize_before_crop"]
        self.mean = np.array(preprocess["normalize_mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(preprocess["normalize_std"], dtype=np.float32).reshape(3, 1, 1)
        self.arch = preprocess.get("arch", "unknown")
        self.confidence_threshold = preprocess.get("confidence_threshold", 0.70)

        # Single-threaded sessions behave better on small shared free-tier CPUs than
        # letting onnxruntime spawn a thread per core.
        options = ort.SessionOptions()
        options.intra_op_num_threads = preprocess.get("intra_op_threads", 1)
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image):
        image = image.convert("RGB")
        image = resize_shorter_side(image, self.resize)
        image = center_crop(image, self.image_size)

        array = np.asarray(image, dtype=np.float32) / 255.0
        array = array.transpose(2, 0, 1)
        array = (array - self.mean) / self.std
        return array[None, ...]

    def predict(self, image):
        logits = self.session.run(None, {self.input_name: self.preprocess(image)})[0][0]
        probabilities = _softmax(logits)

        order = np.argsort(-probabilities)
        top = [
            {"label": self.classes[i], "confidence": round(float(probabilities[i]), 4)}
            for i in order
        ]
        best = top[0]
        return {
            "predicted_label": best["label"],
            "confidence": best["confidence"],
            "low_confidence": best["confidence"] < self.confidence_threshold,
            "confidence_threshold": self.confidence_threshold,
            "top_predictions": top,
            "message": self._message(best["label"], best["confidence"]),
        }

    def _message(self, label, confidence):
        """Wording is intentionally hedged: a softmax score is not a diagnosis."""
        if confidence < self.confidence_threshold:
            return ("The model has low confidence in this result. Try a clearer, "
                    "well-lit photo of a single leaf against a plain background.")
        if label == "healthy":
            return "The model finds no visible disease symptoms in this leaf image."
        return "The model finds visual patterns consistent with a diseased leaf."


def _softmax(x):
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / exp.sum()
