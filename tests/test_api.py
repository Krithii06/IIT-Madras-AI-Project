"""Endpoint tests: the happy path plus the failure modes a public upload form gets.

Anything reachable from the internet gets sent junk, so the error cases here matter
as much as the successful prediction.
"""

import io

from PIL import Image

from tests.conftest import make_image_bytes


def test_health_reports_the_model_is_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_root_lists_the_endpoints(client):
    body = client.get("/").json()
    assert "/predict" in body["endpoints"]


def test_model_info_describes_the_deployed_model(client):
    body = client.get("/model-info").json()
    assert body["num_classes"] == len(body["classes"])
    assert body["input_size"] > 0
    assert 0.0 < body["confidence_threshold"] <= 1.0
    assert body["architecture"]


def test_predict_on_a_real_leaf_image(client, sample_leaf_path):
    with open(sample_leaf_path, "rb") as fh:
        response = client.post("/predict",
                               files={"file": ("leaf.jpg", fh.read(), "image/jpeg")})

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_label"] in ("healthy", "diseased")
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["inference_ms"] > 0
    assert len(body["top_predictions"]) == 2


def test_predict_accepts_png(client):
    response = client.post(
        "/predict",
        files={"file": ("leaf.png", make_image_bytes(fmt="PNG"), "image/png")},
    )
    assert response.status_code == 200


def test_unsupported_file_type_is_rejected(client):
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"this is not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_missing_file_is_rejected(client):
    assert client.post("/predict").status_code == 422


def test_empty_file_is_rejected(client):
    response = client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400


def test_corrupt_image_is_rejected(client):
    """Correct content type, bytes that are not a decodable image."""
    response = client.post(
        "/predict",
        files={"file": ("broken.jpg", b"\xff\xd8\xff\xe0garbage-not-a-jpeg", "image/jpeg")},
    )
    assert response.status_code == 400


def test_truncated_image_is_rejected(client):
    payload = make_image_bytes()
    response = client.post(
        "/predict",
        files={"file": ("cut.jpg", payload[: len(payload) // 3], "image/jpeg")},
    )
    assert response.status_code == 400


def test_oversized_upload_is_rejected(client):
    from app.routes.predict import MAX_UPLOAD_BYTES

    payload = b"\xff\xd8\xff\xe0" + b"0" * (MAX_UPLOAD_BYTES + 1024)
    response = client.post("/predict",
                           files={"file": ("big.jpg", payload, "image/jpeg")})
    assert response.status_code == 413


def test_errors_do_not_leak_internals(client):
    """Clients get a readable message, never a stack trace or a file path."""
    response = client.post("/predict",
                           files={"file": ("broken.jpg", b"nope", "image/jpeg")})
    detail = response.json()["detail"]
    assert "Traceback" not in detail
    assert ".py" not in detail


def test_a_large_but_legal_image_still_predicts(client):
    """Phone photos are much bigger than the 256px training images."""
    buffer = io.BytesIO()
    Image.new("RGB", (2000, 1500), (70, 120, 80)).save(buffer, format="JPEG", quality=70)
    response = client.post(
        "/predict",
        files={"file": ("phone.jpg", buffer.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 200
