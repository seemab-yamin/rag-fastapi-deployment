from types import SimpleNamespace

import numpy as np
import pytest

from app.services import onnx_runtime as rt
from app.services.onnx_runtime import ModelLoadError, TensorSpec

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"


@pytest.mark.parametrize(
    "requested, available, expected",
    [
        ([CUDA, CPU], [CPU], [CPU]),
        ([CUDA, CPU], [CUDA, CPU], [CUDA, CPU]),
        ([CUDA], ["CoreMLExecutionProvider", CPU], [CPU]),
        (["TensorrtExecutionProvider", CUDA], [CUDA, CPU], [CUDA, CPU]),
    ],
    ids=["cpu-build", "gpu-build", "none-available", "next-option"],
)
def test_resolve_providers_falls_back(requested, available, expected):
    assert rt.resolve_providers(requested, available) == expected


class _Session:
    """Fake InferenceSession that records the providers it was given."""

    seen = {}

    def __init__(self, path, providers):
        _Session.seen["providers"] = providers

    def get_inputs(self):
        return []

    def get_outputs(self):
        return []

    def get_providers(self):
        return [CPU]


def test_missing_gpu_falls_back_to_cpu_with_warning(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(rt.ort, "get_available_providers", lambda: [CPU])
    monkeypatch.setattr(rt.ort, "InferenceSession", _Session)
    model = rt.load_model(
        SimpleNamespace(path=tmp_path / "m.onnx"), SimpleNamespace(providers=[CUDA, CPU])
    )

    assert _Session.seen["providers"] == [CPU]
    assert model.info()["providers"] == [CPU]
    assert model.info()["requested_providers"] == [CUDA, CPU]
    assert "skipping" in caplog.text and CUDA in caplog.text


def test_warns_when_session_drops_requested_provider(monkeypatch, tmp_path, caplog):
    # GPU build, but no visible GPU: ONNX Runtime silently runs on CPU
    monkeypatch.setattr(rt.ort, "get_available_providers", lambda: [CUDA, CPU])
    monkeypatch.setattr(rt.ort, "InferenceSession", _Session)
    rt.load_model(SimpleNamespace(path=tmp_path / "m.onnx"), SimpleNamespace(providers=[CUDA, CPU]))
    assert "running on" in caplog.text


def test_invalid_onnx_file(tmp_path):
    bad = tmp_path / "m.onnx"
    bad.write_bytes(b"not a model")
    with pytest.raises(ModelLoadError, match="Could not load"):
        rt.load_model(
            SimpleNamespace(path=bad),
            SimpleNamespace(providers=["CPUExecutionProvider"]),
        )


def test_unsupported_input_type():
    meta = SimpleNamespace(name="x", shape=[1], type="tensor(string)")
    with pytest.raises(ModelLoadError, match="Unsupported"):
        rt._spec(meta, strict=True)


class FakeSession:
    def run(self, names, feed):
        self.feed = feed
        return [np.zeros((1, 5))]

    def get_providers(self):
        return ["CPUExecutionProvider"]


def make_model(session=None):
    inputs = [
        TensorSpec("input_ids", np.int64, ("batch", "seq")),
        TensorSpec("attention_mask", np.int64, ("batch", "seq")),
    ]
    return rt.OnnxModel(
        session or FakeSession(), inputs, [TensorSpec("out", None, (1, 5))]
    )


def test_multi_input_and_dtype_cast():
    session = FakeSession()
    model = make_model(session)
    model.run({"input_ids": np.ones((1, 4)), "attention_mask": np.ones((1, 4))})
    assert session.feed["input_ids"].dtype == np.int64


def test_wrong_input_names_rejected():
    with pytest.raises(ValueError, match="expects inputs"):
        make_model().run({"wrong": np.zeros((1, 4))})


def test_info_is_json_safe():
    import json

    json.dumps(make_model().info())
