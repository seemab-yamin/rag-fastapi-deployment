import pytest

from app.config import ConfigError, load_config


def write(tmp_path, text):
    p = tmp_path / "config.yaml"
    p.write_text(text)
    return p


def test_defaults_when_file_empty(tmp_path):
    s = load_config(write(tmp_path, ""))
    assert s.model.task == "detection"


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_rejects_non_onnx(tmp_path):
    with pytest.raises(ConfigError, match="onnx"):
        load_config(write(tmp_path, "model:\n  path: models/m.pt\n"))


def test_rejects_unknown_key(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, "modle:\n  path: x.onnx\n"))


def test_rejects_bad_threshold(tmp_path):
    with pytest.raises(ConfigError, match="confidence_threshold"):
        load_config(write(tmp_path, "inference:\n  confidence_threshold: 2\n"))


def test_model_url_optional(tmp_path):
    assert load_config(write(tmp_path, "")).model.url is None
    s = load_config(write(tmp_path, "model:\n  url: https://example.com/m.onnx\n"))
    assert str(s.model.url) == "https://example.com/m.onnx"


def test_rejects_bad_model_url(tmp_path):
    with pytest.raises(ConfigError, match="model.url"):
        load_config(write(tmp_path, "model:\n  url: not-a-url\n"))


def test_mean_std_optional_and_validated(tmp_path):
    s = load_config(write(tmp_path, ""))
    assert s.model.mean is None and s.model.std is None
    s = load_config(
        write(tmp_path, "model:\n  mean: [0.485, 0.456, 0.406]\n  std: [0.229, 0.224, 0.225]\n")
    )
    assert s.model.std == (0.229, 0.224, 0.225)
    with pytest.raises(ConfigError, match="model.mean"):
        load_config(write(tmp_path, "model:\n  mean: [0.5, 0.5]\n"))
    with pytest.raises(ConfigError, match="model.std"):
        load_config(write(tmp_path, "model:\n  std: [0.2, 0, 0.2]\n"))


def test_model_sha256_optional_and_validated(tmp_path):
    assert load_config(write(tmp_path, "")).model.sha256 is None
    h = "a" * 64
    assert load_config(write(tmp_path, f"model:\n  sha256: {h}\n")).model.sha256 == h
    with pytest.raises(ConfigError, match="model.sha256"):
        load_config(write(tmp_path, "model:\n  sha256: abc123\n"))
