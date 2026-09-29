from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from skimage.io import imread, imsave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import transfer_learning as app


class FakeEncoder:
    def predict(self, images: np.ndarray, verbose: int = 0) -> np.ndarray:
        self.input_shape = images.shape
        return np.zeros((len(images), 14, 14, 512), dtype=np.float32)


class FakeDecoder:
    def predict(self, features: np.ndarray, verbose: int = 0) -> np.ndarray:
        return np.zeros((len(features), 224, 224, 2), dtype=np.float32)


def test_colorize_writes_rgb_output_with_mocked_models(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    source = np.zeros((96, 120, 3), dtype=np.uint8)
    source[..., 0] = 190
    imsave(input_dir / "sample.png", source)
    model_path = tmp_path / "decoder.keras"
    model_path.touch()
    encoder = FakeEncoder()

    monkeypatch.setattr(app, "build_encoder", lambda: encoder)
    monkeypatch.setattr(app, "load_model", lambda *args, **kwargs: FakeDecoder())
    app.colorize(input_dir, model_path, output_dir)

    result_path = output_dir / "0_sample.jpg"
    assert result_path.is_file()
    assert imread(result_path).shape == (224, 224, 3)
    assert encoder.input_shape == (1, 224, 224, 3)


def test_cli_routes_train_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["colorize", "train", "--data", "images", "--epochs", "2"],
    )
    monkeypatch.setattr(app, "train", lambda *args: calls.append(args))

    app.main()

    assert calls == [(Path("images"), Path("models/colorizer.keras"), 2, 16)]
