"""Train and run the VGG16-based image colorization model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from skimage.color import gray2rgb, lab2rgb, rgb2lab
from skimage.io import imsave
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)

IMAGE_SIZE = (224, 224)


def build_encoder() -> tf.keras.Model:
    """Load the frozen VGG16 feature extractor through block5_conv3."""
    base = tf.keras.applications.VGG16(weights="imagenet", include_top=False)
    encoder = tf.keras.Model(
        inputs=base.input, outputs=base.get_layer("block5_conv3").output
    )
    encoder.trainable = False
    return encoder


def build_decoder() -> Sequential:
    decoder = Sequential(
        [
            tf.keras.Input(shape=(14, 14, 512)),
            Conv2D(256, 3, activation="relu", padding="same"),
            Conv2D(128, 3, activation="relu", padding="same"),
            UpSampling2D(2),
            Conv2D(64, 3, activation="relu", padding="same"),
            UpSampling2D(2),
            Conv2D(32, 3, activation="relu", padding="same"),
            UpSampling2D(2),
            Conv2D(16, 3, activation="relu", padding="same"),
            UpSampling2D(2),
            Conv2D(2, 3, activation="tanh", padding="same"),
        ],
        name="colorization_decoder",
    )
    decoder.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    return decoder


def load_training_data(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    generator = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        str(dataset_dir),
        target_size=IMAGE_SIZE,
        batch_size=32,
        class_mode=None,
        shuffle=False,
    )
    images = np.concatenate([generator[i] for i in range(len(generator))], axis=0)
    labs = np.asarray([rgb2lab(image) for image in images], dtype=np.float32)
    luminance = np.repeat(labs[..., :1], 3, axis=-1)
    chrominance = labs[..., 1:] / 128.0
    return luminance, chrominance


def train(dataset_dir: Path, model_path: Path, epochs: int, batch_size: int) -> None:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Training image directory does not exist: {dataset_dir}"
        )
    luminance, chrominance = load_training_data(dataset_dir)
    encoder = build_encoder()
    vgg_input = tf.keras.applications.vgg16.preprocess_input(luminance * 255.0)
    features = encoder.predict(vgg_input, verbose=0)
    decoder = build_decoder()
    decoder.fit(features, chrominance, epochs=epochs, batch_size=batch_size, verbose=1)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    decoder.save(model_path)
    print(f"Saved trained decoder to {model_path}")


def colorize(test_dir: Path, model_path: Path, output_dir: Path) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(f"Decoder model does not exist: {model_path}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test image directory does not exist: {test_dir}")

    encoder = build_encoder()
    decoder = load_model(model_path, compile=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        path
        for path in test_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    for index, path in enumerate(image_paths):
        image = img_to_array(load_img(path, target_size=IMAGE_SIZE)) / 255.0
        lab = rgb2lab(image)
        luminance = lab[..., 0]
        encoder_input = gray2rgb(luminance)[None, ...] * 255.0
        encoder_input = tf.keras.applications.vgg16.preprocess_input(encoder_input)
        features = encoder.predict(encoder_input, verbose=0)
        ab = decoder.predict(features, verbose=0)[0] * 128.0

        result = np.empty((*IMAGE_SIZE, 3), dtype=np.float32)
        result[..., 0] = luminance
        result[..., 1:] = ab
        rgb = np.clip(lab2rgb(result) * 255.0, 0, 255).astype(np.uint8)
        destination = output_dir / f"{index}_{path.stem}.jpg"
        imsave(destination, rgb)
        print(f"Saved {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser(
        "train", help="Train a decoder from images in class subdirectories"
    )
    train_parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Root directory with training image subdirectories",
    )
    train_parser.add_argument(
        "--model", type=Path, default=Path("models/colorizer.keras")
    )
    train_parser.add_argument("--epochs", type=int, default=2000)
    train_parser.add_argument("--batch-size", type=int, default=16)

    predict_parser = commands.add_parser(
        "colorize", help="Colorize images using a trained decoder"
    )
    predict_parser.add_argument(
        "--input", type=Path, required=True, help="Directory of images to colorize"
    )
    predict_parser.add_argument(
        "--model", type=Path, default=Path("models/colorizer.keras")
    )
    predict_parser.add_argument("--output", type=Path, default=Path("results"))

    args = parser.parse_args()
    if args.command == "train":
        if args.epochs < 1 or args.batch_size < 1:
            parser.error("--epochs and --batch-size must be positive")
        train(args.data, args.model, args.epochs, args.batch_size)
    else:
        colorize(args.input, args.model, args.output)


if __name__ == "__main__":
    main()
