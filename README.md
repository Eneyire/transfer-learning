# Transfer Learning - VGG16 Image Colourisation

A Python project that trains a convolutional decoder to predict Lab colour
channels from frozen VGG16 features, then colourises images in an input
directory.

## Requirements

- Python 3.10–3.12
- [uv](https://docs.astral.sh/uv/)

Install the project and its dependencies into a local virtual environment:

```sh
uv sync
```

Run commands through the environment with `uv run` (no manual activation is
needed):

```sh
uv run colorize train --data path/to/training-images --model models/colorizer.keras
uv run colorize colorize --input path/to/test-images --model models/colorizer.keras --output results
```

The training directory should contain images organised into one or more
subdirectories, as expected by Keras' directory image loader. The colourisation
input directory should contain supported image files directly (`.jpg`, `.jpeg`,
`.png`, `.bmp` or `.webp`). Images are resized to 224 × 224 pixels. The trained
decoder is saved to `models/colorizer.keras` by default; colourised images are
written to `results` by default, with names prefixed by an index.

Training defaults to 2,000 epochs and a batch size of 16. Set `--epochs` and
`--batch-size` to change these values. VGG16 ImageNet weights are downloaded by
Keras on first use, so a network connection is required for that initial
download.

The virtual environment can also be activated in a shell:

```sh
. .venv/bin/activate
```

On Windows PowerShell, use `.venv\Scripts\Activate.ps1`.

Run the test suite with pytest:

```sh
uv run pytest
```
