# VGG16 Image Colorization

A Python project that trains a convolutional decoder to predict Lab color channels
from frozen VGG16 features, then colorizes images in an input directory.

## Requirements

- Python 3.10–3.12
- [uv](https://docs.astral.sh/uv/)

Install dependencies and the project into a local virtual environment:

```sh
uv sync
```

Run commands through the environment with `uv run` (no manual activation is
needed):

```sh
uv run colorize train --data path/to/training-images --model models/colorizer.keras
uv run colorize colorize --input path/to/test-images --model models/colorizer.keras --output results
```

The training directory should contain images organized into one or more
subdirectories, as expected by Keras' directory image loader. The test directory
can contain supported image files directly. VGG16 ImageNet weights are downloaded
by Keras on first use; the network is required for that initial download.

The virtual environment can also be activated in a shell:

```sh
. .venv/bin/activate
```

On Windows PowerShell, use `.venv\\Scripts\\Activate.ps1`.
