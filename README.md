# SBB Binarization
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

Pixelwise image binarization with selectional auto-encoders.<br>
This repository is a PyTorch-based adaptation of the original SBB Binarization code.

## Origin
This repository is a fork of [sbb_binarization](https://github.com/qurator-spk/sbb_binarization/tree/transformer_model_integration).<br>
Parts of that project were later integrated into [eynollah](https://github.com/qurator-spk/eynollah/blob/main/src/eynollah/sbb_binarize.py), which also influenced this codebase.<br>
Additionally, this repository includes code from [sbb_binarizer_pytorch_converter](https://github.com/twphl/sbb_binarizer_pytorch_converter) to provide a PyTorch-based implementation.

## Setup

> [!NOTE]
> The setup process is configured for [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/jahtz/sbbbin
```

```bash
uv tool install ./sbbbin --torch-backend <backend>
```
See `$ uv tool install --help` for possible backends.


## Usage
```bash
sbbbin [OPTIONS] IMAGES...
```

```text
$ sbbbin --help
Usage: sbbbin [OPTIONS] IMAGES...
  Pixelwise binarization with selectional auto-encoders using the SBB
  binarization algorithm

  IMAGES: List of image file paths to process.

Options:
  --help                          Show this message and exit.
  --version                       Show the version and exit.
  -m, --model FILE                Path to the trained PyTorch (.pth) model.
                                  [required]
  -o, --output DIRECTORY          Specify output directory for processed
                                  files. Defaults to the parent directory of
                                  each input file.
  -s, --suffix TEXT               Specify suffix for output images.  [default:
                                  .sbb.bin.png]
  -d, --device [auto|cpu|cuda]    Select the computing device. "cuda" requires
                                  a bundled CUDA/PyTorch version.  [default:
                                  auto]
  --logging [ERROR|WARNING|INFO]  Set logging level.  [default: ERROR]

  Developed at Centre for Philology and Digitality (ZPD), University of
  Würzburg
```

## Model
The Tensorflow model can be downloaded [huggingface](https://huggingface.co/SBB/eynollah-binarization/tree/main).<br>
To convert the model from Tensorflow to PyTorch, use [sbb_binarizer_pytorch_converter](https://github.com/twphl/sbb_binarizer_pytorch_converter).

## ZPD
Developed at Centre for [Philology and Digitality](https://www.uni-wuerzburg.de/en/zpd/) (ZPD), [University of Würzburg](https://www.uni-wuerzburg.de/en/).
