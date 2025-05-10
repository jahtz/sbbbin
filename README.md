# sbbbin

Generate binary versions of input images using SBB binarization. Pixelwise binarization with selectional auto-encoders in Keras.

Forked from [sbb_binarization](https://github.com/qurator-spk/sbb_binarization/tree/transformer_model_integration).<br>
Updated with the version from [eynollah](https://github.com/qurator-spk/eynollah/blob/main/src/eynollah/sbb_binarize.py).

## Model

>[!TIP]
> To use a model with `sbbbin`, provide the path to the directory containing `saved_model.pb`

Download from [here](https://huggingface.co/SBB/eynollah-binarization) (huggingface)

## Docker

### Use available image

```shell
docker pull ghcr.io/jahtz/sbbbin:latest
```

```shell
docker run --rm -it --gpus all -v $(pwd):/data ghcr.io/jahtz/sbbbin:latest IMAGES... [OPTIONS]
```

### Build from source

1. Clone repository

    ```shell
    git clone https://github.com/jahtz/sbbbin
    ```

2. Build the image

    ```shell
    docker build -t sbbbin .
    ```

3. Run with

    ```shell
    docker run --rm -it --gpus all -v $(pwd):/data sbbbin IMAGES... [OPTIONS]
    ```

## PIP

>[!NOTE]
> Python: `3.8-3.11`<br>
> CUDA: [version table](https://www.tensorflow.org/install/source#gpu)

>[!TIP]
> Use a virtual enviroment, e.g. with [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#linuxunix).

1. Clone repository

    ```shell
    git clone https://github.com/jahtz/sbbbin
    ```

2. Install

    ```shell
    pip install sbbbin/.
    ```

3. Set `LD_LIBRARY_PATH` to the correct CUDA runtime

    ```shell
    export LD_LIBRARY_PATH="/usr/local/<version>/lib64:$LD_LIBRARY_PATH"
    ```

4. Run

    ```shell
    sbbbin IMAGES... [OPTIONS]
    ```

## Usage

```txt
$ sbbbin --help
Usage: sbbbin [OPTIONS] IMAGES...

  Pixelwise binarization with selectional auto-encoders in Keras.

  IMAGES: List of image file paths to process. Accepts individual files, glob
  wildcards, or directories.

Options:
  --help                  Show this message and exit.
  --version               Show the version and exit.
  -o, --output DIRECTORY  Specify output directory for processed files.
                          Defaults to the parent directory of each input file.
  -m, --model DIRECTORY   Path to directory containing the binarization
                          'saved_model' directory. See README.md for more
                          information.  [required]
  -s, --suffix TEXT       Specify suffix for output images.  [default:
                          .sbb.bin.png]
  --gpu / --cpu           Select computation device. Use '--gpu' for CUDA
                          acceleration (recommended).  [default: gpu]
```

## ZPD

Developed at Centre for [Philology and Digitality](https://www.uni-wuerzburg.de/en/zpd/) (ZPD), [University of Würzburg](https://www.uni-wuerzburg.de/en/).
