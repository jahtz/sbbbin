# sbbbin
Generate binary versions of input images using SBB binarization. Pixelwise binarization with selectional auto-encoders in Keras. 

Forked from [sbb_binarization](https://github.com/qurator-spk/sbb_binarization/tree/transformer_model_integration)

## Setup
>[!NOTE]
> Tested Versions:
> - Python: `3.10.16`
> - CUDA: `11.7`
> - CuDNN: `8.1`

>[!IMPORTANT]
>The following setup process uses [PyEnv](https://github.com/pyenv/pyenv?tab=readme-ov-file#linuxunix)

1. Clone repository
	```shell
	git clone https://github.com/jahtz/sbbbin
	```

2. Create Virtual Environment
	```shell
	pyenv install 3.10.16
	pyenv virtualenv 3.10.16 sbbbin
	pyenv activate sbbbin
	```

3. Install sbbbin
    ```shell
    pip install sbbbin/.
    ```

4. (Optional) Select CUDA version
    ```shell
    export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
    ```

## Model
Download from here: 
https://qurator-data.de/sbb_binarization/ (2022-08-16)
and extract to the format below.
```
📂 sbb_hybrid_model
┗ 📂 saved_model
  ┣ 📂 assets
  ┣ 📂 variables
  ┣ 📜 keras_metadata.pb
  ┗ 📜 saved_model.pb
```
The `--model-dir` option should now be set to `/path/to/models/sbb_hybrid_model`.


## Usage
```shell
$ sbbbin --help
                                                                                          
 Usage: sbbbin [OPTIONS] IMAGES...                                                        
                                                                                          
 Pixelwise binarization with selectional auto-encoders in Keras.                          
 IMAGES: List of image file paths to process. Accepts individual files, wildcards, or     
 directories (with -g option for pattern matching).                                       
 For GPU computation, CUDA 11 is required (>=11.2)                                        
                                                                                          
╭─ Input ────────────────────────────────────────────────────────────────────────────────╮
│ *  IMAGES            (PATH) [required]                                                 │
│ *  --model-dir   -m  Path to directory containing the binarization model               │
│                      directories. See README.md for more information.                  │
│                      (DIRECTORY)                                                       │
│                      [required]                                                        │
│    --glob        -g  Glob pattern for matching images within directories. Only         │
│                      applicable when directories are passed in IMAGES.                 │
│                      (TEXT)                                                            │
│                      [default: *.png]                                                  │
│    --batch-size  -b  Reload the model after processing a specified number of images.   │
│                      Recommended for large input sets (>200 images) to manage memory   │
│                      usage and prevent unexpected terminations.                        │
│                      (INTEGER RANGE)                                                   │
│    --force-cpu       Force CPU computation. This should be slower but has better       │
│                      compatibility.                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ───────────────────────────────────────────────────────────────────────────────╮
│ --output        -o  Specify output directory for processed files. Defaults to the      │
│                     parent directory of each input file.                               │
│                     (DIRECTORY)                                                        │
│ --suffix        -s  Specify suffix for processed output images. (TEXT)                 │
│                     [default: .sbb.bin.png]                                            │
│ --no-overwrite      Prevent the overwriting of existing files if they already exist.   │
╰────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Help ─────────────────────────────────────────────────────────────────────────────────╮
│ --help         Show this message and exit.                                             │
│ --version      Show the version and exit.                                              │
│ --verbose  -v  Set verbosity level. `-v`: WARNING, `-vv`: INFO, `-vvv`: DEBUG.         │
│                (INTEGER)                                                               │
╰────────────────────────────────────────────────────────────────────────────────────────╯
```

## ZPD
Developed at Centre for [Philology and Digitality](https://www.uni-wuerzburg.de/en/zpd/) (ZPD), [University of Würzburg](https://www.uni-wuerzburg.de/en/).