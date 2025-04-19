# Copyright 2025 Janik Haitz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import rich_click as click
from rich.logging import RichHandler
from rich.progress import (Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn, 
                           SpinnerColumn)

from sbbbin import SbbBinarizer


__version__ = "0.2.0"
__prog__ = "template"
__footer__ = "Developed at Centre for Philology and Digitality (ZPD), University of Würzburg"


logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(markup=True)]
)
logger = logging.getLogger("sbbbin")


click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.MAX_WIDTH = 90
click.rich_click.RANGE_STRING = ""
click.rich_click.SHOW_METAVARS_COLUMN = False
click.rich_click.APPEND_METAVARS_HELP = True
click.rich_click.FOOTER_TEXT = __footer__
click.rich_click.OPTION_GROUPS = {
    "sbbbin": [
        {
            "name": "Input",
            "options": ["images", "--model-dir", "--glob", "--batch-size", "--force-cpu"]
        },
        {
            "name": "Output",
            "options": ["--output", "--suffix", "--no-overwrite"]
        },
        {
            "name": "Help",
            "options": ["--help", "--version", "--verbose"],
        },
    ]
}
progress_bar = Progress(
    TextColumn("[progress.description]{task.description}"), 
    BarColumn(), 
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), 
    MofNCompleteColumn(), 
    TextColumn("•"), 
    TimeElapsedColumn(), 
    TextColumn("•"), 
    TimeRemainingColumn(), 
    TextColumn("• {task.fields[filename]}")
)
progress_spinner = Progress(
    SpinnerColumn(), 
    TextColumn("[progress.description]{task.description}"), 
    transient=True
)

def callback_paths(ctx, param, value: Optional[list[str]]) -> list[Path]:
    """ Parse a list of click paths to a list of pathlib Path objects """
    return [] if value is None else list([Path(p) for p in value])

def callback_path(ctx, param, value: Optional[str]) -> Optional[Path]:
    """ Parse a click path to a pathlib Path object """
    return None if value is None else Path(value)

def callback_suffix(ctx, param, value: Optional[str]) -> Optional[str]:
    """ Parses a string to a valid suffix """
    return None if value is None else (value if value.startswith('.') else f".{value}")

def callback_logging(ctx, param, value: Optional[int]) -> int:
    """ Returns the logging level based on a verbosity counter (`0`: ERROR, `1`: WARNING, `2`: INFO, `>2`: DEBUG) """
    return 40 if value is None else 40 - (min(3, value) * 10)

def expand_paths(paths: Union[Path, list[Path]], glob: str = '*') -> list[Path]:
    """Expands a list of paths by unpacking directories."""
    result = []
    if isinstance(paths, list):
        for path in paths:
            if path.is_dir():
                result.extend([p for p in path.glob(glob) if p.is_file()])
            else:
                result.append(path)
    elif isinstance(paths, Path):
        if paths.is_dir():
            result.extend([p for p in paths.glob(glob) if p.is_file()])
        else:
            result.append(paths)
    return sorted(result)


@click.command()
@click.help_option("--help")
@click.version_option(__version__,
                      "--version",
                      prog_name=__prog__,
                      message=f"{__prog__} v{__version__}\n{__footer__}")
@click.argument("images",
                type=click.Path(exists=True, dir_okay=True, file_okay=True, resolve_path=True),
                callback=callback_paths, required=True, nargs=-1)
@click.option("-g", "--glob", "glob",
              help="Glob pattern for matching images within directories. "
                   "Only applicable when directories are passed in IMAGES.",
              type=click.STRING, default="*.png", required=False, show_default=True)
@click.option("-o", "--output", "output",
              help="Specify output directory for processed files. Defaults to the parent directory of each input file.",
              type=click.Path(exists=False, dir_okay=True, file_okay=False, resolve_path=True),
              callback=callback_path, required=False)
@click.option("-s", "--suffix", "suffix",
              help="Specify suffix for processed output images.",
              type=click.STRING, callback=callback_suffix, required=False, default=".sbb.bin.png", show_default=True)
@click.option("-m", "--model-dir", "model_dir",
              help="Path to directory containing the binarization model directories. "
                   "See README.md for more information.",
              type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True),
              callback=callback_path, required=True)
@click.option("-b", "--batch-size", "batch_size",
              help="Reload the model after processing a specified number of images. Recommended for large input sets "
                   "(>200 images) to manage memory usage and prevent unexpected terminations.",
              type=click.IntRange(min=1), required=False)
@click.option("--no-overwrite", "no_overwrite",
              help="Prevent the overwriting of existing files if they already exist.",
              type=click.BOOL, is_flag=True, required=False)
@click.option("--force-cpu", "force_cpu",
              help="Force CPU computation. This should be slower but has better compatibility.",
              type=click.BOOL, is_flag=True, required=False)
@click.option("-v", "--verbose", "verbosity",
              help="Set verbosity level. `-v`: WARNING, `-vv`: INFO, `-vvv`: DEBUG.", 
              type=click.INT, count=True, callback=callback_logging)
def cli(images: list[Path], model_dir: Path, glob: str = "*.png", output: Optional[Path] = None, 
        suffix: str = ".sbb.bin.png", batch_size: Optional[int] = None, no_overwrite: bool = False, 
        force_cpu: bool = False, verbosity: int = 40):
    """
    Pixelwise binarization with selectional auto-encoders in Keras.
    
    IMAGES: List of image file paths to process. Accepts individual files,
    wildcards, or directories (with -g option for pattern matching).
    
    For GPU computation, CUDA 11 is required (>=11.2)
    """
    logger.setLevel(verbosity)
    
    logger.info("Loading images")
    images = expand_paths(images, glob)
    if no_overwrite:
        if output:
            images = list([fp for fp in images if not output.joinpath(f"{fp.name.split('.')[0]}{suffix}").exists()])
        else:
            images = list([fp for fp in images if not fp.parent.joinpath(f"{fp.name.split('.')[0]}{suffix}").exists()])
    if output is not None:
        output.mkdir(exist_ok=True, parents=True)
    logger.info(f"{len(images)} images found")
    
    with progress_spinner as spinner:
        spinner.add_task(description="Loading models", total=None)
        binarizer = SbbBinarizer(model_dir, force_cpu)
    
    with progress_bar as bar:
        task = bar.add_task("Processing images", total=len(images), filename="")
        for i, fp in enumerate(images):
            bar.update(task, filename=Path("/", *fp.parts[-min(len(fp.parts), 4):]))
            if batch_size is not None and i > 0 and i % batch_size == 0:
                logger.info("Reloading model.")
                binarizer.end_session()
                binarizer = SbbBinarizer(model_dir)
            if output is None:
                outfile = fp.parent.joinpath(f"{fp.name.split('.')[0]}{suffix}")
            else:
                outfile = output.joinpath(f"{fp.name.split('.')[0]}{suffix}")
            try:
                image = cv2.imread(fp.as_posix())
                bin_im = binarizer.run(image, use_patches=True)
                cv2.imwrite(outfile.as_posix(), bin_im)
            except Exception as e:
                logger.error(f"Processing failed for file {fp.as_posix()}: {e}")
            bar.update(task, advance=1)
        binarizer.end_session()
        bar.update(task, filename="Done")


if __name__ == "__main__":
    cli()
