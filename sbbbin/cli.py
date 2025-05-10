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

import glob
from pathlib import Path
from typing import Optional

import click
import cv2
from rich.progress import (Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn,
                           SpinnerColumn)

from sbbbin.sbbbin import SbbBinarizer


__version__ = "0.3.0"
__prog__ = "sbbbin"
__footer__ = "Developed at Centre for Philology and Digitality (ZPD), University of Würzburg"

PROGRESS = Progress(
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

SPINNER = Progress(
    SpinnerColumn(), 
    TextColumn("[progress.description]{task.description}"), 
    transient=True
)

def callback_paths(ctx, param, value) -> list[Path]:
    if not value:
        raise click.BadParameter("", param=param)
    paths = []
    for pattern in value:
        expanded = glob.glob(pattern, recursive=True)
        if not expanded:
            p = Path(pattern)
            if p.exists() and p.is_file():
                paths.append(p)
        else:
            paths.extend(Path(p) for p in expanded if Path(p).is_file())
    if not paths:
        raise click.BadParameter("None of the provided paths or patterns matched existing files.")
    return paths


@click.command(epilog=__footer__)
@click.help_option("--help")
@click.version_option(
    __version__, "--version",
    prog_name=__prog__,
    message=f"{__prog__} v{__version__}\n{__footer__}"
)
@click.argument(
    "images",
    type=click.Path(exists=False, dir_okay=True, file_okay=True, resolve_path=True),
    required=True,
    callback=callback_paths,
    nargs=-1
)
@click.option(
    "-o", "--output", "output",
    help="Specify output directory for processed files. Defaults to the parent directory of each input file.",
    type=click.Path(exists=False, dir_okay=True, file_okay=False, resolve_path=True, path_type=Path)
)
@click.option(
    "-m", "--model", "model",
    help="Path to directory containing the binarization 'saved_model' directory. See README.md for more information.",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True, path_type=Path),
    required=True
)
@click.option(
    "-s", "--suffix", "suffix",
    help="Specify suffix for output images.", 
    type=click.STRING,
    default=".sbb.bin.png", 
    show_default=True
)
@click.option(
    "--gpu/--cpu", "gpu",
    help="Select computation device. Use '--gpu' for CUDA acceleration (recommended).",
    type=click.BOOL,
    default=True,
    show_default=True
)
def cli(
    images: list[Path],
    model: Path,
    output: Optional[Path] = None,
    suffix: str = ".sbb.bin.png",
    gpu: bool = True
) -> None:
    """
    Pixelwise binarization with selectional auto-encoders in Keras.
    
    IMAGES: List of image file paths to process. Accepts individual files, glob wildcards, or directories.
    """
    if not images:
        raise click.BadArgumentUsage("No images found")
    if output is not None:
        output.mkdir(exist_ok=True, parents=True)
    suffix = suffix if suffix.startswith('.') else '.' + suffix
    
    with SPINNER as spinner:
        spinner.add_task(description="Loading model", total=None)
        binarizer = SbbBinarizer(model, gpu)
    
    with PROGRESS as progressbar:
        task = progressbar.add_task("Processing...", total=len(images), filename="")
        for fp in images:
            progressbar.update(task, filename=Path(*fp.parts[-min(len(fp.parts), 4):]))
            try:
                fn = fp.name.split('.')[0]
                outd = output if output else fp.parent
                bin_im = binarizer.run(fp)
                cv2.imwrite(outd.joinpath(fn + suffix).as_posix(), bin_im)
            except Exception as e:
                progressbar.log(f"Processing failed for file {fp.as_posix()}:\n{e}")
            progressbar.advance(task)
        binarizer.end_session()
        progressbar.update(task, filename="Done") 
