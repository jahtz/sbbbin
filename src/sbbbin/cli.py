# SPDX-License-Identifier: Apache-2.0
import glob
import logging
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import click
import cv2
import numpy as np
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


logger: logging.Logger = logging.getLogger(__name__)


def setup_logging(level: int = 0) -> None:
    logging.basicConfig(
        level=max(10, 40 - (10 * level)),
        format='%(message)s', 
        datefmt='[%X]', 
        handlers=[RichHandler(markup=True, rich_tracebacks=True)]
    )
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.info(f'Logging verbosity set to {logging.getLevelName(logger.getEffectiveLevel())}')
    
    
def expand_glob(ctx: click.Context, param: click.Parameter, patterns: list[str] | str) -> list[Path]:
    if isinstance(patterns, str):
        patterns = [patterns]
    paths: list[Path] = []
    for pattern in patterns:
        if glob.has_magic(pattern):
            for match in glob.iglob(pattern, recursive=True):
                p = Path(match)
                if p.is_file():
                    paths.append(p.resolve())
        else:
            p = Path(pattern)
            if p.is_file() and p.exists():
                paths.append(p.resolve())
    return paths


@click.command(epilog='Developed at Centre for Philology and Digitality (ZPD), University of Würzburg')
@click.help_option('--help', hidden=True)
@click.version_option(version('sbbbin'), '--version', prog_name='sbbbin', hidden=True)
@click.argument('images', type=click.Path(), callback=expand_glob, nargs=-1, required=True)
@click.option(
    '-o', '--output',
    help='Output directory for generated image files. If omitted, each image file is written next to its input.',
    type=click.Path(file_okay=False, resolve_path=True, path_type=Path),
)
@click.option(
    '-m', '--model',
    help='Path to the PyTorch (.pth) model file.',
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    required=True
)
@click.option(
    '-s', '--suffix',
    help='Suffix for output images.', 
    type=click.STRING,
    default='.sbb.bin.png', 
    show_default=True
)
@click.option(
     '-d', '--device',
     help='Compute device for inference. Use "auto" to automatically choose the best available device.', 
     type=click.Choice(['auto', 'cpu', 'cuda']),
     default='auto',
     show_default=True
)
@click.option(
     '-v', '--verbose', 'verbosity',
     help='Set the verbosity level. Default: ERROR. Use -v for WARNING, -vv for INFO, -vvv for DEBUG.', 
     count=True
)
def cli(
    images: list[Path],
    model: Path,
    output: Path | None,
    suffix: str,
    device: Literal['auto', 'cpu', 'cuda'],
    verbosity: int,
) -> None:
    """
    Pixelwise image binarization with selectional auto-encoders using the SBB Binarization algorithm.
    
    \b
    Arguments:
      IMAGES: One or more image paths. Use glob patterns in quotes to process 
              multiple files.
    """
    setup_logging(verbosity)
        
    if not images:
        raise click.BadArgumentUsage('No input images found')
    if output is not None:
        output.mkdir(exist_ok=True, parents=True)

    with Progress(
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn('[progress.description]{task.description}'),
    ) as progress:
        load_task = progress.add_task('Loading model', total=None)
        from .sbbbin import SbbBinarizer
        binarizer = SbbBinarizer(model, device)
        progress.remove_task(load_task)
        
        task = progress.add_task('Processing images', total=len(images))
        for fp in images:
            progress.update(task, description='/'.join(fp.parts[-4:]))
            logger.info(f'Processing image: {fp}')
            try:
                img: np.ndarray | None = cv2.imread(str(fp))
                if img is None:
                    logger.error(f'Could not open image: {fp.as_posix()}')
                    progress.advance(task)
                    continue
                
                res: np.ndarray = binarizer.process(img, use_patches=True)
                
                out_dir: Path = output or fp.parent
                out_path: Path = out_dir / f'{fp.name.split(".")[0]}{suffix}'
                cv2.imwrite(str(out_path), res)
                
            except Exception as exc:  # noqa: BLE001
                logger.error(f'Failed for {fp}: {exc}')
                
            progress.advance(task)
        progress.update(task, description='Done')
