# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import click
import cv2
import numpy as np
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from .util import setup_logging, read_boolean_environment, expand_glob


logger: logging.Logger = logging.getLogger(__name__)
SHORT_HELP: bool = read_boolean_environment('SBBBIN_EXTENDED_HELP', True)


@click.command(epilog='Developed at Centre for Philology and Digitality (ZPD), University of Würzburg')
@click.help_option('--help', hidden=SHORT_HELP)
@click.version_option(version('sbbbin'), '--version', prog_name='sbbbin', hidden=SHORT_HELP)
@click.argument(
    'images',
    type=click.Path(),
    callback=expand_glob,
    nargs=-1,
    required=True
)
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
     '--logging', 'level',
     type=click.Choice(['ERROR', 'WARNING', 'INFO', 'DEBUG']),
     default='ERROR',
     show_default=True
)
def cli(
    images: list[Path],
    model: Path,
    output: Path | None = None,
    suffix: str = '.sbb.bin.png',
    device: Literal['auto', 'cpu', 'cuda'] = 'auto',
    level: Literal['ERROR', 'WARNING', 'INFO', 'DEBUG'] = 'ERROR'
) -> None:
    """
    Pixelwise image binarization with selectional auto-encoders using the SBB Binarization algorithm.
    
    To view all available options with '--help', set the environment variable:
        SBBBIN_EXTENDED_HELP=True
    
    IMAGES: One or more image paths. Use glob patterns in quotes to process multiple files.
    """
    setup_logging(level)
    
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
                
            except Exception as exc:
                logger.error(f'Failed for {fp}: {exc}')
                
            progress.advance(task)
        progress.update(task, description='Done')
