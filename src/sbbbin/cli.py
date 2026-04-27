# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import click
import cv2
import numpy as np
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn


logger: logging.Logger = logging.getLogger(__name__)


def setup_logging(level: Literal['ERROR', 'WARNING', 'INFO', 'DEBUG']) -> None:
    logging.basicConfig(
        level=level,
        format='%(message)s', 
        datefmt='[%X]', 
        handlers=[RichHandler(markup=True, rich_tracebacks=True)]
    )


@click.command(epilog='Developed at Centre for Philology and Digitality (ZPD), University of Würzburg')
@click.version_option(version('sbbbin'), '--version', prog_name='sbbbin')
@click.argument(
    'images',
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    nargs=-1,
    required=True
)
@click.option(
    '-m', '--model',
    help='Path to the PyTorch (.pth) model file.',
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    required=True
)
@click.option(
    '-o', '--output',
    help='Output directory (defaults to parent directory of each input file).',
    type=click.Path(file_okay=False, resolve_path=True, path_type=Path),
)
@click.option(
    '-s', '--suffix', 'suffix',
    help='Suffix for output images.', 
    type=click.STRING,
    default=".sbb.bin.png", 
    show_default=True
)
@click.option(
     '-d', '--device', 'device',
     help='Compute device. \'cuda\' requires a bundled CUDA/PyTorch version.', 
     type=click.Choice(['auto', 'cpu', 'cuda']),
     default='auto',
     show_default=True
)
@click.option(
     '--logging', 'logging_level',
     type=click.Choice(['ERROR', 'WARNING', 'INFO', 'DEBUG']),
     default='ERROR',
     show_default=True
)
def main(
    images: list[Path],
    model: Path,
    output: Path | None,
    suffix: str,
    device: Literal['auto', 'cpu', 'cuda'],
    logging_level: Literal['ERROR', 'WARNING', 'INFO', 'DEBUG']
) -> None:
    """
    Pixelwise image binarization with selectional auto-encoders using the SBB Binarization algorithm.
    
    IMAGES: List of image file paths to process.
    """
    setup_logging(logging_level)
    
    if output:
        output.mkdir(exist_ok=True, parents=True)
        
    if not suffix.startswith('.'):
        suffix = f'.{suffix}'

    with Progress(
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn('[progress.description]{task.description}'),
    ) as progress:
        load_task = progress.add_task('Loading model...', total=None)
        from .sbbbin import SbbBinarizer
        binarizer = SbbBinarizer(model, device)
        progress.remove_task(load_task)
        
        task = progress.add_task('Processing images', total=len(images))
        for fp in images:
            progress.update(task, description='/'.join(fp.parts[-4:]))
            logging.info(f'Processing image: {fp}')
            try:
                img: np.ndarray | None = cv2.imread(str(fp))
                if img is None:
                    logger.error(f'Could not open image: {fp.as_posix()}')
                    progress.advance(task)
                    continue
                
                res: np.ndarray = binarizer.run(img, use_patches=True)
                
                out_dir: Path = output or fp.parent
                out_path: Path = out_dir / f'{fp.stem}{suffix}'
                cv2.imwrite(str(out_path), res)
                
            except Exception as exc:
                logger.error(f'Failed for {fp}: {exc}')
                
            progress.advance(task)
        progress.update(task, description='Done')
