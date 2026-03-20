# SPDX-License-Identifier: Apache-2.0
from importlib.metadata import version
import logging
from pathlib import Path
from typing import Literal

import click
import cv2
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn, 
    Progress, 
    SpinnerColumn,
    TextColumn, 
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .sbb_binarization import SbbBinarizer

try:
    __version__: str = version('sbbbin')
except Exception:
    __version__: str = '0.0.0'
    
logging.basicConfig(
    format='%(message)s',
    datefmt='[%X]',
    handlers=[RichHandler(markup=True)]
)
logger: logging.Logger = logging.getLogger('sbbbin')

spinner: Progress = Progress(
    SpinnerColumn(), 
    TextColumn("[progress.description]{task.description}"), 
    transient=True
)
progressbar: Progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(bar_width=30),
    TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    TextColumn('• {task.fields[status]}')
)


@click.command(epilog='Developed at Centre for Philology and Digitality (ZPD), University of Würzburg')
@click.help_option('--help')
@click.version_option(__version__, '--version', prog_name='sbbbin')
@click.argument(
    'images', 
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path), 
    required=True,
    nargs=-1
)
@click.option(
    '-m', '--model', 'model',
    help="Path to the trained PyTorch (.pth) model.",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    required=True
)
@click.option(
    '-o', '--output',
    help='Specify output directory for processed files. Defaults to the parent directory of each input file.',
    type=click.Path(file_okay=False, resolve_path=True, path_type=Path),
)
@click.option(
    "-s", "--suffix", "suffix",
    help="Specify suffix for output images.", 
    type=click.STRING,
    default=".sbb.bin.png", 
    show_default=True
)
@click.option(
     '-d', '--device', 'device',
     help='Set logging level.', 
     type=click.Choice(['auto', 'cpu', 'cuda']),
     default='auto',
     show_default=True
)
@click.option(
     '--logging', 'logging_level',
     help='Set logging level.', 
     type=click.Choice(['ERROR', 'WARNING', 'INFO']),
     default='ERROR',
     show_default=True
)
def main(
    images: list[Path],
    model: Path,
    output: Path | None = None,
    suffix: str = '.sbb.bin.png',
    device: Literal['auto', 'cpu', 'cuda'] = 'auto',
    logging_level: Literal['ERROR', 'WARNING', 'INFO'] = 'ERROR'
) -> None:
    """
    Pixelwise binarization with selectional auto-encoders using the SBB binarization algorithm
    
    IMAGES: List of image file paths to process.
    """
    logging.getLogger().setLevel(logging_level)
    
    if output is not None:
        output.mkdir(exist_ok=True, parents=True)
    suffix: str = suffix if suffix.startswith('.') else '.' + suffix
    
    with spinner as sp:
        sp.add_task('Loading model', total=None)
        binarizer = SbbBinarizer(model, device)
    
    with progressbar as pb:
        task = pb.add_task('', total=len(images), status='')
        for fp in images:
            pb.update(task, status='/'.join(fp.parts[-4:]))
            try:
                img = cv2.imread(fp)
                if img is None:
                    logger.error(f'Could not open image file at {fp.as_posix()}')
                    continue
                res = binarizer.run(img, use_patches=True)
                if output is None:
                    fn = fp.parent / (fp.name.split('.')[0] + suffix)
                else:
                    fn = output / (fp.name.split('.')[0] + suffix)
                cv2.imwrite(fn, res)
            except Exception as ex:
                logger.error(f'Binarization failed for image {fp.as_posix()}: {ex}')
            pb.advance(task)
        pb.update(task, status='Done')
