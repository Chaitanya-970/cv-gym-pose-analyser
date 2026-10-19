"""Optional browser-compatible preview; original annotated video is preserved."""
from pathlib import Path
import shutil
import subprocess
import uuid


def create_preview(source: Path, destination: Path, timeout: float = 120.0):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.parent != source.parent or destination == source:
        raise ValueError('Preview must be a separate file in the same results directory')
    if destination.exists():
        raise FileExistsError('Refusing to overwrite preview')
    encoder = shutil.which('ffmpeg')
    if not encoder or not source.is_file():
        return {'available': False, 'warning': 'Browser preview unavailable; download the original annotated video.'}
    pending = destination.with_name('preview-' + uuid.uuid4().hex + '.mp4')
    command = [encoder, '-nostdin', '-n', '-loglevel', 'error', '-i', str(source),
               '-map', '0:v:0', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
               '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-preset', 'fast', '-crf', '23',
               '-threads', '2', '-movflags', '+faststart', str(pending)]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                timeout=timeout, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode or not pending.is_file() or not pending.stat().st_size:
            return {'available': False, 'warning': 'Preview encoding failed; original analysis and downloads are preserved.'}
        pending.rename(destination)
        return {'available': True, 'warning': None}
    except (OSError, subprocess.TimeoutExpired):
        return {'available': False, 'warning': 'Preview encoding unavailable or timed out; download the original annotated video.'}
