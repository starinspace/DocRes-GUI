"""DocRes Studio: a single-file PySide6 desktop UI for the tiled DocRes fork.

Place next to inference.py, utils.py, models/, and data/. Start with launch_gui.bat.
The persistent worker runs this same file with --worker, without importing Qt.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import logging
import os
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
PROTOCOL = "@@DOCRES@@"
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}
STAGES = [
    ('dewarping', 'Dewarp', 'Straighten curved pages',
     'Predicts a global page coordinate map and straightens curved or photographed pages. '
     'Runs first. The fork\'s MBD helper requires CUDA and models/mbd.pkl.'),
    ('deshadowing', 'Remove shadows', 'Recover even illumination',
     'Removes uneven lighting and page shadows while retaining original-size detail.'),
    ('appearance', 'Enhance appearance', 'Clean and balance the page',
     'Improves page appearance and contrast using the document restoration model.'),
    ('deblurring', 'Deblur', 'Restore clearer edges',
     'Restores blurred document detail. Uses overlapping tiles without shrinking the full-resolution export.'),
    ('binarization', 'Binarize', 'Create pure black and white',
     'Converts the document to black and white. Always runs last. Export as packed 1-bit PNG '
     'or Group 4 TIFF for compact files without dithering.'),
]
DEFAULTS = dict(theme='dark', tasks=['binarization'], automatic=False,
                comparison='split', wipe=50, preview_edge=1600, debounce=900,
                tile_size=512, overlap=64, precision='auto', device='cuda',
                output_format='png', binary_bits=1, jpeg_quality=95,
                conflict='keep_both', preserve_dpi=True, recursive=False,
                remember_files=True, remember_settings=True,
                model_path=str(ROOT / 'models' / 'docres.pkl'),
                engine_path=str(ROOT / 'inference.py'), output_dir=str(ROOT / 'output' / 'gui'))


def natural_key(path):
    import re
    return [int(part) if part.isdigit() else part.casefold()
            for part in re.split(r'(\d+)', str(path))]


def collect_images(paths, recursive=False):
    found, seen = [], set()
    for supplied in paths:
        source = Path(supplied).expanduser()
        candidates = sorted(source.rglob('*') if recursive else source.iterdir(), key=natural_key) \
            if source.is_dir() else [source]
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            key = str(path.resolve()).casefold()
            if key not in seen:
                seen.add(key)
                found.append(path.resolve())
    return found


def validated_settings(raw):
    cfg = dict(DEFAULTS)
    cfg['tasks'] = list(DEFAULTS['tasks'])
    if not isinstance(raw, dict):
        return cfg
    choices = dict(theme=('dark', 'light'), comparison=('split', 'slider'),
                   preview_edge=(0, 1024, 1600, 2048), tile_size=(128, 256, 384, 512, 768, 1024),
                   overlap=(0, 16, 32, 64, 96, 128), precision=('auto', 'fp32'),
                   device=('auto', 'cuda', 'cpu'), output_format=('png', 'tiff', 'jpg'),
                   binary_bits=(1, 8), conflict=('keep_both', 'skip', 'replace'))
    for key, values in choices.items():
        if raw.get(key) in values:
            cfg[key] = raw[key]
    for key in ('automatic', 'preserve_dpi', 'recursive', 'remember_files', 'remember_settings'):
        if isinstance(raw.get(key), bool):
            cfg[key] = raw[key]
    for key in ('model_path', 'engine_path', 'output_dir'):
        if isinstance(raw.get(key), str) and raw[key].strip():
            cfg[key] = raw[key]
    for key, low, high in (('wipe', 0, 100), ('jpeg_quality', 1, 100), ('debounce', 300, 5000)):
        value = raw.get(key)
        if isinstance(value, int):
            cfg[key] = max(low, min(high, value))
    if isinstance(raw.get('tasks'), list):
        cfg['tasks'] = [stage[0] for stage in STAGES if stage[0] in raw['tasks']]
    if cfg['overlap'] >= cfg['tile_size']:
        cfg['overlap'] = min(64, cfg['tile_size'] // 4)
    if 'binarization' in cfg['tasks'] and cfg['output_format'] == 'jpg':
        cfg['output_format'] = 'png'
    return cfg


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(str(temporary), str(path))


def processing_signature(source, cfg):
    path = Path(source)
    stat = path.stat()
    keys = ('tasks', 'preview_edge', 'tile_size', 'overlap', 'precision',
            'device', 'model_path', 'engine_path')
    data = {key: cfg[key] for key in keys}
    data['source'] = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    for key in ('model_path', 'engine_path'):
        asset = Path(cfg[key])
        if asset.is_file():
            data[key + '_mtime'] = asset.stat().st_mtime_ns
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def read_pixels(path, edge=0):
    with Image.open(path) as opened:
        dpi = opened.info.get('dpi')
        image = ImageOps.exif_transpose(opened).convert('RGB')
        original_size = image.size
        if edge and max(image.size) > edge:
            image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        rgb = np.asarray(image).copy()
    # DocRes/OpenCV expects BGR, not RGB. EXIF orientation is applied once.
    return np.ascontiguousarray(rgb[:, :, ::-1]), dpi, original_size


def save_pixels(path, pixels, binary=False, bits=1, quality=95, dpi=None, job_id='save', index=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray(pixels)
    if binary:
        if pixels.ndim != 2 or not np.all((pixels == 0) | (pixels == 255)):
            raise ValueError('Binary output must contain only black and white pixels.')
        image = Image.fromarray(pixels.astype(np.uint8))
        if bits == 1:
            image = image.convert('1', dither=0)
    else:
        rgb = pixels[:, :, ::-1] if pixels.ndim == 3 else pixels
        image = Image.fromarray(np.ascontiguousarray(rgb).astype(np.uint8))
    suffix = path.suffix.lower()
    options = {}
    if dpi and isinstance(dpi, (tuple, list)) and len(dpi) == 2:
        if all(isinstance(number, (int, float)) and np.isfinite(number) and number > 0 for number in dpi):
            options['dpi'] = tuple(dpi)
    if suffix == '.png':
        fmt, options = 'PNG', dict(options, optimize=True, compress_level=9)
    elif suffix in ('.tif', '.tiff'):
        fmt = 'TIFF'
        options['compression'] = 'group4' if binary and bits == 1 else 'tiff_adobe_deflate'
    elif suffix in ('.jpg', '.jpeg') and not binary:
        fmt, options = 'JPEG', dict(options, quality=quality, optimize=True, subsampling=0)
    else:
        raise ValueError('Use PNG or TIFF for binary output; PNG, TIFF or JPEG for color output.')
    temporary = path.parent / '.docres-{}-{}.part'.format(job_id, index)
    try:
        image.save(str(temporary), format=fmt, **options)
        os.replace(str(temporary), str(path))
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


def output_label(tasks):
    return 'binarized' if 'binarization' in tasks else tasks[0] if len(tasks) == 1 else 'restored'


def choose_output(source, cfg, reserved=None):
    reserved = reserved if reserved is not None else set()
    ext = {'png': '.png', 'tiff': '.tiff', 'jpg': '.jpg'}[cfg['output_format']]
    folder = Path(cfg['output_dir'])
    label = output_label(cfg['tasks'])
    source = Path(source)
    candidate = folder / '{}_{}{}'.format(source.stem, label, ext)
    key = str(candidate.resolve()).casefold()
    if cfg['conflict'] == 'skip' and (candidate.exists() or key in reserved):
        return None
    # Even Replace may not overwrite the input or another item in this batch.
    index = 2
    while (key in reserved or candidate.resolve() == source.resolve()
           or (cfg['conflict'] == 'keep_both' and candidate.exists())):
        candidate = folder / '{}_{}_{}{}'.format(source.stem, index, label, ext)
        key = str(candidate.resolve()).casefold()
        index += 1
    reserved.add(key)
    return candidate


def emit_event(event, **values):
    print(PROTOCOL + json.dumps(dict(event=event, **values), ensure_ascii=True), flush=True)


def runtime_report(torch_module=None):
    """Probe the same interpreter used for inference, without importing Qt."""
    report = dict(python=sys.executable, python_version=sys.version.split()[0],
                  environment=os.environ.get('CONDA_DEFAULT_ENV') or os.environ.get('VIRTUAL_ENV') or 'not activated',
                  torch_version='', torch_path='', cuda_build='', cuda_available=False,
                  cuda_usable=False, gpu_name='', gpu_memory=0,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'), reason='')
    try:
        if torch_module is None:
            import torch as torch_module
        report['torch_version'] = str(torch_module.__version__)
        report['torch_path'] = str(getattr(torch_module, '__file__', ''))
        report['cuda_build'] = getattr(torch_module.version, 'cuda', None) or ''
        report['cuda_available'] = bool(torch_module.cuda.is_available())
        if not report['cuda_available']:
            report['reason'] = ('This Python has CPU-only PyTorch. Start launch_gui.bat to use the docresfork environment. '
                'If that environment also has CPU-only PyTorch, install a compatible CUDA-enabled build there.'
                if not report['cuda_build'] else
                'PyTorch includes CUDA, but no usable NVIDIA GPU is visible. Check the NVIDIA driver, '
                'CUDA_VISIBLE_DEVICES and the Python environment.')
            return report
        report['gpu_name'] = torch_module.cuda.get_device_name(0)
        report['gpu_memory'] = int(torch_module.cuda.get_device_properties(0).total_memory)
        probe = None
        try:
            # Availability alone does not detect missing GPU kernels or driver failures.
            probe = torch_module.ones(1, device='cuda:0')
            probe.add_(1)
            torch_module.cuda.synchronize()
            if probe.item() != 2:
                raise RuntimeError('The CUDA calculation returned an invalid result.')
            report['cuda_usable'] = True
        finally:
            del probe
            with contextlib.suppress(Exception):
                torch_module.cuda.empty_cache()
    except Exception as error:
        report['reason'] = 'CUDA/runtime check failed: ' + str(error)
    return report


def runtime_text(report):
    return '\n'.join((
        'Python: ' + report['python'],
        'Python version: ' + report['python_version'],
        'Environment: ' + report['environment'],
        'PyTorch: ' + (report['torch_version'] or 'not available'),
        'PyTorch path: ' + report['torch_path'],
        'CUDA build: ' + (report['cuda_build'] or 'none'),
        'CUDA available: ' + str(report['cuda_available']),
        'CUDA calculation passed: ' + str(report['cuda_usable']),
        'GPU: ' + (report['gpu_name'] or 'not detected'),
        'GPU memory: {:.1f} GiB'.format(report['gpu_memory'] / 1024 ** 3),
        'CUDA_VISIBLE_DEVICES: ' + report['cuda_visible_devices'],
        'Result: ' + ('GPU ready.' if report['cuda_usable'] else report['reason'])))


def select_device(requested, report):
    if requested == 'cpu':
        return 'cpu'
    if report['cuda_usable']:
        return 'cuda'
    if requested == 'auto':
        return 'cpu'
    raise RuntimeError('CUDA / GPU is required; processing was not started on CPU. ' + report['reason'] +
                       '\nPython: ' + report['python'] + '\nOpen Engine → Check GPU / Python for diagnostics.')


def runtime_main():
    report = runtime_report()
    print(runtime_text(report), flush=True)
    return 1 if '--require-cuda' in sys.argv and not report['cuda_usable'] else 0


class EngineSession:
    def __init__(self):
        self.module = None
        self.key = None
        self.model = None
        self.runtime = None

    def check_runtime(self):
        self.runtime = runtime_report()
        emit_event('runtime', report=self.runtime)
        return self.runtime

    def prepare(self, cfg):
        engine = Path(cfg['engine_path']).resolve()
        checkpoint = Path(cfg['model_path']).resolve()
        if not engine.is_file():
            raise FileNotFoundError('inference.py was not found. Select it in the Engine tab.')
        if not checkpoint.is_file():
            raise FileNotFoundError('Model checkpoint was not found. Select models/docres.pkl in the Engine tab.')
        report = self.runtime if self.runtime is not None else self.check_runtime()
        device = select_device(cfg['device'], report)
        key = (str(engine), engine.stat().st_mtime_ns, str(checkpoint), checkpoint.stat().st_mtime_ns, device)
        if self.key == key and self.model is not None:
            return self.module, self.model
        emit_event('loading', message='Loading the restoration model…')
        import torch
        if device == 'cpu' and 'dewarping' in cfg['tasks']:
            raise RuntimeError('The MBD dewarping helper in this fork requires CUDA. Use Auto/CUDA or uncheck Dewarp.')
        self.model = None
        import gc
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        os.chdir(str(engine.parent))
        if str(engine.parent) not in sys.path:
            sys.path.insert(0, str(engine.parent))
        spec = importlib.util.spec_from_file_location('_docres_studio_engine', str(engine))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, '_predict_tiled'):
            raise RuntimeError('This GUI needs the updated tiled inference.py included in the package.')
        if device == 'cpu' and 'dewarping' in cfg['tasks']:
            raise RuntimeError('Dewarping requires CUDA in the fork\'s MBD helper. Uncheck Dewarp to use CPU.')
        module.DEVICE = torch.device(device)
        self.model = module.model_init(SimpleNamespace(model_path=str(checkpoint)))
        if callable(getattr(self.model, 'parameters', None)):
            parameter = next(self.model.parameters(), None)
            if parameter is not None and parameter.device.type != device:
                self.model = None
                raise RuntimeError('The inference engine loaded the model on the wrong device. '
                                   'Select the updated inference.py included in this package.')
        self.module, self.key = module, key
        if device == 'cpu' and cfg['device'] == 'auto':
            emit_event('warning', message='Auto selected CPU: ' + report['reason'])
        description = '{} · {}'.format(report['gpu_name'], device.upper()) if device == 'cuda' else 'CPU'
        emit_event('engine', device=device, message='Model ready on {}.'.format(description))
        return module, self.model

    def run(self, request):
        cfg = validated_settings(request['settings'])
        if not cfg['tasks']:
            raise ValueError('Select at least one restoration step.')
        module, model = self.prepare(cfg)
        if 'dewarping' in cfg['tasks'] and module.DEVICE.type != 'cuda':
            raise RuntimeError('Dewarping requires CUDA in this fork.')
        mode, job_id = request['mode'], request['id']
        preview = mode == 'preview'
        inputs = request['inputs']
        # Protect every queued source, including sources outside this selection.
        protected = {str(Path(path).resolve()).casefold()
                     for path in request.get('protected_sources', inputs)}
        reserved = protected.copy()
        success = failures = skipped = total_bytes = 0
        started = time.perf_counter()
        for index, source in enumerate(inputs):
            tick = index * (len(cfg['tasks']) + 1)
            maximum = len(inputs) * (len(cfg['tasks']) + 1)
            emit_event('item_start', id=job_id, source=source, index=index + 1, total=len(inputs))
            try:
                destination = Path(request['destination']) if mode == 'save_one' else \
                    Path(request['preview_path']) if preview else choose_output(source, cfg, reserved)
                if destination is None:
                    skipped += 1
                    emit_event('skipped', id=job_id, source=source, message='Output already exists.')
                    continue
                if str(destination.resolve()).casefold() in protected:
                    raise ValueError('Choose a different filename. Source images are protected.')
                pixels, dpi, original_size = read_pixels(source, cfg['preview_edge'] if preview else 0)
                for step, task in enumerate(cfg['tasks']):
                    emit_event('stage', id=job_id, source=source, task=task,
                               value=tick + step, maximum=maximum,
                               message='{} / {} · {}'.format(index + 1, len(inputs),
                                   next(stage[1] for stage in STAGES if stage[0] == task)))
                    # A binary stage runs last. Each preceding output is BGR.
                    result = getattr(module, task)(model, pixels, memory_fix=0,
                        tile_size=cfg['tile_size'], tile_overlap=cfg['overlap'], precision=cfg['precision'])
                    pixels = result[3]
                    del result
                binary = 'binarization' in cfg['tasks']
                save_pixels(destination, pixels, binary=binary,
                            bits=cfg['binary_bits'] if not preview else 1,
                            quality=cfg['jpeg_quality'], dpi=dpi if cfg['preserve_dpi'] else None,
                            job_id=job_id, index=index)
                size = destination.stat().st_size
                total_bytes += size
                success += 1
                emit_event('result', id=job_id, source=source, path=str(destination),
                           preview=preview, full_resolution=not preview or tuple(pixels.shape[1::-1]) == original_size,
                           width=int(pixels.shape[1]), height=int(pixels.shape[0]), bytes=size,
                           value=tick + len(cfg['tasks']) + 1, maximum=maximum)
                del pixels
            except Exception as error:
                failures += 1
                emit_event('item_error', id=job_id, source=source, message=str(error))
                traceback.print_exc(file=sys.stderr)
            import gc
            gc.collect()
            import torch
            if module.DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
        emit_event('complete', id=job_id, success=success, failures=failures,
                   skipped=skipped, bytes=total_bytes, seconds=time.perf_counter() - started)


def worker_main():
    session = EngineSession()
    emit_event('ready')
    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line)
            if request.get('op') == 'shutdown':
                return 0
            if request.get('op') == 'diagnose':
                session.check_runtime()
                continue
            session.run(request)
        except Exception as error:
            emit_event('fatal', id=request.get('id') if isinstance(request, dict) else None, message=str(error))
            traceback.print_exc(file=sys.stderr)
    return 0


# Worker startup bypasses every Qt import. This also isolates OpenCV's Qt
# plugin environment changes and all GPU allocations from the desktop UI.
if __name__ == '__main__':
    if '--check-runtime' in sys.argv:
        raise SystemExit(runtime_main())
    if '--worker' in sys.argv:
        raise SystemExit(worker_main())

from PySide6.QtCore import (Qt, QTimer, QProcess, QProcessEnvironment, QStandardPaths,
                           Signal, QObject, QRunnable, QThreadPool, QRectF, QPointF, QSize)
from PySide6.QtGui import (QAction, QKeySequence, QImage, QPixmap, QPainter, QColor,
                          QPen, QFont, QIcon, QDesktopServices, QPalette)
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QFrame, QLabel,
    QPushButton, QCheckBox, QComboBox, QSpinBox, QSlider, QLineEdit, QFileDialog,
    QMessageBox, QListWidget, QListWidgetItem, QAbstractItemView, QVBoxLayout,
    QHBoxLayout, QSplitter, QTabWidget, QScrollArea,
    QProgressBar, QPlainTextEdit, QToolButton, QSizePolicy,
    QStyleOptionButton, QStyle)


def app_directory():
    override = os.environ.get('DOCRES_GUI_CONFIG_DIR')
    return Path(override) if override else Path(QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation))


def qimage_from_pil(image):
    rgb = image.convert('RGB')
    data = rgb.tobytes()
    # copy detaches the QImage from the short-lived Python bytes buffer.
    return QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888).copy()


class LoadSignals(QObject):
    done = Signal(str, str, QImage, str)
    failed = Signal(str, str, str)


class ImageLoader(QRunnable):
    def __init__(self, path, role, thumbnail=False):
        super().__init__()
        self.path, self.role, self.thumbnail = str(path), role, thumbnail
        self.signals = LoadSignals()

    def run(self):
        try:
            with Image.open(self.path) as source:
                width, height = source.size
                if source.getexif().get(274) in (5, 6, 7, 8):
                    width, height = height, width
                if self.thumbnail:
                    source.draft('RGB', (80, 96))
                image = ImageOps.exif_transpose(source).convert('RGB')
                metadata = '{} × {} px'.format(width, height)
                if self.thumbnail:
                    image.thumbnail((54, 66), Image.Resampling.LANCZOS)
                result = qimage_from_pil(image)
            self.signals.done.emit(self.path, self.role, result, metadata)
        except Exception as error:
            self.signals.failed.emit(self.path, self.role, str(error))


class CheckBox(QCheckBox):
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.isChecked():
            return
        option = QStyleOptionButton()
        self.initStyleOption(option)
        rect = self.style().subElementRect(QStyle.SubElement.SE_CheckBoxIndicator, option, self)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor('white' if self.isEnabled() else '#94a3b9'), 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(QPointF(rect.left() + 4, rect.center().y()),
                         QPointF(rect.left() + 7, rect.bottom() - 4))
        painter.drawLine(QPointF(rect.left() + 7, rect.bottom() - 4),
                         QPointF(rect.right() - 3, rect.top() + 4))


class ChoiceBox(QComboBox):
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(self.palette().color(QPalette.ColorRole.Text), 1.5))
        x, y = self.width() - 16, self.height() / 2
        painter.drawLine(QPointF(x - 3, y - 2), QPointF(x, y + 1))
        painter.drawLine(QPointF(x, y + 1), QPointF(x + 3, y - 2))


class ImageQueue(QListWidget):
    filesDropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setObjectName('imageQueue')
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setIconSize(QSize(54, 66))
        self.setSpacing(5)
        self.setAcceptDrops(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            self.filesDropped.emit([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class CompareCanvas(QWidget):
    wipeChanged = Signal(int)
    filesDropped = Signal(list)
    zoomChanged = Signal(str)

    def __init__(self):
        super().__init__()
        self.original = QImage()
        self.edited = QImage()
        self.mode, self.wipe, self.zoom = 'split', 0.5, 1.0
        self.pan = QPointF(0, 0)
        self.drag_point = None
        self.drag_wipe = False
        self.dark = True
        self.setMinimumSize(360, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setToolTip('Scroll to zoom. Drag the image to pan. In Slider view, drag the blue divider. '
                        'Zoom and pan stay synchronized for both images.')

    def set_images(self, original=None, edited=None, reset=False):
        if original is not None:
            self.original = original
        if edited is not None:
            self.edited = edited
        if reset:
            self.fit()
        self.update()

    def fit(self):
        self.zoom, self.pan = 1.0, QPointF(0, 0)
        self.zoomChanged.emit('Fit')
        self.update()

    def pane_rects(self):
        area = QRectF(self.rect()).adjusted(6, 6, -6, -6)
        if self.mode == 'slider':
            return [area]
        width = (area.width() - 16) / 2
        return [QRectF(area.x(), area.y(), width, area.height()),
                QRectF(area.x() + width + 16, area.y(), width, area.height())]

    def image_rect(self, pane):
        if self.original.isNull():
            return QRectF()
        usable = pane.adjusted(14, 16, -14, -16)
        scale = min(usable.width() / self.original.width(), usable.height() / self.original.height()) * self.zoom
        w, h = self.original.width() * scale, self.original.height() * scale
        return QRectF(pane.center().x() - w / 2 + self.pan.x(),
                      pane.center().y() - h / 2 + self.pan.y(), w, h)

    def actual_size(self):
        if self.original.isNull():
            return
        usable = self.pane_rects()[0].adjusted(14, 16, -14, -16)
        fitted = min(usable.width() / self.original.width(), usable.height() / self.original.height())
        self.zoom = 1.0 / fitted
        self.pan = QPointF(0, 0)
        self.zoomChanged.emit('100%')
        self.update()

    def zoom_by(self, multiplier):
        self.zoom = max(0.08, min(32.0, self.zoom * multiplier))
        self.zoomChanged.emit('{:.0f}% of fit'.format(self.zoom * 100))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom <= 3)
        bg = QColor('#111822' if self.dark else '#edf1f6')
        border = QColor('#263345' if self.dark else '#d5deea')
        muted = QColor('#8797ae' if self.dark else '#61728a')
        for pane in self.pane_rects():
            painter.setPen(QPen(border, 1))
            painter.setBrush(bg)
            painter.drawRoundedRect(pane, 12, 12)
        if self.original.isNull():
            painter.setPen(muted)
            painter.setFont(QFont('Segoe UI', 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             'Drop images here\n\nAdd a folder or choose files to get started')
            return
        panes = self.pane_rects()
        if self.mode == 'split':
            for pane, image in zip(panes, (self.original, self.edited)):
                painter.save()
                painter.setClipRect(pane.adjusted(1, 1, -1, -1))
                if not image.isNull():
                    painter.drawImage(self.image_rect(pane), image)
                else:
                    painter.setPen(muted)
                    painter.setFont(QFont('Segoe UI', 12))
                    painter.drawText(pane, Qt.AlignmentFlag.AlignCenter,
                                     'Your edited image appears here\n\nChoose steps, then generate a preview')
                painter.restore()
        else:
            pane = panes[0]
            target = self.image_rect(pane)
            x = pane.x() + pane.width() * self.wipe
            painter.save()
            painter.setClipRect(pane.adjusted(1, 1, -1, -1))
            painter.drawImage(target, self.original)
            if not self.edited.isNull():
                painter.setClipRect(QRectF(x, pane.y(), pane.right() - x, pane.height()), Qt.ClipOperation.IntersectClip)
                painter.drawImage(target, self.edited)
            painter.restore()
            painter.setPen(QPen(QColor('#5b9bff'), 2))
            painter.drawLine(QPointF(x, pane.top() + 8), QPointF(x, pane.bottom() - 8))
            painter.setBrush(QColor('#3678eb'))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(x, pane.center().y()), 18, 18)
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            cy = pane.center().y()
            for dx, direction in ((-6, -1), (6, 1)):
                painter.drawLine(QPointF(x + dx, cy - 4), QPointF(x + dx + direction * 4, cy))
                painter.drawLine(QPointF(x + dx + direction * 4, cy), QPointF(x + dx, cy + 4))

    def wheelEvent(self, event):
        self.zoom_by(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pane = self.pane_rects()[0]
            x = pane.x() + pane.width() * self.wipe
            self.drag_wipe = self.mode == 'slider' and abs(event.position().x() - x) < 23
            self.drag_point = event.position()
            self.setCursor(Qt.CursorShape.SplitHCursor if self.drag_wipe else Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.drag_point is not None:
            if self.drag_wipe:
                pane = self.pane_rects()[0]
                value = round(max(0.0, min(1.0, (event.position().x() - pane.x()) / pane.width())) * 100)
                self.wipeChanged.emit(value)
            else:
                self.pan += event.position() - self.drag_point
                self.drag_point = event.position()
                self.update()

    def mouseReleaseEvent(self, event):
        self.drag_point = None
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, event):
        self.fit()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.filesDropped.emit([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
        event.acceptProposedAction()


def theme_css(theme):
    dark = theme == 'dark'
    c = dict(bg='#0d121b' if dark else '#f2f5fa', panel='#171f2b' if dark else '#ffffff',
             raised='#202b3b' if dark else '#edf2f9', text='#e6edf8' if dark else '#1b2940',
             muted='#94a3b9' if dark else '#63748b', line='#2b394d' if dark else '#dbe3ee',
             accent='#619cff', selected='#203956' if dark else '#e5efff')
    return """
    QWidget { color: $text; font-family: 'Segoe UI'; font-size: 12px; }
    QMainWindow, QDialog, QWidget#root { background: $bg; }
    QLabel { background: transparent; }
    QLabel#brand { font-size: 23px; font-weight: 700; letter-spacing: -0.4px; }
    QLabel#eyebrow { color: $muted; font-size: 10px; font-weight: 700; letter-spacing: 1.5px; }
    QLabel#muted { color: $muted; }
    QLabel#badge { color: $accent; background: $selected; border-radius: 8px; padding: 5px 9px; font-size: 10px; font-weight: 600; }
    QFrame#card { background: $panel; border: 1px solid $line; border-radius: 14px; }
    QFrame#stageCard { background: $panel; border: 1px solid $line; border-radius: 10px; }
    QFrame#stageCard[active="true"] { background: $selected; border-color: #3d6596; }
    QLabel#stageNumber { background: $raised; border-radius: 8px; color: $muted; font-weight: 700; }
    QPushButton, QToolButton { background: $raised; border: 1px solid $line; border-radius: 8px; padding: 8px 12px; font-weight: 500; }
    QPushButton:hover, QToolButton:hover { background: $selected; border-color: #5283bf; }
    QPushButton:pressed { background: #285ca2; }
    QPushButton:disabled, QToolButton:disabled { color: $muted; border-color: $line; background: $panel; }
    QPushButton#primary { background: #377ce5; border-color: #377ce5; color: white; font-weight: 600; }
    QPushButton#primary:hover { background: #4b8df2; }
    QPushButton#primary:disabled { background: $raised; color: $muted; border-color: $line; }
    QPushButton#stop { border-color: #874b50; color: #ef9ea4; }
    QPushButton#stop:disabled { color: $muted; border-color: $line; }
    QCheckBox { spacing: 9px; background: transparent; padding: 3px 0px; }
    QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid $muted; border-radius: 4px; background: $panel; }
    QCheckBox::indicator:checked { background: #377ce5; border: 2px solid #96bcff; }
    QCheckBox::indicator:disabled { border-color: $line; }
    QLineEdit, QComboBox, QSpinBox { background: $raised; border: 1px solid $line; border-radius: 7px; padding: 6px 9px; min-height: 19px; }
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border-color: $accent; }
    QLineEdit:disabled, QComboBox:disabled { color: $muted; }
    QComboBox::drop-down { border: none; width: 22px; }
    QComboBox::down-arrow { image: none; }
    QComboBox QAbstractItemView { background: $raised; selection-background-color: $selected; padding: 4px; }
    QListWidget { color: $text; selection-color: $text; selection-background-color: $selected; border: none; background: transparent; outline: none; }
    QListWidget::item { background: $raised; border: 1px solid transparent; border-radius: 9px; padding: 7px; }
    QListWidget::item:selected { color: $text; background: $selected; border: 1px solid #4d85d1; }
    QListWidget::item:hover { border-color: $line; }
    QTabWidget::pane { border: none; background: transparent; }
    QTabBar::tab { background: transparent; color: $muted; padding: 10px 13px; border-bottom: 2px solid transparent; }
    QTabBar::tab:selected { color: $text; border-bottom-color: $accent; }
    QScrollArea { border: none; background: transparent; }
    QScrollBar:vertical { background: transparent; width: 8px; margin: 3px; }
    QScrollBar::handle:vertical { background: $line; border-radius: 3px; min-height: 24px; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    QSlider::groove:horizontal { height: 5px; background: $line; border-radius: 2px; }
    QSlider::sub-page:horizontal { background: #377ce5; border-radius: 2px; }
    QSlider::handle:horizontal { width: 15px; margin: -6px 0; border-radius: 7px; background: #80b0ff; }
    QProgressBar { background: $raised; border: none; border-radius: 4px; height: 7px; max-height: 7px; }
    QProgressBar::chunk { background: #4b8df2; border-radius: 4px; }
    QPlainTextEdit { background: $bg; color: $muted; border: 1px solid $line; border-radius: 8px; font-family: 'Consolas'; font-size: 11px; }
    QMenuBar { background: $bg; color: $muted; padding: 3px 12px; }
    QMenuBar::item:selected { background: $raised; }
    QMenu { background: $panel; border: 1px solid $line; padding: 6px; }
    QMenu::item { padding: 7px 22px; }
    QMenu::item:selected { background: $selected; }
    QToolTip { color: $text; background: $raised; border: 1px solid $line; padding: 9px; }
    QSplitter::handle { background: transparent; width: 9px; }
    """.replace('$bg', c['bg']).replace('$panel', c['panel']).replace('$raised', c['raised']) \
        .replace('$text', c['text']).replace('$muted', c['muted']).replace('$line', c['line']) \
        .replace('$accent', c['accent']).replace('$selected', c['selected'])


def label(text, name=None, wrap=False):
    widget = QLabel(text)
    if name:
        widget.setObjectName(name)
    widget.setWordWrap(wrap)
    return widget


def button(text, callback, primary=False, tooltip=None):
    widget = QPushButton(text)
    if primary:
        widget.setObjectName('primary')
    widget.clicked.connect(callback)
    if tooltip:
        widget.setToolTip(tooltip)
    return widget


def combo(items, tip):
    widget = ChoiceBox()
    for text, value in items:
        widget.addItem(text, value)
    widget.setToolTip(tip)
    return widget


class StudioWindow(QMainWindow):
    def __init__(self, device_override=None):
        super().__init__()
        self.setWindowTitle('DocRes Studio')
        self.resize(1460, 920)
        self.setMinimumSize(1090, 650)
        self.config_path = app_directory() / 'settings.json'
        self.cfg = dict(DEFAULTS)
        self.saved_raw = {}
        if self.config_path.is_file():
            try:
                self.saved_raw = json.loads(self.config_path.read_text(encoding='utf-8'))
                if not isinstance(self.saved_raw, dict):
                    self.saved_raw = {}
                self.cfg = validated_settings(self.saved_raw)
            except (OSError, ValueError):
                logging.exception('Could not read saved settings')
        if device_override is not None:
            self.cfg['device'] = device_override
        self.initializing = True
        self.temp = tempfile.TemporaryDirectory(prefix='docres-studio-')
        self.worker = None
        self.worker_engine = None
        self.runtime_info = None
        self.diagnose_pending = False
        self.stdout_buffer = ''
        self.active = None
        self.pending_auto = False
        self.cancelled = False
        self.closing = False
        self.preview_cache = {}
        self.current_signature = None
        self.loader_tasks = set()
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._auto_preview)
        self.settings_timer = QTimer(self)
        self.settings_timer.setSingleShot(True)
        self.settings_timer.setInterval(700)
        self.settings_timer.timeout.connect(self._autosave)
        self._build_ui()
        self._build_menu()
        self._apply_controls(self.cfg)
        self.initializing = False
        self.apply_theme(self.cfg['theme'])
        geometry = self.saved_raw.get('geometry')
        if isinstance(geometry, str):
            from PySide6.QtCore import QByteArray
            self.restoreGeometry(QByteArray.fromBase64(geometry.encode()))
        if self.cfg['remember_files']:
            files = self.saved_raw.get('files', [])
            if isinstance(files, list):
                self.add_images([path for path in files if isinstance(path, str) and Path(path).is_file()])
        self._update_buttons()
        QTimer.singleShot(0, self.check_gpu)

    def _build_ui(self):
        root = QWidget()
        root.setObjectName('root')
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(20, 14, 20, 16)
        main.setSpacing(14)
        header = QHBoxLayout()
        mark = label('DR', 'badge')
        mark.setFixedSize(42, 42)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setStyleSheet('font-size: 17px; font-weight: 800; border-radius: 12px;')
        header.addWidget(mark)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(label('DocRes Studio', 'brand'))
        titles.addWidget(label('Restore pages. Compare every detail.', 'muted'))
        header.addLayout(titles)
        header.addStretch()
        self.theme_button = button('Light mode', self.toggle_theme,
                                   tooltip='Switch between dark and light appearance. Your choice is remembered.')
        header.addWidget(self.theme_button)
        header.addWidget(button('Save settings', self.save_settings,
                                tooltip='Save the current pipeline, export options and layout for next time.'))
        main.addLayout(header)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        # Image queue.
        queue_card = QFrame()
        queue_card.setObjectName('card')
        queue_card.setMinimumWidth(195)
        queue_card.setMaximumWidth(360)
        queue_layout = QVBoxLayout(queue_card)
        queue_layout.setContentsMargins(14, 16, 14, 14)
        heading = QHBoxLayout()
        heading.addWidget(label('IMAGE QUEUE', 'eyebrow'))
        heading.addStretch()
        self.queue_count = label('0', 'badge')
        heading.addWidget(self.queue_count)
        queue_layout.addLayout(heading)
        add_row = QHBoxLayout()
        add_row.addWidget(button('Add files', self.add_files_dialog))
        add_row.addWidget(button('Folder', self.add_folder_dialog))
        queue_layout.addLayout(add_row)
        queue_layout.addWidget(label('Drop images or folders.\nDrag items to change batch order.', 'muted', True))
        self.queue = ImageQueue()
        self.queue.filesDropped.connect(self.add_images)
        self.queue.currentItemChanged.connect(self._selection_changed)
        self.queue.itemSelectionChanged.connect(self._update_buttons)
        self.queue.model().rowsMoved.connect(self._queue_changed)
        queue_layout.addWidget(self.queue, 1)
        remove_row = QHBoxLayout()
        self.remove_button = button('Remove', self.remove_selected)
        self.clear_button = button('Clear', self.clear_queue)
        remove_row.addWidget(self.remove_button)
        remove_row.addWidget(self.clear_button)
        queue_layout.addLayout(remove_row)
        self.splitter.addWidget(queue_card)

        # Large comparison workspace.
        workspace = QFrame()
        workspace.setObjectName('card')
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(16, 15, 16, 15)
        workspace_layout.setSpacing(10)
        toolbar = QHBoxLayout()
        self.compare_mode = combo([('Side by side', 'split'), ('Comparison slider', 'slider')],
                                  'Side by side places the original on the left and edited result on the right. '
                                  'Comparison slider overlays them with a draggable divider.')
        self.compare_mode.currentIndexChanged.connect(self._comparison_changed)
        toolbar.addWidget(self.compare_mode)
        toolbar.addStretch()
        self.preview_badge = label('READY TO GENERATE', 'badge')
        toolbar.addWidget(self.preview_badge)
        workspace_layout.addLayout(toolbar)
        captions = QHBoxLayout()
        captions.addWidget(label('ORIGINAL', 'eyebrow'), 1)
        edited_label = label('EDITED', 'eyebrow')
        edited_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        captions.addWidget(edited_label, 1)
        workspace_layout.addLayout(captions)
        self.canvas = CompareCanvas()
        self.canvas.filesDropped.connect(self.add_images)
        self.canvas.wipeChanged.connect(self._set_wipe)
        workspace_layout.addWidget(self.canvas, 1)
        self.wipe_row = QWidget()
        wipe_layout = QHBoxLayout(self.wipe_row)
        wipe_layout.setContentsMargins(5, 0, 5, 0)
        wipe_layout.addWidget(label('Original', 'muted'))
        self.wipe_slider = QSlider(Qt.Orientation.Horizontal)
        self.wipe_slider.setRange(0, 100)
        self.wipe_slider.setValue(50)
        self.wipe_slider.valueChanged.connect(self._set_wipe)
        wipe_layout.addWidget(self.wipe_slider, 1)
        wipe_layout.addWidget(label('Edited', 'muted'))
        workspace_layout.addWidget(self.wipe_row)
        zoom_row = QHBoxLayout()
        self.image_info = label('No image selected', 'muted')
        zoom_row.addWidget(self.image_info, 1)
        zoom_row.addWidget(button('−', lambda: self.canvas.zoom_by(1 / 1.2)))
        zoom_row.addWidget(button('+', lambda: self.canvas.zoom_by(1.2)))
        zoom_row.addWidget(button('Fit', self.canvas.fit))
        zoom_row.addWidget(button('100%', self.canvas.actual_size,
                                  tooltip='Show original pixels at 100%. Use Full resolution preview for matching edited detail.'))
        self.zoom_label = label('Fit', 'muted')
        zoom_row.addWidget(self.zoom_label)
        self.canvas.zoomChanged.connect(self.zoom_label.setText)
        workspace_layout.addLayout(zoom_row)
        preview_row = QHBoxLayout()
        self.automatic = CheckBox('Auto-generate preview')
        self.automatic.setToolTip('Generate after selecting an image or changing processing settings. '
                                 'Changes are debounced; only the newest pending preview runs. Exports are never automatic.')
        self.automatic.toggled.connect(self._processing_changed)
        preview_row.addWidget(self.automatic)
        preview_row.addStretch()
        self.generate_button = button('Generate preview', self.generate_preview, True,
                                     'Generate the selected image manually using the preview resolution in the Engine tab.')
        preview_row.addWidget(self.generate_button)
        workspace_layout.addLayout(preview_row)
        self.splitter.addWidget(workspace)

        # Pipeline/settings panel.
        controls = QFrame()
        controls.setObjectName('card')
        controls.setMinimumWidth(285)
        controls.setMaximumWidth(410)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(13, 12, 13, 15)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._pipeline_tab(), 'Pipeline')
        self.tabs.addTab(self._export_tab(), 'Export')
        self.tabs.addTab(self._engine_tab(), 'Engine')
        controls_layout.addWidget(self.tabs, 1)
        self.save_button = button('Save edited image…', self.save_current,
                                  tooltip='Process this image at full resolution and save it. Never silently saves a reduced preview.')
        controls_layout.addWidget(self.save_button)
        self.selected_button = button('Process selected', lambda: self.run_batch(True))
        controls_layout.addWidget(self.selected_button)
        self.batch_button = button('Process all images', lambda: self.run_batch(False), True,
                                   'Process the entire queue at full resolution in its current order. '
                                   'The model is loaded once and each image is saved immediately.')
        controls_layout.addWidget(self.batch_button)
        self.splitter.addWidget(controls)
        self.splitter.setSizes([220, 820, 330])
        main.addWidget(self.splitter, 1)

        footer = QHBoxLayout()
        status_column = QVBoxLayout()
        status_column.setSpacing(7)
        status_row = QHBoxLayout()
        self.status = label('Ready · add images and choose a restoration pipeline', 'muted')
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        status_row.addWidget(self.status, 1)
        self.engine_badge = label('ENGINE IDLE', 'badge')
        status_row.addWidget(self.engine_badge)
        status_column.addLayout(status_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        status_column.addWidget(self.progress)
        footer.addLayout(status_column, 1)
        self.log_button = button('Details', self.toggle_log)
        footer.addWidget(self.log_button)
        self.stop_button = button('Stop', self.cancel_job,
                                  tooltip='Stop the current job and release its worker/GPU memory. Completed batch exports are kept.')
        self.stop_button.setObjectName('stop')
        footer.addWidget(self.stop_button)
        main.addLayout(footer)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(155)
        self.log.setMaximumBlockCount(1500)
        self.log.hide()
        main.addWidget(self.log)

    def _scroll_page(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(2, 13, 2, 8)
        layout.setSpacing(12)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll, layout

    def _pipeline_tab(self):
        scroll, layout = self._scroll_page()
        layout.addWidget(label('YOUR RESTORATION PIPELINE', 'eyebrow'))
        layout.addWidget(label('Check the steps you need.\nRuns top to bottom; binarization is last.', 'muted', True))
        self.stage_checks = {}
        self.stage_cards = {}
        for index, (task, title, description, tooltip) in enumerate(STAGES, 1):
            card = QFrame()
            card.setObjectName('stageCard')
            card_layout = QHBoxLayout(card)
            card_layout.setContentsMargins(10, 10, 10, 10)
            number = label(str(index), 'stageNumber')
            number.setFixedSize(25, 25)
            number.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(number, 0, Qt.AlignmentFlag.AlignTop)
            texts = QVBoxLayout()
            texts.setSpacing(3)
            check = CheckBox(title)
            check.setToolTip(tooltip)
            check.setStyleSheet('font-weight: 600;')
            check.toggled.connect(self._processing_changed)
            texts.addWidget(check)
            texts.addWidget(label(description, 'muted', True))
            card_layout.addLayout(texts, 1)
            card.setToolTip(tooltip)
            self.stage_checks[task], self.stage_cards[task] = check, card
            layout.addWidget(card)
        layout.addWidget(label('Preview is for comparison. Save and batch always use the full input resolution.', 'muted', True))
        layout.addStretch()
        return scroll

    def _path_field(self, text, callback, tip):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        edit = QLineEdit(text)
        edit.setToolTip(tip)
        edit.editingFinished.connect(self._processing_changed)
        row_layout.addWidget(edit, 1)
        browse = button('…', callback)
        browse.setFixedWidth(34)
        browse.setToolTip('Browse')
        row_layout.addWidget(browse)
        return row, edit

    def _export_tab(self):
        scroll, layout = self._scroll_page()
        layout.addWidget(label('FULL-RESOLUTION EXPORT', 'eyebrow'))
        layout.addWidget(label('Output folder', 'muted'))
        row, self.output_dir = self._path_field('', self.browse_output, 'Batch exports are written to this folder.')
        # Output-only settings never schedule GPU work.
        self.output_dir.editingFinished.disconnect(self._processing_changed)
        self.output_dir.editingFinished.connect(self._export_changed)
        layout.addWidget(row)
        self.format_combo = combo([('PNG · lossless', 'png'), ('TIFF · lossless', 'tiff'),
                                   ('JPEG · color / grayscale', 'jpg')],
                                  'PNG and TIFF preserve all pixels. JPEG is lossy and unavailable when Binarize is checked.')
        layout.addWidget(label('File format', 'muted'))
        layout.addWidget(self.format_combo)
        self.format_combo.currentIndexChanged.connect(self._export_changed)
        self.bits_combo = combo([('1-bit · compact black / white', 1), ('8-bit · black / white grayscale', 8)],
                                'Applies only to Binarize. 1-bit PNG packs eight pixels per byte before compression. '
                                '1-bit TIFF uses CCITT Group 4. No dithering or resizing is added.')
        layout.addWidget(label('Binary storage', 'muted'))
        layout.addWidget(self.bits_combo)
        self.bits_combo.currentIndexChanged.connect(self._export_changed)
        self.jpeg_quality = QSpinBox()
        self.jpeg_quality.setRange(1, 100)
        self.jpeg_quality.setToolTip('JPEG quality. 95 favors detail; lower values reduce file size.')
        self.jpeg_quality.valueChanged.connect(self._export_changed)
        layout.addWidget(label('JPEG quality', 'muted'))
        layout.addWidget(self.jpeg_quality)
        self.conflict_combo = combo([('Keep both · add a number', 'keep_both'),
                                     ('Skip existing files', 'skip'), ('Replace existing output', 'replace')],
                                    'Keep both preserves existing exports. Replace only affects output files; input images are protected.')
        self.conflict_combo.currentIndexChanged.connect(self._export_changed)
        layout.addWidget(label('If an output already exists', 'muted'))
        layout.addWidget(self.conflict_combo)
        self.preserve_dpi = CheckBox('Preserve source DPI metadata')
        self.preserve_dpi.setToolTip('Copy valid source DPI metadata to exports for print/layout applications. '
                                   'Pixel dimensions are retained regardless of this setting.')
        self.preserve_dpi.toggled.connect(self._export_changed)
        layout.addWidget(self.preserve_dpi)
        self.export_hint = label('', 'muted', True)
        layout.addWidget(self.export_hint)
        layout.addWidget(button('Open output folder', self.open_output))
        layout.addStretch()
        return scroll

    def _engine_tab(self):
        scroll, layout = self._scroll_page()
        layout.addWidget(label('PREVIEW & PROCESSING', 'eyebrow'))
        self.gpu_status = label('Checking GPU / Python…', 'muted', True)
        layout.addWidget(self.gpu_status)
        diagnostic_row = QHBoxLayout()
        self.gpu_check_button = button('Check GPU / Python', self.check_gpu,
            tooltip='Test CUDA in the exact Python used by the model worker. Does not load the document model.')
        self.gpu_copy_button = button('Copy report', self.copy_gpu_report,
            tooltip='Copy the Python path, PyTorch version, CUDA build, GPU and diagnostic result.')
        self.gpu_copy_button.setEnabled(False)
        diagnostic_row.addWidget(self.gpu_check_button)
        diagnostic_row.addWidget(self.gpu_copy_button)
        layout.addLayout(diagnostic_row)
        self.preview_quality = combo([('Fast · 1024 px', 1024), ('Balanced · 1600 px', 1600),
                                      ('Detailed · 2048 px', 2048), ('Full resolution', 0)],
                                     'Longest edge used only for previews. Batch and Save always process the full-resolution original.')
        self.preview_quality.currentIndexChanged.connect(self._processing_changed)
        layout.addWidget(label('Preview resolution', 'muted'))
        layout.addWidget(self.preview_quality)
        self.tile_combo = combo([(str(size) + ' px', size) for size in (128, 256, 384, 512, 768, 1024)],
                                'Maximum GPU tile edge. 512 is the default for your 12 GB GPU. '
                                'Lower values use less VRAM. The engine automatically shrinks tiles after CUDA OOM.')
        self.tile_combo.currentIndexChanged.connect(self._processing_changed)
        layout.addWidget(label('GPU tile size', 'muted'))
        layout.addWidget(self.tile_combo)
        self.overlap_combo = combo([(str(size) + ' px', size) for size in (0, 16, 32, 64, 96, 128)],
                                   'Blend overlapping tiles to reduce seams. The engine limits overlap to a quarter of the active tile edge.')
        self.overlap_combo.currentIndexChanged.connect(self._processing_changed)
        layout.addWidget(label('Tile overlap', 'muted'))
        layout.addWidget(self.overlap_combo)
        self.device_combo = combo([('CUDA / GPU · required', 'cuda'), ('Auto · allow CPU fallback', 'auto'), ('CPU · explicit', 'cpu')],
                                  'CUDA is the default and refuses CPU fallback. Auto may use CPU when CUDA fails. '
                                  'CPU must be selected explicitly. Dewarping requires CUDA.')
        self.device_combo.currentIndexChanged.connect(self._processing_changed)
        layout.addWidget(label('Processing device', 'muted'))
        layout.addWidget(self.device_combo)
        self.precision_combo = combo([('Auto · mixed precision on CUDA', 'auto'), ('Full precision · FP32', 'fp32')],
                                     'Auto reduces GPU memory use. FP32 may help if the model reports NaN/Inf but uses more memory.')
        self.precision_combo.currentIndexChanged.connect(self._processing_changed)
        layout.addWidget(label('Precision', 'muted'))
        layout.addWidget(self.precision_combo)
        layout.addWidget(label('Model checkpoint', 'muted'))
        row, self.model_path = self._path_field('', self.browse_model, 'Select the DocRes checkpoint, normally models/docres.pkl.')
        layout.addWidget(row)
        layout.addWidget(label('Inference engine', 'muted'))
        row, self.engine_path = self._path_field('', self.browse_engine, 'Select the updated tiled inference.py. Its models/ and data/ folders stay beside it.')
        layout.addWidget(row)
        self.debounce = QSpinBox()
        self.debounce.setRange(300, 5000)
        self.debounce.setSingleStep(100)
        self.debounce.setSuffix(' ms')
        self.debounce.setToolTip('Wait after the latest change before auto-generating. Only one pending preview is kept.')
        self.debounce.valueChanged.connect(self._export_changed)
        layout.addWidget(label('Auto-preview delay', 'muted'))
        layout.addWidget(self.debounce)
        self.recursive = CheckBox('Include subfolders when adding folders')
        self.recursive.setToolTip('Add supported images from all subfolders. Without this option, only the chosen folder is scanned.')
        self.recursive.toggled.connect(self._export_changed)
        self.remember_files = CheckBox('Restore image queue on next launch')
        self.remember_files.setToolTip('Remember image paths and queue order. Missing images are ignored on the next launch.')
        self.remember_files.toggled.connect(self._export_changed)
        self.remember_settings = CheckBox('Automatically remember settings')
        self.remember_settings.setToolTip('Save changes automatically and on exit. You can always use Save settings manually.')
        self.remember_settings.toggled.connect(self._export_changed)
        self.remember_settings.toggled.connect(lambda checked: self.save_settings(notify=False)
                                               if not self.initializing else None)
        for widget in (self.recursive, self.remember_files, self.remember_settings):
            layout.addWidget(widget)
        layout.addStretch()
        return scroll

    def _build_menu(self):
        def action(menu, title, callback, shortcut=None):
            entry = QAction(title, self)
            entry.triggered.connect(callback)
            if shortcut:
                entry.setShortcut(QKeySequence(shortcut))
            menu.addAction(entry)
            return entry
        file_menu = self.menuBar().addMenu('File')
        action(file_menu, 'Add images…', self.add_files_dialog, 'Ctrl+O')
        action(file_menu, 'Add folder…', self.add_folder_dialog, 'Ctrl+Shift+O')
        file_menu.addSeparator()
        action(file_menu, 'Save edited image…', self.save_current, 'Ctrl+S')
        action(file_menu, 'Open output folder', self.open_output)
        file_menu.addSeparator()
        action(file_menu, 'Exit', self.close, 'Ctrl+Q')
        processing = self.menuBar().addMenu('Processing')
        action(processing, 'Generate preview', self.generate_preview, 'Ctrl+Return')
        action(processing, 'Process selected images', lambda: self.run_batch(True))
        action(processing, 'Process entire queue', lambda: self.run_batch(False))
        action(processing, 'Stop current job', self.cancel_job, 'Esc')
        view = self.menuBar().addMenu('View')
        action(view, 'Side by side', lambda: self.compare_mode.setCurrentIndex(0))
        action(view, 'Comparison slider', lambda: self.compare_mode.setCurrentIndex(1))
        action(view, 'Fit images', self.canvas.fit, 'F')
        action(view, '100% original pixels', self.canvas.actual_size, 'Ctrl+1')
        action(view, 'Toggle dark / light mode', self.toggle_theme)
        action(view, 'Show processing details', self.toggle_log)
        settings = self.menuBar().addMenu('Settings')
        action(settings, 'Save settings for next launch', self.save_settings)
        action(settings, 'Export settings profile…', self.export_profile)
        action(settings, 'Import settings profile…', self.import_profile)
        action(settings, 'Reset settings to defaults', self.reset_settings)
        help_menu = self.menuBar().addMenu('Help')
        action(help_menu, 'Quick guide', self.show_help)
        action(help_menu, 'Check GPU / Python', self.check_gpu)

    def _set_combo(self, widget, value):
        index = widget.findData(value)
        if index >= 0:
            widget.setCurrentIndex(index)

    def _apply_controls(self, cfg):
        was_initializing = self.initializing
        self.initializing = True
        for task, widget in self.stage_checks.items():
            widget.setChecked(task in cfg['tasks'])
        for widget, key in ((self.compare_mode, 'comparison'), (self.preview_quality, 'preview_edge'),
                            (self.tile_combo, 'tile_size'), (self.overlap_combo, 'overlap'),
                            (self.device_combo, 'device'), (self.precision_combo, 'precision'),
                            (self.format_combo, 'output_format'), (self.bits_combo, 'binary_bits'),
                            (self.conflict_combo, 'conflict')):
            self._set_combo(widget, cfg[key])
        for widget, key in ((self.automatic, 'automatic'), (self.preserve_dpi, 'preserve_dpi'),
                            (self.recursive, 'recursive'), (self.remember_files, 'remember_files'),
                            (self.remember_settings, 'remember_settings')):
            widget.setChecked(cfg[key])
        for widget, key in ((self.output_dir, 'output_dir'), (self.model_path, 'model_path'), (self.engine_path, 'engine_path')):
            widget.setText(cfg[key])
        self.jpeg_quality.setValue(cfg['jpeg_quality'])
        self.debounce.setValue(cfg['debounce'])
        self.wipe_slider.setValue(cfg['wipe'])
        self.initializing = was_initializing
        self._update_export_controls()
        self._comparison_changed()
        self._restyle_stages()

    def snapshot(self):
        cfg = dict(self.cfg)
        cfg['tasks'] = [stage[0] for stage in STAGES if self.stage_checks[stage[0]].isChecked()]
        for widget, key in ((self.compare_mode, 'comparison'), (self.preview_quality, 'preview_edge'),
                            (self.tile_combo, 'tile_size'), (self.overlap_combo, 'overlap'),
                            (self.device_combo, 'device'), (self.precision_combo, 'precision'),
                            (self.format_combo, 'output_format'), (self.bits_combo, 'binary_bits'),
                            (self.conflict_combo, 'conflict')):
            cfg[key] = widget.currentData()
        for widget, key in ((self.automatic, 'automatic'), (self.preserve_dpi, 'preserve_dpi'),
                            (self.recursive, 'recursive'), (self.remember_files, 'remember_files'),
                            (self.remember_settings, 'remember_settings')):
            cfg[key] = widget.isChecked()
        for widget, key in ((self.output_dir, 'output_dir'), (self.model_path, 'model_path'), (self.engine_path, 'engine_path')):
            cfg[key] = widget.text().strip()
        cfg.update(wipe=self.wipe_slider.value(), jpeg_quality=self.jpeg_quality.value(), debounce=self.debounce.value())
        return validated_settings(cfg)

    def all_paths(self):
        return [self.queue.item(index).data(Qt.ItemDataRole.UserRole) for index in range(self.queue.count())]

    def current_path(self):
        item = self.queue.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def selected_paths(self):
        selected = set(item.data(Qt.ItemDataRole.UserRole) for item in self.queue.selectedItems())
        return [path for path in self.all_paths() if path in selected]

    def _restyle_stages(self):
        for task, card in self.stage_cards.items():
            card.setProperty('active', self.stage_checks[task].isChecked())
            card.style().unpolish(card)
            card.style().polish(card)

    def _processing_changed(self, *args):
        if self.initializing:
            return
        # Ensure tile overlap is valid before starting the engine.
        if self.overlap_combo.currentData() >= self.tile_combo.currentData():
            self._set_combo(self.overlap_combo, min(64, self.tile_combo.currentData() // 4))
        self.cfg = self.snapshot()
        self._restyle_stages()
        self._update_export_controls()
        self._mark_preview()
        self._update_buttons()
        self._update_runtime_status()
        self._schedule_settings()
        if self.automatic.isChecked():
            self.preview_timer.start(self.debounce.value())
        else:
            self.pending_auto = False
            self.preview_timer.stop()

    def _export_changed(self, *args):
        if self.initializing:
            return
        self.cfg = self.snapshot()
        self._update_export_controls()
        self._schedule_settings()

    def _update_export_controls(self):
        binary = self.stage_checks['binarization'].isChecked()
        self.format_combo.model().item(2).setEnabled(not binary)
        if binary and self.format_combo.currentData() == 'jpg':
            self._set_combo(self.format_combo, 'png')
        self.bits_combo.setEnabled(binary)
        self.jpeg_quality.setEnabled(self.format_combo.currentData() == 'jpg')
        suffix = {'png': 'png', 'tiff': 'tiff', 'jpg': 'jpg'}[self.format_combo.currentData()]
        tasks = [task for task, widget in self.stage_checks.items() if widget.isChecked()]
        ending = output_label(tasks) if tasks else 'restored'
        self.export_hint.setText('Example: page_{}.{}\n{}\nSource images are never overwritten.'.format(
            ending, suffix, 'Pure black / white. No dithering.' if binary else 'Original pixel dimensions are retained.'))

    def _comparison_changed(self, *args):
        self.canvas.mode = self.compare_mode.currentData()
        self.wipe_row.setVisible(self.canvas.mode == 'slider')
        self.canvas.update()
        if not self.initializing:
            self._schedule_settings()

    def _set_wipe(self, value):
        self.canvas.wipe = value / 100
        if self.wipe_slider.value() != value:
            self.wipe_slider.setValue(value)
        self.canvas.update()
        if not self.initializing:
            self._schedule_settings()

    def apply_theme(self, theme):
        self.cfg['theme'] = theme
        app = QApplication.instance()
        dark = theme == 'dark'
        palette = QPalette()
        for role, color in ((QPalette.ColorRole.Window, '#171f2b' if dark else '#ffffff'),
                            (QPalette.ColorRole.Base, '#171f2b' if dark else '#ffffff'),
                            (QPalette.ColorRole.AlternateBase, '#202b3b' if dark else '#edf2f9'),
                            (QPalette.ColorRole.Text, '#e6edf8' if dark else '#1b2940'),
                            (QPalette.ColorRole.WindowText, '#e6edf8' if dark else '#1b2940'),
                            (QPalette.ColorRole.Button, '#202b3b' if dark else '#edf2f9'),
                            (QPalette.ColorRole.ButtonText, '#e6edf8' if dark else '#1b2940'),
                            (QPalette.ColorRole.Highlight, '#377ce5' if dark else '#e5efff'),
                            (QPalette.ColorRole.HighlightedText, '#ffffff' if dark else '#1b2940')):
            palette.setColor(role, QColor(color))
        app.setPalette(palette)
        app.setStyleSheet(theme_css(theme))
        self.canvas.dark = theme == 'dark'
        self.canvas.update()
        self.theme_button.setText('Light mode' if theme == 'dark' else 'Dark mode')
        self._restyle_stages()

    def toggle_theme(self):
        self.apply_theme('light' if self.cfg['theme'] == 'dark' else 'dark')
        self._schedule_settings()

    def _schedule_settings(self):
        if not self.initializing and self.remember_settings.isChecked():
            self.settings_timer.start()

    def _autosave(self):
        if self.remember_settings.isChecked():
            self.save_settings(notify=False)

    def save_settings(self, checked=False, notify=True):
        data = self.snapshot()
        data['geometry'] = bytes(self.saveGeometry().toBase64()).decode()
        data['files'] = self.all_paths() if data['remember_files'] else []
        try:
            atomic_json(self.config_path, data)
            self.cfg = validated_settings(data)
            if notify:
                self.status.setText('Settings saved for the next launch.')
        except OSError as error:
            self._report_error('Could not save settings: ' + str(error))

    def export_profile(self):
        name, _ = QFileDialog.getSaveFileName(self, 'Export settings profile', 'docres_profile.json',
            'JSON settings (*.json)', options=QFileDialog.Option.DontUseNativeDialog)
        if name:
            try:
                atomic_json(name, self.snapshot())
                self.status.setText('Settings profile exported.')
            except OSError as error:
                self._report_error(str(error))

    def import_profile(self):
        name, _ = QFileDialog.getOpenFileName(self, 'Import settings profile', '',
            'JSON settings (*.json)', options=QFileDialog.Option.DontUseNativeDialog)
        if name:
            try:
                data = json.loads(Path(name).read_text(encoding='utf-8'))
                if not isinstance(data, dict):
                    raise ValueError('The settings file must contain a JSON object.')
                self.cfg = validated_settings(data)
                self._apply_controls(self.cfg)
                self.apply_theme(self.cfg['theme'])
                self._processing_changed()
                self.status.setText('Settings profile loaded.')
            except (OSError, ValueError) as error:
                self._report_error(str(error))

    def reset_settings(self):
        self.cfg = validated_settings({})
        self._apply_controls(self.cfg)
        self.apply_theme('dark')
        self._processing_changed()

    def add_files_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Add document images', '',
            'Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)',
            options=QFileDialog.Option.DontUseNativeDialog)
        self.add_images(paths)

    def add_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, 'Add image folder', '',
            options=QFileDialog.Option.DontUseNativeDialog)
        if folder:
            self.add_images([folder])

    def add_images(self, paths):
        if self.active and self.active['mode'] != 'preview':
            self.status.setText('Wait for the batch to finish, or stop it before changing the queue.')
            return
        try:
            found = collect_images(paths, self.recursive.isChecked())
        except OSError as error:
            self._report_error(str(error))
            return
        existing = set(path.casefold() for path in self.all_paths())
        added = []
        for path in found:
            text = str(path)
            if text.casefold() in existing:
                continue
            existing.add(text.casefold())
            item = QListWidgetItem(path.name + '\nReady')
            item.setData(Qt.ItemDataRole.UserRole, text)
            item.setToolTip(text)
            item.setSizeHint(QSize(1, 82))
            self.queue.addItem(item)
            added.append(item)
            self._load_image(text, 'thumbnail', True)
        if added and not self.queue.currentItem():
            self.queue.setCurrentItem(added[0])
        self._queue_changed()
        if added:
            self.status.setText('{} image(s) added. Select a page to compare.'.format(len(added)))
        elif paths:
            self.status.setText('No new supported images found.')

    def _queue_changed(self, *args):
        self.queue_count.setText(str(self.queue.count()))
        self.batch_button.setText('Process all images ({})'.format(self.queue.count()))
        self._update_buttons()
        self._schedule_settings()

    def remove_selected(self):
        if self.active and self.active['mode'] != 'preview':
            return
        for item in self.queue.selectedItems():
            self.queue.takeItem(self.queue.row(item))
        self._queue_changed()
        if self.queue.count() == 0:
            self._clear_view()

    def clear_queue(self):
        if self.active and self.active['mode'] != 'preview':
            return
        self.queue.clear()
        self._clear_view()
        self._queue_changed()

    def _clear_view(self):
        self.preview_timer.stop()
        self.canvas.set_images(QImage(), QImage(), reset=True)
        self.current_signature = None
        self.image_info.setText('No image selected')
        self.preview_badge.setText('READY TO GENERATE')

    def _selection_changed(self, current, previous):
        if not current:
            self._clear_view()
            return
        source = current.data(Qt.ItemDataRole.UserRole)
        self.canvas.set_images(QImage(), QImage(), reset=True)
        self.image_info.setText('Loading ' + Path(source).name + '…')
        self._load_image(source, 'original')
        self._mark_preview()
        self._update_buttons()
        if self.automatic.isChecked():
            self.preview_timer.start(self.debounce.value())

    def _load_image(self, path, role, thumbnail=False):
        if any(task.path == str(path) and task.role == role for task in self.loader_tasks):
            return
        task = ImageLoader(path, role, thumbnail)
        task.signals.done.connect(self._image_loaded)
        task.signals.failed.connect(self._image_failed)
        self.loader_tasks.add(task)
        self.pool.start(task)

    def _remove_loader(self, path, role):
        self.loader_tasks = {task for task in self.loader_tasks if not (task.path == path and task.role == role)}

    def _image_loaded(self, path, role, image, metadata):
        self._remove_loader(path, role)
        if self.closing:
            return
        if role == 'thumbnail':
            for index in range(self.queue.count()):
                item = self.queue.item(index)
                if item.data(Qt.ItemDataRole.UserRole) == path:
                    item.setIcon(QIcon(QPixmap.fromImage(image)))
                    item.setToolTip(path + '\n' + metadata)
                    break
        elif role == 'original' and path == self.current_path():
            self.canvas.set_images(original=image)
            self.image_info.setText(metadata + ' · ' + Path(path).name)
        elif role.startswith('edited:'):
            signature = role.split(':', 1)[1]
            source = self.current_path()
            if source and self.current_signature == signature:
                self.canvas.set_images(edited=image)

    def _image_failed(self, path, role, message):
        self._remove_loader(path, role)
        if role != 'thumbnail' and not self.closing:
            self._report_error('Could not display {}: {}'.format(Path(path).name, message))

    def _mark_preview(self):
        source = self.current_path()
        if not source:
            return
        try:
            signature = processing_signature(source, self.snapshot())
        except OSError:
            self.preview_badge.setText('SOURCE MISSING')
            return
        cached = self.preview_cache.get(source)
        if cached and cached['signature'] == signature and Path(cached['path']).is_file():
            self.current_signature = signature
            self.preview_badge.setText('FULL RESOLUTION' if cached['full_resolution'] else 'PREVIEW')
            self._load_image(cached['path'], 'edited:' + signature)
        else:
            self.current_signature = None
            self.preview_badge.setText('OUT OF DATE' if not self.canvas.edited.isNull() else 'READY TO GENERATE')

    def _update_buttons(self):
        busy = self.active is not None
        has_image = bool(self.current_path())
        has_tasks = any(widget.isChecked() for widget in self.stage_checks.values())
        can_run = has_image and has_tasks and not busy
        self.generate_button.setEnabled(can_run)
        self.save_button.setEnabled(can_run)
        self.selected_button.setEnabled(bool(self.selected_paths()) and has_tasks and not busy)
        self.batch_button.setEnabled(self.queue.count() > 0 and has_tasks and not busy)
        self.stop_button.setEnabled(busy)
        self.gpu_check_button.setEnabled(not busy)
        batch_busy = busy and self.active['mode'] != 'preview'
        self.tabs.setEnabled(not batch_busy)
        self.remove_button.setEnabled(not batch_busy and bool(self.selected_paths()))
        self.clear_button.setEnabled(not batch_busy and self.queue.count() > 0)

    def _auto_preview(self):
        if not self.automatic.isChecked() or not self.current_path():
            return
        if self.active:
            self.pending_auto = self.active['mode'] == 'preview'
        else:
            self.generate_preview(automatic=True)

    def generate_preview(self, checked=False, automatic=False):
        if self.active:
            return
        source = self.current_path()
        if not source:
            return
        cfg = self.snapshot()
        try:
            signature = processing_signature(source, cfg)
        except OSError as error:
            self._report_error('Source image is unavailable: ' + str(error))
            return
        if automatic:
            cached = self.preview_cache.get(source)
            if cached and cached['signature'] == signature and Path(cached['path']).is_file():
                return
        self._start_job('preview', [source], cfg, signature=signature)

    def run_batch(self, selected=False):
        if self.active:
            return
        paths = self.selected_paths() if selected else self.all_paths()
        if paths:
            self._start_job('batch', paths, self.snapshot())

    def save_current(self):
        if self.active or not self.current_path():
            return
        cfg = self.snapshot()
        binary = 'binarization' in cfg['tasks']
        filters = 'PNG image (*.png);;TIFF image (*.tiff *.tif)'
        if not binary:
            filters += ';;JPEG image (*.jpg *.jpeg)'
        suffix = {'png': '.png', 'tiff': '.tiff', 'jpg': '.jpg'}[cfg['output_format']]
        filename = '{}_{}{}'.format(Path(self.current_path()).stem, output_label(cfg['tasks']), suffix)
        destination, selected_filter = QFileDialog.getSaveFileName(self, 'Save full-resolution edited image',
            str(Path(cfg['output_dir']) / filename), filters,
            options=QFileDialog.Option.DontUseNativeDialog)
        if not destination:
            return
        if not Path(destination).suffix:
            destination += '.tiff' if selected_filter.startswith('TIFF') else '.jpg' if selected_filter.startswith('JPEG') else '.png'
        if str(Path(destination).resolve()).casefold() in {path.casefold() for path in self.all_paths()}:
            self._report_error('Choose a different filename. Source images are protected.')
            return
        cfg['output_format'] = {'.png': 'png', '.tif': 'tiff', '.tiff': 'tiff',
                                '.jpg': 'jpg', '.jpeg': 'jpg'}.get(Path(destination).suffix.lower(), '')
        if not cfg['output_format'] or (binary and cfg['output_format'] == 'jpg'):
            self._report_error('Choose PNG or TIFF for binary output; PNG, TIFF or JPEG otherwise.')
            return
        self._start_job('save_one', [self.current_path()], cfg, destination=destination,
                        signature=processing_signature(self.current_path(), self.snapshot()))

    def _start_job(self, mode, paths, cfg, **extra):
        if not cfg['tasks']:
            self._report_error('Check at least one restoration step in the Pipeline tab.')
            return
        if not Path(cfg['engine_path']).is_file() or not Path(cfg['model_path']).is_file():
            self._report_error('Select valid inference.py and model checkpoint paths in the Engine tab.')
            self.tabs.setCurrentIndex(2)
            return
        if cfg['device'] == 'cpu' and 'dewarping' in cfg['tasks']:
            self._report_error('The MBD dewarping helper requires CUDA. Choose Auto/CUDA or uncheck Dewarp.')
            return
        if cfg['device'] == 'cuda' and self.runtime_info is not None and not self.runtime_info['cuda_usable']:
            self._report_error('GPU processing is unavailable: ' + self.runtime_info['reason'])
            self.tabs.setCurrentIndex(2)
            return
        self.preview_timer.stop()
        self.pending_auto = False
        self.cancelled = False
        job_id = uuid.uuid4().hex
        request = dict(op='run', id=job_id, mode=mode, inputs=paths, settings=cfg, **extra)
        request['protected_sources'] = self.all_paths()
        request['preview_path'] = str(Path(self.temp.name) / (job_id + '.png'))
        self.active = request
        if mode == 'preview':
            self.preview_badge.setText('GENERATING…')
        self.status.setText('Generating preview…' if mode == 'preview' else 'Processing full-resolution images…')
        self.progress.setRange(0, 0)
        self._update_buttons()
        self._ensure_worker(cfg['engine_path'])
        if self.worker.state() == QProcess.ProcessState.Running:
            self._send_active()

    def _ensure_worker(self, engine_path):
        engine = str(Path(engine_path).resolve())
        if self.worker and self.worker_engine == engine and self.worker.state() != QProcess.ProcessState.NotRunning:
            return
        if self.worker:
            old = self.worker
            old.blockSignals(True)
            old.kill()
            old.waitForFinished(3000)
            old.deleteLater()
        self.stdout_buffer = ''
        self.worker_engine = engine
        self.runtime_info = None
        self.diagnose_pending = True
        process = QProcess(self)
        self.worker = process
        process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        process.setWorkingDirectory(str(ROOT))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert('PYTHONUNBUFFERED', '1')
        environment.insert('PYTHONIOENCODING', 'utf-8')
        process.setProcessEnvironment(environment)
        executable = Path(sys.executable)
        if executable.name.lower() == 'pythonw.exe' and executable.with_name('python.exe').is_file():
            executable = executable.with_name('python.exe')
        process.setProgram(str(executable))
        process.setArguments([str(Path(__file__).resolve()), '--worker'])
        process.started.connect(self._worker_started)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(self._worker_exited)
        process.errorOccurred.connect(self._worker_error)
        process.start()

    def _worker_started(self):
        self._send_diagnosis()
        self._send_active()

    def _send_diagnosis(self):
        if self.diagnose_pending and self.worker and self.worker.state() == QProcess.ProcessState.Running:
            self.worker.write(b'{"op":"diagnose"}\n')
            self.diagnose_pending = False

    def check_gpu(self):
        if self.closing:
            return
        if self.active:
            self.status.setText('Finish or stop the current job before checking the runtime.')
            return
        self.diagnose_pending = True
        self.gpu_status.setText('Checking GPU / Python…')
        self.engine_badge.setText('CHECKING GPU')
        self._ensure_worker(self.engine_path.text())
        self._send_diagnosis()

    def copy_gpu_report(self):
        if self.runtime_info is None:
            self.check_gpu()
            return
        text = runtime_text(self.runtime_info) + '\nSelected processing device: ' + self.device_combo.currentData()
        QApplication.instance().clipboard().setText(text)
        self.status.setText('GPU / Python diagnostic report copied.')

    def _update_runtime_status(self):
        if self.runtime_info is None:
            return
        report = self.runtime_info
        selected = self.device_combo.currentData()
        if report['cuda_usable']:
            summary = '{} · {:.1f} GiB\nPyTorch {} · CUDA {}'.format(
                report['gpu_name'], report['gpu_memory'] / 1024 ** 3, report['torch_version'], report['cuda_build'])
            badge = 'CPU SELECTED' if selected == 'cpu' else 'GPU AVAILABLE'
        else:
            summary = 'CUDA unavailable\n' + report['reason']
            badge = 'CPU SELECTED' if selected == 'cpu' else 'AUTO → CPU' if selected == 'auto' else 'CUDA UNAVAILABLE'
        self.gpu_status.setText(summary)
        self.gpu_status.setToolTip(runtime_text(report))
        self.gpu_copy_button.setEnabled(True)
        if not self.active:
            self.engine_badge.setText(badge)
        self.engine_badge.setToolTip(runtime_text(report))

    def _send_active(self):
        if self.active and not self.active.get('_sent') and self.worker:
            payload = {key: value for key, value in self.active.items() if not key.startswith('_')}
            self.worker.write((json.dumps(payload, ensure_ascii=True) + '\n').encode('utf-8'))
            self.active['_sent'] = True

    def _read_stdout(self):
        self.stdout_buffer += bytes(self.worker.readAllStandardOutput()).decode('utf-8', errors='replace')
        while '\n' in self.stdout_buffer:
            line, self.stdout_buffer = self.stdout_buffer.split('\n', 1)
            if line.startswith(PROTOCOL):
                try:
                    self._handle_event(json.loads(line[len(PROTOCOL):]))
                except (ValueError, KeyError) as error:
                    self.log.appendPlainText('Invalid worker message: ' + str(error))
            elif line.strip():
                self.log.appendPlainText(line)

    def _read_stderr(self):
        message = bytes(self.worker.readAllStandardError()).decode('utf-8', errors='replace').strip()
        if message:
            self.log.appendPlainText(message)
            logging.warning(message)

    def _handle_event(self, event):
        kind = event['event']
        if kind == 'ready':
            return
        if kind == 'runtime':
            self.runtime_info = event['report']
            self._update_runtime_status()
            self.log.appendPlainText(runtime_text(self.runtime_info))
            if not self.active and self.device_combo.currentData() == 'cuda' and not self.runtime_info['cuda_usable']:
                self.status.setText('CUDA is unavailable in this Python. See Engine → Check GPU / Python.')
            return
        if kind == 'warning':
            self.log.appendPlainText('WARNING: ' + event['message'])
            self.log.show()
            self.log_button.setText('Hide details')
            return
        if kind in ('loading', 'engine'):
            self.status.setText(event['message'])
            if kind == 'engine':
                self.engine_badge.setText(event['device'].upper() + ' READY')
            return
        if not self.active or event.get('id') != self.active['id']:
            return
        if kind == 'stage':
            self.status.setText(event['message'] + ' · ' + Path(event['source']).name)
            self.progress.setRange(0, event['maximum'])
            self.progress.setValue(event['value'])
        elif kind == 'item_start':
            self._set_item_status(event['source'], 'Processing…')
        elif kind == 'result':
            self.progress.setRange(0, event['maximum'])
            self.progress.setValue(event['value'])
            size = self._size_text(event['bytes'])
            self._set_item_status(event['source'], 'Preview ready' if event['preview'] else 'Saved · ' + size)
            self.log.appendPlainText('{} → {} ({})'.format(Path(event['source']).name, event['path'], size))
            signature = self.active.get('signature')
            if not signature:
                try:
                    signature = processing_signature(event['source'], self.active['settings'])
                except OSError:
                    signature = None
            if signature:
                self.preview_cache[event['source']] = dict(path=event['path'], signature=signature,
                                                           full_resolution=event['full_resolution'])
                if event['source'] == self.current_path():
                    self._mark_preview()
        elif kind == 'skipped':
            self._set_item_status(event['source'], 'Skipped · already exists')
        elif kind == 'item_error':
            self._set_item_status(event['source'], 'Failed')
            self._report_error(event['message'])
        elif kind == 'fatal':
            self._report_error(event['message'])
            self._finish_job(failed=True)
        elif kind == 'complete':
            if self.active['mode'] == 'preview' and event['success']:
                self.status.setText('Preview ready · {:.1f}s · exports use the full-resolution original'.format(event['seconds']))
            else:
                self.status.setText('{} saved · {} skipped · {} failed · {} · {:.1f}s'.format(
                    event['success'], event['skipped'], event['failures'], self._size_text(event['bytes']), event['seconds']))
            self._finish_job(failed=bool(event['failures']))

    def _finish_job(self, failed=False):
        self.active = None
        self.progress.setRange(0, 100)
        self.progress.setValue(0 if failed else 100)
        self._mark_preview()
        self._update_buttons()
        if failed:
            self._update_runtime_status()
        if self.pending_auto and self.automatic.isChecked() and not self.cancelled:
            self.pending_auto = False
            self.preview_timer.start(self.debounce.value())

    def cancel_job(self):
        if not self.active:
            return
        self.cancelled = True
        self.pending_auto = False
        self.preview_timer.stop()
        self.status.setText('Stopping… completed exports will be kept.')
        self.worker.kill()

    def _worker_exited(self, code, status):
        if self.closing:
            return
        self._read_stdout()
        self._read_stderr()
        job = self.active
        if job:
            folder = Path(job.get('destination', job['settings']['output_dir'])).parent \
                if job['mode'] == 'save_one' else Path(job['settings']['output_dir'])
            if job['mode'] == 'preview':
                folder = Path(self.temp.name)
            for partial in folder.glob('.docres-{}-*.part'.format(job['id'])):
                with contextlib.suppress(OSError):
                    partial.unlink()
            if self.cancelled:
                self.status.setText('Stopped. Completed exports were kept; GPU worker was released.')
            else:
                self._report_error('Processing worker exited unexpectedly. Open Details for the error log.')
            for index in range(self.queue.count()):
                item = self.queue.item(index)
                if item.text().endswith('\nProcessing…'):
                    self._set_item_status(item.data(Qt.ItemDataRole.UserRole),
                                          'Stopped' if self.cancelled else 'Failed')
            self._finish_job(failed=True)
        self.engine_badge.setText('ENGINE IDLE')
        self._update_runtime_status()

    def _worker_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self._report_error('Could not start the processing worker: ' + self.worker.errorString())
            self._finish_job(failed=True)

    def _set_item_status(self, source, status):
        for index in range(self.queue.count()):
            item = self.queue.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == source:
                item.setText(Path(source).name + '\n' + status)
                break

    @staticmethod
    def _size_text(size):
        value = float(size)
        for unit in ('B', 'KiB', 'MiB', 'GiB'):
            if value < 1024 or unit == 'GiB':
                return '{:.1f} {}'.format(value, unit)
            value /= 1024

    def _report_error(self, message):
        self.status.setText(message)
        self.log.appendPlainText('ERROR: ' + message)
        self.log.show()
        self.log_button.setText('Hide details')
        logging.error(message)

    def toggle_log(self):
        self.log.setVisible(not self.log.isVisible())
        self.log_button.setText('Hide details' if self.log.isVisible() else 'Details')

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, 'Choose output folder', self.output_dir.text(),
            options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            self.output_dir.setText(path)
            self._export_changed()

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select DocRes model checkpoint', self.model_path.text(),
            'Model checkpoints (*.pkl *.pth *.pt)', options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            self.model_path.setText(path)
            self._processing_changed()

    def browse_engine(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select tiled inference.py', self.engine_path.text(),
            'Python inference engine (*.py)', options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            self.engine_path.setText(path)
            self._processing_changed()

    def open_output(self):
        path = Path(self.output_dir.text())
        try:
            path.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
        except OSError as error:
            self._report_error(str(error))

    def show_help(self):
        QMessageBox.information(self, 'DocRes Studio · Quick guide',
            '1. Add images or a folder. Drag queue items to change batch order.\n'
            '2. Check the restoration steps you need; hover for explanations.\n'
            '3. Generate a preview, or enable Auto-generate preview.\n'
            '4. Compare side by side or use the draggable comparison slider.\n'
            '5. Save the current image, process selected images, or run the entire queue.\n\n'
            'Save and batch always use the full-resolution input. Binary PNG/TIFF can be packed to 1 bit without dithering.\n\n'
            'Scroll to zoom; drag to pan; double-click to fit.\n'
            'Ctrl+O: add images · Ctrl+Enter: preview · Ctrl+S: save · Esc: stop.\n\n'
            'Dewarping requires the fork\'s MBD checkpoint and CUDA. '
            'This app uses the local DocRes model; it does not upload your images.\n\n'
            'Settings and logs: ' + str(self.config_path.parent))

    def closeEvent(self, event):
        if self.active:
            answer = QMessageBox.question(self, 'Stop processing?',
                'Stop the active job and close? Completed exports will be kept.',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self.closing = True
        self.settings_timer.stop()
        self.preview_timer.stop()
        if self.remember_settings.isChecked():
            self.save_settings(notify=False)
        if self.worker and self.worker.state() != QProcess.ProcessState.NotRunning:
            self.worker.blockSignals(True)
            self.worker.kill()
            self.worker.waitForFinished(3000)
        self.pool.clear()
        self.pool.waitForDone(5000)
        self.temp.cleanup()
        event.accept()


def gui_main():
    import argparse
    parser = argparse.ArgumentParser(description='DocRes Studio desktop workspace.')
    parser.add_argument('--device', choices=('cuda', 'auto', 'cpu'), default=None,
                        help='Override the saved device for this launch. The BAT uses cuda by default.')
    arguments, qt_arguments = parser.parse_known_args()
    app = QApplication([sys.argv[0]] + qt_arguments)
    app.setApplicationName('DocRes Studio')
    app.setOrganizationName('DocResStudio')
    app.setStyle('Fusion')
    app.setFont(QFont('Segoe UI', 10))
    log_dir = app_directory() / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(handlers=[logging.FileHandler(str(log_dir / 'studio.log'), encoding='utf-8')],
                        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    def exception_hook(kind, value, tb):
        logging.error('Unhandled exception', exc_info=(kind, value, tb))
        QMessageBox.critical(None, 'DocRes Studio error', str(value) + '\n\nDetails were written to the application log.')
    sys.excepthook = exception_hook
    window = StudioWindow(device_override=arguments.device)
    available = app.primaryScreen().availableGeometry()
    window.resize(min(window.width(), available.width() - 30),
                  min(window.height(), available.height() - 50))
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(gui_main())
