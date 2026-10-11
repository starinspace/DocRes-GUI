import cv2
import gc
import sys
import time
from contextlib import nullcontext
from pathlib import Path
import utils
import argparse
import numpy as np
from PIL import Image

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def dewarp_prompt(img):
    # Load MBD only for dewarping, so other tasks do not load its networks.
    mbd_dir = str(Path(__file__).resolve().parent / 'data' / 'MBD')
    if mbd_dir not in sys.path:
        sys.path.append(mbd_dir)
    from data.MBD.infer import net1_net2_infer_single_im

    mask = net1_net2_infer_single_im(img,'models/mbd.pkl')
    base_coord = utils.getBasecoord(256,256)/256
    img[mask==0]=0
    mask = cv2.resize(mask,(256,256))/255
    return img,np.concatenate((base_coord,np.expand_dims(mask,-1)),-1)

def deshadow_prompt(img):
    h,w = img.shape[:2]
    # img = cv2.resize(img,(128,128))
    img = cv2.resize(img,(1024,1024))
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    bg_imgs = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        bg_imgs.append(bg_img)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    bg_imgs = cv2.merge(bg_imgs)
    bg_imgs = cv2.resize(bg_imgs,(w,h))
    # result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    result_norm[result_norm==0]=1
    shadow_map = np.clip(img.astype(float)/result_norm.astype(float)*255,0,255).astype(np.uint8)
    shadow_map = cv2.resize(shadow_map,(w,h))
    shadow_map = cv2.cvtColor(shadow_map,cv2.COLOR_BGR2GRAY)
    shadow_map = cv2.cvtColor(shadow_map,cv2.COLOR_GRAY2BGR)
    # return shadow_map
    return bg_imgs

def deblur_prompt(img):
    x = cv2.Sobel(img,cv2.CV_16S,1,0)  
    y = cv2.Sobel(img,cv2.CV_16S,0,1)  
    absX = cv2.convertScaleAbs(x)   # 转回uint8  
    absY = cv2.convertScaleAbs(y)  
    high_frequency = cv2.addWeighted(absX,0.5,absY,0.5,0)
    high_frequency = cv2.cvtColor(high_frequency,cv2.COLOR_BGR2GRAY)
    high_frequency = cv2.cvtColor(high_frequency,cv2.COLOR_GRAY2BGR)
    return high_frequency

def appearance_prompt(img):
    h,w = img.shape[:2]
    # img = cv2.resize(img,(128,128))
    img = cv2.resize(img,(1024,1024))
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    result_norm = cv2.merge(result_norm_planes)
    result_norm = cv2.resize(result_norm,(w,h))
    return result_norm

def binarization_promptv2(img):
    result,thresh = utils.SauvolaModBinarization(img)
    thresh = thresh.astype(np.uint8)
    result[result>155]=255
    result[result<=155]=0

    x = cv2.Sobel(img,cv2.CV_16S,1,0)  
    y = cv2.Sobel(img,cv2.CV_16S,0,1)  
    absX = cv2.convertScaleAbs(x)   # 转回uint8  
    absY = cv2.convertScaleAbs(y)  
    high_frequency = cv2.addWeighted(absX,0.5,absY,0.5,0)
    high_frequency = cv2.cvtColor(high_frequency,cv2.COLOR_BGR2GRAY)
    return np.concatenate((np.expand_dims(thresh,-1),np.expand_dims(high_frequency,-1),np.expand_dims(result,-1)),-1)


# Restormer uses three downsampling levels: sizes and tile origins must be
# divisible by 8. Only one tile is on the GPU; blending happens in CPU RAM.
FACTOR = 8
MIN_TILE_SIZE = 128
DEFAULT_TILE_SIZE = 512
TASKS = ('dewarping', 'deshadowing', 'appearance', 'deblurring',
         'binarization', 'end2end')
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def _read_image(source):
    # Arrays let end2end pass images between stages without JPEG round trips.
    image = source if isinstance(source, np.ndarray) else cv2.imread(str(source))
    if image is None:
        raise ValueError('Cannot read image: {}'.format(source))
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError('Expected a non-empty BGR image with three channels.')
    return image


def _model_device(model):
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else DEVICE


def _inference_context():
    # inference_mode was introduced in PyTorch 1.9; support older forks too.
    if hasattr(torch, 'inference_mode'):
        return torch.inference_mode()
    return torch.no_grad()


def _amp_context(device, precision):
    if device.type != 'cuda' or precision == 'fp32':
        return nullcontext()
    if hasattr(torch, 'autocast'):
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    return torch.cuda.amp.autocast()


def _tile_settings(memory_fix, tile_size, tile_overlap):
    presets = {0: DEFAULT_TILE_SIZE, 1: 256, 2: 512, 3: 768}
    if memory_fix not in presets:
        raise ValueError('memory_fix must be 0, 1, 2 or 3.')
    size = presets[memory_fix] if tile_size is None else tile_size
    if size < MIN_TILE_SIZE or size % FACTOR:
        raise ValueError('tile_size must be at least 128 and divisible by 8.')
    if tile_overlap < 0 or tile_overlap % FACTOR or tile_overlap >= size:
        raise ValueError('tile_overlap must be nonnegative, divisible by 8, '
                         'and smaller than tile_size.')
    return size


def _pad_to_factor(array):
    h, w = array.shape[:2]
    return np.pad(array, ((0, (-h) % FACTOR), (0, (-w) % FACTOR), (0, 0)),
                  mode='symmetric')


def _tile_starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, tile_size - overlap))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def _blend_axis(length, overlap, first, last):
    weights = np.ones(length, dtype=np.float32)
    ramp_size = min(overlap, length // 2)
    if ramp_size:
        ramp = np.arange(1, ramp_size + 1, dtype=np.float32) / (ramp_size + 1)
        if not first:
            weights[:ramp_size] = ramp
        if not last:
            weights[-ramp_size:] = ramp[::-1]
    return weights


def _forward_patch(model, image, prompt, precision, binarize):
    # Convert on the CPU FIRST. The original code copied a float64 full page
    # to CUDA before converting it to half, creating a large memory spike.
    patch = np.concatenate((image, prompt), axis=2).astype(np.float32)
    patch *= np.float32(1.0 / 255.0)
    tensor = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)))
    tensor = tensor.unsqueeze(0).to(_model_device(model))
    with _inference_context():
        with _amp_context(tensor.device, precision):
            prediction = model(tensor)
        channels = 2 if binarize else 3
        if (prediction.ndim != 4 or prediction.shape[0] != 1
                or prediction.shape[1] < channels
                or tuple(prediction.shape[2:]) != image.shape[:2]):
            raise RuntimeError('Unexpected Restormer output shape: {}'.format(
                tuple(prediction.shape)))
        if binarize:
            # argmax(softmax(logits)) == argmax(logits). Blend the logit
            # difference before thresholding, so overlapping tiles do not
            # create gray bands in the binary output. Ties remain class 0.
            scores = prediction[0, :2].float()
            result = (scores[1] - scores[0]).unsqueeze(-1).cpu().numpy()
        else:
            result = prediction[0, :3].float().clamp(0, 1)
            result = result.permute(1, 2, 0).cpu().numpy()
    if not np.isfinite(result).all():
        raise RuntimeError('Model produced NaN/Inf. Try --precision fp32.')
    return result


def _tiled_pass(model, image, prompt, size, overlap, precision, binarize):
    h, w = image.shape[:2]
    ys = _tile_starts(h, size, overlap)
    xs = _tile_starts(w, size, overlap)
    channels = 1 if binarize else 3
    summed = np.zeros((h, w, channels), dtype=np.float32)
    weights = np.zeros((h, w, 1), dtype=np.float32)
    for y in ys:
        y2 = min(y + size, h)
        wy = _blend_axis(y2 - y, overlap, y == 0, y2 == h)
        for x in xs:
            x2 = min(x + size, w)
            wx = _blend_axis(x2 - x, overlap, x == 0, x2 == w)
            tile_weight = (wy[:, None] * wx[None, :])[:, :, None]
            result = _forward_patch(model, image[y:y2, x:x2],
                                    prompt[y:y2, x:x2], precision, binarize)
            summed[y:y2, x:x2] += result * tile_weight
            weights[y:y2, x:x2] += tile_weight
            del result
    if np.any(weights <= 0):
        raise RuntimeError('Tile coverage error: some pixels were not processed.')
    summed /= weights
    return summed


def _is_cuda_oom(error):
    oom_class = getattr(torch.cuda, 'OutOfMemoryError', None)
    if oom_class is not None and isinstance(error, oom_class):
        return True
    message = str(error).lower()
    return ('cuda out of memory' in message
            or 'cuda error: out of memory' in message)


def _predict_tiled(model, image, prompt, memory_fix=0, tile_size=None,
                   tile_overlap=64, precision='auto', binarize=False):
    size = _tile_settings(memory_fix, tile_size, tile_overlap)
    if precision not in ('auto', 'fp32'):
        raise ValueError('precision must be auto or fp32.')
    if prompt.shape != image.shape:
        raise ValueError('Prompt shape must match the input image.')
    size = min(size, getattr(model, '_docres_safe_tile_size', size))
    original_h, original_w = image.shape[:2]
    image = _pad_to_factor(image)
    prompt = _pad_to_factor(prompt)
    model.float().eval()
    device = _model_device(model)
    while True:
        # Reduce overlap as tiles shrink, retaining aligned tile origins.
        overlap = min(tile_overlap, (size // 4) // FACTOR * FACTOR)
        try:
            prediction = _tiled_pass(model, image, prompt, size, overlap,
                                     precision, binarize)
            return prediction[:original_h, :original_w]
        except RuntimeError as error:
            if device.type != 'cuda' or not _is_cuda_oom(error):
                raise
        # Recovery must happen AFTER leaving except. An exception traceback
        # retains the failed forward's tensors until its references are freed.
        gc.collect()
        torch.cuda.empty_cache()
        if size <= MIN_TILE_SIZE:
            raise RuntimeError('CUDA out of memory even with 128px tiles. '
                               'Free GPU memory or run with --device cpu.')
        size = max(MIN_TILE_SIZE, (size // 2) // FACTOR * FACTOR)
        model._docres_safe_tile_size = size
        print('CUDA OOM: retrying with {}x{} tiles.'.format(size, size), flush=True)


def dewarping(model, im_path, memory_fix=0, **options):
    # Dewarping predicts a GLOBAL coordinate map from a fixed 256px image.
    # Tiling that map would break page geometry. Remap the original image.
    input_size = 256
    image = _read_image(im_path)
    h, w = image.shape[:2]
    masked, prompt = dewarp_prompt(image.copy())
    masked = cv2.resize(masked, (input_size, input_size)).astype(np.float32)
    masked *= np.float32(1.0 / 255.0)
    model.float().eval()
    device = _model_device(model)
    image_tensor = torch.from_numpy(masked.transpose(2, 0, 1)).unsqueeze(0)
    prompt_tensor = torch.from_numpy(
        prompt.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0)
    tensor = torch.cat((image_tensor, prompt_tensor), dim=1).to(device)
    with _inference_context():
        prediction = model(tensor)[0, :2].permute(1, 2, 0).float().cpu().numpy()
    prediction = prediction + utils.getBasecoord(input_size, input_size) / input_size
    for _ in range(15):
        prediction = cv2.blur(prediction, (3, 3), borderType=cv2.BORDER_REPLICATE)
    prediction = (cv2.resize(prediction, (w, h)) * (w, h)).astype(np.float32)
    output = cv2.remap(image, prediction[:, :, 0], prediction[:, :, 1],
                       cv2.INTER_LINEAR)
    prompt = cv2.resize((prompt * 255).astype(np.uint8), (w, h))
    return prompt[:, :, 0], prompt[:, :, 1], prompt[:, :, 2], output


def _restore_detail(model, im_path, prompt_function, memory_fix=0,
                    binarize=False, **options):
    image = _read_image(im_path)
    prompt = prompt_function(image)
    prediction = _predict_tiled(model, image, prompt, memory_fix=memory_fix,
                                binarize=binarize, **options)
    if binarize:
        output = (prediction[:, :, 0] > 0).astype(np.uint8) * 255
    else:
        output = (np.clip(prediction, 0, 1) * 255).astype(np.uint8)
    return prompt[:, :, 0], prompt[:, :, 1], prompt[:, :, 2], output


def _restore_lighting(model, im_path, prompt_function, memory_fix=0, **options):
    # Preserve the fork's global lighting-map method. Only its model forward
    # is tiled. Text/detail correction is applied to the original-size image.
    max_size = {0: 1600, 1: 1500, 2: 2000, 3: 3000}[memory_fix]
    image = _read_image(im_path)
    h, w = image.shape[:2]
    prompt = prompt_function(image)
    if max(h, w) > max_size:
        scale = float(max_size) / max(h, w)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized_image = cv2.resize(image, (new_w, new_h))
        resized_prompt = cv2.resize(prompt, (new_w, new_h))
        prediction = _predict_tiled(model, resized_image, resized_prompt,
                                    memory_fix=memory_fix, **options)
        prediction = np.maximum((prediction * 255).astype(np.uint8), 1)
        shadow_map = resized_image.astype(np.float32) / prediction.astype(np.float32)
        shadow_map = cv2.resize(shadow_map, (w, h))
        np.maximum(shadow_map, np.float32(0.00001), out=shadow_map)
        output = np.clip(image.astype(np.float32) / shadow_map, 0, 255).astype(np.uint8)
    else:
        prediction = _predict_tiled(model, image, prompt, memory_fix=memory_fix,
                                    **options)
        output = (prediction * 255).astype(np.uint8)
    return prompt[:, :, 0], prompt[:, :, 1], prompt[:, :, 2], output


def appearance(model, im_path, memory_fix=0, **options):
    return _restore_lighting(model, im_path, appearance_prompt, memory_fix, **options)


def deshadowing(model, im_path, memory_fix=0, **options):
    return _restore_lighting(model, im_path, deshadow_prompt, memory_fix, **options)


def deblurring(model, im_path, memory_fix=0, **options):
    return _restore_detail(model, im_path, deblur_prompt, memory_fix, **options)


def binarization(model, im_path, memory_fix=0, **options):
    return _restore_detail(model, im_path, binarization_promptv2, memory_fix,
                           binarize=True, **options)


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='DocRes inference with bounded GPU memory.')
    parser.add_argument('--model_path', nargs='?', default='models/docres.pkl',
                        help='Path of the saved checkpoint.')
    parser.add_argument('--im_path', nargs='?', default='./distorted/',
                        help='Input image or directory.')
    parser.add_argument('--out_folder', nargs='?', default='./output/',
                        help='Output directory.')
    parser.add_argument('--task', nargs='?', choices=TASKS, default='dewarping')
    parser.add_argument('--save_dtsprompt', nargs='?', type=int, default=0,
                        help='Save prompt images when nonzero.')
    parser.add_argument('--memory_fix', nargs='?', type=int, choices=(0, 1, 2, 3),
                        default=0, help='Tile presets: 0=512, 1=256, 2=512, 3=768px. '
                        'Lighting tasks keep their previous resize limits.')
    parser.add_argument('--tile_size', type=int, default=None,
                        help='Maximum tile edge, divisible by 8, minimum 128. '
                        'Overrides memory_fix; default 512.')
    parser.add_argument('--tile_overlap', type=int, default=64,
                        help='Tile overlap, divisible by 8; default 64. '
                        'Limited to one quarter of the current tile size.')
    parser.add_argument('--precision', choices=('auto', 'fp32'), default='auto',
                        help='auto=CUDA mixed precision / CPU float32; '
                        'fp32=full precision.')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto',
                        help='Restormer device; default auto. '
                        'The separate MBD dewarping helper controls its own device.')
    parser.add_argument('--output_format', choices=('png', 'tiff', 'jpg'), default='png',
                        help='Output format; default lossless PNG. '
                        'Binarization supports PNG and TIFF.')
    parser.add_argument('--binary_bits', type=int, choices=(1, 8), default=1,
                        help='Binarization storage: 1=packed black/white (default), '
                        '8=grayscale containing only black/white pixels. '
                        'Does not affect other tasks.')
    parser.add_argument('--jpeg_quality', type=int, default=95,
                        help='JPEG quality from 1 to 100; default 95.')
    args = parser.parse_args(argv)
    try:
        _tile_settings(args.memory_fix, args.tile_size, args.tile_overlap)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Use --device cpu or --device auto.')
    if args.task == 'binarization' and args.output_format == 'jpg':
        parser.error('Binarization requires lossless PNG or TIFF; JPEG alters binary pixels.')
    if not 1 <= args.jpeg_quality <= 100:
        parser.error('jpeg_quality must be between 1 and 100.')
    return args


def model_init(args):
    from utils import convert_state_dict
    from models import restormer_arch

    model = restormer_arch.Restormer(
        inp_channels=6, out_channels=3, dim=48,
        num_blocks=[2, 3, 3, 4], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias', dual_pixel_task=True)
    # Load on CPU to avoid temporarily keeping TWO copies of weights on CUDA.
    checkpoint = torch.load(args.model_path, map_location='cpu')
    state = convert_state_dict(checkpoint['model_state'])
    model.load_state_dict(state)
    del state, checkpoint
    model.requires_grad_(False)
    return model.float().eval().to(DEVICE)


def inference_one_im(model, im_path, args):
    options = dict(memory_fix=args.memory_fix,
                   tile_size=getattr(args, 'tile_size', None),
                   tile_overlap=getattr(args, 'tile_overlap', 64),
                   precision=getattr(args, 'precision', 'auto'))
    if args.task == 'end2end':
        result = dewarping(model, im_path, **options)
        result = deshadowing(model, result[3], **options)
        return appearance(model, result[3], **options)
    functions = dict(dewarping=dewarping, deshadowing=deshadowing,
                     appearance=appearance, deblurring=deblurring,
                     binarization=binarization)
    return functions[args.task](model, im_path, **options)


def _write_image(path, image, binary_bits=None, jpeg_quality=95):
    array = np.asarray(image)
    if binary_bits is not None:
        if array.ndim != 2 or not np.all((array == 0) | (array == 255)):
            raise ValueError('Binarization output must contain only 0 and 255.')
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(array.astype(np.uint8))
    if binary_bits == 1:
        # Explicitly disable dithering. The model's binary pixels are packed
        # into one bit each without adding dots, gray pixels or new thresholds.
        pil_image = pil_image.convert('1', dither=0)
    suffix = Path(path).suffix.lower()
    if suffix == '.png':
        pil_image.save(str(path), format='PNG', optimize=True, compress_level=9)
    elif suffix in ('.tif', '.tiff'):
        compression = 'group4' if binary_bits == 1 else 'tiff_adobe_deflate'
        pil_image.save(str(path), format='TIFF', compression=compression)
    elif suffix in ('.jpg', '.jpeg'):
        if binary_bits is not None:
            raise ValueError('Use PNG or TIFF for binary images.')
        pil_image.save(str(path), format='JPEG', quality=jpeg_quality,
                       optimize=True, subsampling=0)
    else:
        raise ValueError('Unsupported output extension: {}'.format(suffix))


def _save_path(img_path, out_folder, task, output_format, output_stem=None):
    source = Path(img_path)
    suffix = {'png': '.png', 'tiff': '.tiff', 'jpg': '.jpg'}[output_format]
    label = 'binarized' if task == 'binarization' else task
    return Path(out_folder) / '{}_{}{}'.format(output_stem or source.stem, label, suffix)


def save_results(img_path, out_folder, task, save_dtsprompt, results,
                 output_format='png', binary_bits=1, jpeg_quality=95, output_stem=None):
    folder = Path(out_folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_path = _save_path(img_path, out_folder, task, output_format, output_stem)
    prompt1, prompt2, prompt3, output = results
    bits = binary_bits if task == 'binarization' else None
    _write_image(save_path, output, binary_bits=bits, jpeg_quality=jpeg_quality)
    if save_dtsprompt:
        for index, prompt in enumerate((prompt1, prompt2, prompt3), 1):
            # Prompts include gray values, so they are always ordinary PNGs.
            prompt_path = save_path.with_name('{}_prompt{}.png'.format(
                save_path.stem, index))
            _write_image(prompt_path, prompt)
    return save_path


def _output_stems(paths):
    # Different source formats can share a stem. Preserve every image when
    # e.g. page.jpg and page.png both become PNG; also handle name/tag clashes.
    counts = {}
    for path in paths:
        key = path.stem.casefold()
        counts[key] = counts.get(key, 0) + 1
    used = set()
    stems = []
    for path in paths:
        base = path.stem
        if counts[base.casefold()] > 1:
            base += '_' + path.suffix.lstrip('.').lower()
        candidate = base
        index = 2
        while candidate.casefold() in used:
            candidate = '{}_{}'.format(base, index)
            index += 1
        used.add(candidate.casefold())
        stems.append(candidate)
    return stems


def _format_size(size):
    value = float(size)
    for unit in ('B', 'KiB', 'MiB', 'GiB'):
        if value < 1024 or unit == 'GiB':
            return '{:.1f} {}'.format(value, unit)
        value /= 1024


def main(argv=None):
    global DEVICE
    args = get_args(argv)
    if args.device == 'auto':
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device(args.device)
    source = Path(args.im_path)
    if source.is_dir():
        paths = sorted(path for path in source.iterdir()
                       if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    elif source.is_file():
        paths = [source]
    else:
        raise FileNotFoundError('Input path does not exist: {}'.format(source))
    if not paths:
        raise ValueError('No supported images found: {}'.format(source))
    model = model_init(args)
    print('Device: {}; precision: {}; tile size: {}'.format(
        DEVICE, args.precision,
        _tile_settings(args.memory_fix, args.tile_size, args.tile_overlap)), flush=True)
    failures = 0
    total_bytes = 0
    started = time.perf_counter()
    for index, (path, output_stem) in enumerate(zip(paths, _output_stems(paths)), 1):
        print('[{}/{}] Processing: {}'.format(index, len(paths), path.name), flush=True)
        image_started = time.perf_counter()
        try:
            results = inference_one_im(model, str(path), args)
            saved_path = save_results(path, args.out_folder, args.task, args.save_dtsprompt,
                                      results, output_format=args.output_format,
                                      binary_bits=args.binary_bits,
                                      jpeg_quality=args.jpeg_quality, output_stem=output_stem)
            saved_bytes = saved_path.stat().st_size
            total_bytes += saved_bytes
            print('      Saved: {} | {} | {:.1f}s'.format(
                saved_path.name, _format_size(saved_bytes),
                time.perf_counter() - image_started), flush=True)
            del results
        except Exception as error:
            failures += 1
            print('Failed: {}: {}'.format(path.name, error), file=sys.stderr, flush=True)
        # Free cached allocations between pages, not between every successful tile.
        gc.collect()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    print('\nCompleted: {}/{} images; {} failed; {}; {:.1f}s.'.format(
        len(paths) - failures, len(paths), failures, _format_size(total_bytes),
        time.perf_counter() - started), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())

