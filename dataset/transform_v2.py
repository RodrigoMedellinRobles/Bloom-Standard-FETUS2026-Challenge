import json
import numpy as np
import random
import torch
import torch.nn.functional as F
import cv2
from PIL import ImageFilter
from scipy import ndimage
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.color import rgb2gray


# Preprocessing constants
CLIP_P_LOW, CLIP_P_HIGH = 1, 99
CLAHE_CLIP_LIMIT, CLAHE_TILE = 2.0, 8

VIEW_STATS = {}


def load_view_stats(stats_json_path: str):
    global VIEW_STATS
    with open(stats_json_path, "r") as f:
        raw = json.load(f)
    VIEW_STATS = {int(k): (float(v["mean"]), float(v["std"])) for k, v in raw.items()}
    print(f"[transform_v2] Loaded view stats from {stats_json_path}")
    for v, (m, s) in sorted(VIEW_STATS.items()):
        print(f"  View {v}: mean={m:.6f}  std={s:.6f}")


def preprocess_no_zscore(raw_image: np.ndarray) -> np.ndarray:
    """rgb2gray -> percentile clip -> CLAHE. Returns float32 in [0,1]."""
    gray = rgb2gray(raw_image).astype(np.float32)
    lo, hi = np.percentile(gray, CLIP_P_LOW), np.percentile(gray, CLIP_P_HIGH)
    if hi - lo < 1e-6:
        return np.zeros_like(gray, dtype=np.float32)
    clip = np.clip((gray - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                             tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    return clahe.apply((clip * 255).astype(np.uint8)).astype(np.float32) / 255.0


def zscore_normalize(img: np.ndarray, view_1based: int) -> np.ndarray:
    m, s = VIEW_STATS.get(view_1based, (0.0, 1.0))
    return ((img - m) / max(s, 1e-6)).astype(np.float32)


def zscore_normalize_tensor(t: torch.Tensor, view_1based: int) -> torch.Tensor:
    m, s = VIEW_STATS.get(view_1based, (0.0, 1.0))
    return (t - m) / max(s, 1e-6)


# --- Geometric augmentations ---

def random_rot_flip(img, mask=None):
    k = np.random.randint(0, 4)
    axis = np.random.randint(0, 2)
    if mask is not None:
        img = np.rot90(img, k)
        mask = np.rot90(mask, k)
        img = np.flip(img, axis=axis).copy()
        mask = np.flip(mask, axis=axis).copy()
        return img, mask
    else:
        img = np.rot90(img, k)
        img = np.flip(img, axis=axis).copy()
        return img


def random_rotate(img, mask=None):
    angle = np.random.randint(-20, 20)
    if mask is not None:
        img = ndimage.rotate(img, angle, order=0, reshape=False)
        mask = ndimage.rotate(mask, angle, order=0, reshape=False)
        return img, mask
    else:
        img = ndimage.rotate(img, angle, order=0, reshape=False)
        return img


def random_zoom(img, mask=None, scale_min=0.85, scale_max=1.15):
    scale = np.random.uniform(scale_min, scale_max)
    h, w = img.shape
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    img_z = ndimage.zoom(img, (new_h / h, new_w / w), order=1)
    msk_z = ndimage.zoom(mask, (new_h / h, new_w / w), order=0) if mask is not None else None

    def _crop_or_pad(arr, th, tw):
        ch, cw = arr.shape
        if ch < th or cw < tw:
            pad_h, pad_w = max(0, th - ch), max(0, tw - cw)
            arr = np.pad(arr, ((pad_h // 2, pad_h - pad_h // 2),
                               (pad_w // 2, pad_w - pad_w // 2)),
                         mode='constant', constant_values=0)
            ch, cw = arr.shape
        sh, sw = (ch - th) // 2, (cw - tw) // 2
        return arr[sh:sh + th, sw:sw + tw]

    img_out = _crop_or_pad(img_z, h, w).astype(np.float32)
    if mask is not None:
        msk_out = _crop_or_pad(msk_z, h, w).astype(mask.dtype)
        return img_out, msk_out
    return img_out


def elastic_deformation(img, mask=None, alpha=30.0, sigma=5.0):
    shape = img.shape
    dx = gaussian_filter(np.random.randn(*shape) * alpha, sigma)
    dy = gaussian_filter(np.random.randn(*shape) * alpha, sigma)

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = [np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))]

    img_out = map_coordinates(img, indices, order=1, mode='reflect').reshape(shape)
    if mask is not None:
        mask_out = map_coordinates(mask.astype(np.float32), indices,
                                   order=0, mode='reflect').reshape(shape)
        return img_out.astype(np.float32), mask_out.astype(np.int64)
    return img_out.astype(np.float32)


def random_crop_resize(img, mask=None, crop_range=(0.80, 1.0)):
    h, w = img.shape[:2]
    scale = np.random.uniform(*crop_range)
    ch, cw = int(h * scale), int(w * scale)

    y0 = np.random.randint(0, h - ch + 1) if ch < h else 0
    x0 = np.random.randint(0, w - cw + 1) if cw < w else 0

    img_crop = img[y0:y0 + ch, x0:x0 + cw]
    img_out = cv2.resize(img_crop, (w, h), interpolation=cv2.INTER_LINEAR)

    if mask is not None:
        mask_crop = mask[y0:y0 + ch, x0:x0 + cw]
        mask_out = cv2.resize(mask_crop.astype(np.float32), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(np.int64)
        return img_out.astype(np.float32), mask_out
    return img_out.astype(np.float32)


# --- Intensity augmentations (float32 [0,1]) ---

def speckle_noise(img, sigma=0.05):
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    return np.clip(img + img * noise, 0, 1)


def gamma_correction(img, gamma_range=(0.7, 1.5)):
    gamma = np.random.uniform(*gamma_range)
    return np.power(np.clip(img, 1e-8, 1.0), gamma).astype(np.float32)


def random_brightness(img, max_delta=0.15):
    delta = np.random.uniform(-max_delta, max_delta)
    return np.clip(img + delta, 0, 1).astype(np.float32)


def random_contrast(img, contrast_range=(0.7, 1.3)):
    factor = np.random.uniform(*contrast_range)
    mean = img.mean()
    return np.clip((img - mean) * factor + mean, 0, 1).astype(np.float32)


def random_brightness_contrast(img, brightness=0.2, contrast=0.2):
    alpha = 1.0 + np.random.uniform(-contrast, contrast)
    beta = np.random.uniform(-brightness, brightness)
    return np.clip(alpha * img + beta, 0.0, 1.0).astype(np.float32)


def random_gaussian_noise(img, std_max=0.05):
    std = np.random.uniform(0.0, std_max)
    noise = np.random.normal(0.0, std, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0).astype(np.float32)


def coarse_dropout(img, max_holes=6, max_size=30, fill_value=0.0):
    h, w = img.shape[:2]
    img_out = img.copy()
    for _ in range(np.random.randint(1, max_holes + 1)):
        sh = np.random.randint(5, max_size)
        sw = np.random.randint(5, max_size)
        y = np.random.randint(0, max(1, h - sh))
        x = np.random.randint(0, max(1, w - sw))
        img_out[y:y + sh, x:x + sw] = fill_value
    return img_out


# --- PIL utilities (UniMatch strong augmentations) ---

def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4,
                       ratio_1=0.3, ratio_2=1 / 0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask
    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)
        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break
    mask[y:y + cutmix_h, x:x + cutmix_w] = 1
    return mask


# --- Composed pipelines ---

def augment_labeled_strong(img, mask):
    """Strong aug for labeled data. Applied before resize + z-score."""
    if random.random() < 0.5:
        img, mask = random_rot_flip(img, mask)
    if random.random() < 0.5:
        img, mask = random_rotate(img, mask)
    if random.random() < 0.5:
        img, mask = random_zoom(img, mask, scale_min=0.85, scale_max=1.15)
    if random.random() < 0.3:
        img, mask = elastic_deformation(img, mask, alpha=30.0, sigma=5.0)

    if random.random() < 0.5:
        img = gamma_correction(img, gamma_range=(0.7, 1.5))
    if random.random() < 0.4:
        img = random_brightness(img, max_delta=0.12)
    if random.random() < 0.4:
        img = random_contrast(img, contrast_range=(0.75, 1.25))
    if random.random() < 0.3:
        img = speckle_noise(img, sigma=0.04)

    return img, mask


def augment_unlabeled_weak(img):
    if random.random() < 0.5:
        img = random_rot_flip(img)
    elif random.random() < 0.5:
        img = random_rotate(img)
    return img


def augment_unlabeled_strong_np(img):
    """Strong numpy aug for unlabeled, applied before PIL ColorJitter."""
    if random.random() < 0.4:
        img = gamma_correction(img, gamma_range=(0.75, 1.4))
    if random.random() < 0.3:
        img = speckle_noise(img, sigma=0.03)
    if random.random() < 0.25:
        img = coarse_dropout(img, max_holes=4, max_size=20)
    return img


# --- MixUp (same-view constraint) ---

def mixup_data_same_view(images, labels, views, alpha=0.3):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)
    else:
        lam = 1.0

    B = images.size(0)
    device = images.device
    perm = torch.arange(B, device=device)

    for v in views.unique():
        idx = (views == v).nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n < 2:
            continue
        shuffled = idx[torch.randperm(n, device=device)]
        perm[idx] = shuffled

    mixed_images = lam * images + (1 - lam) * images[perm]
    mixed_labels = lam * labels + (1 - lam) * labels[perm]
    return mixed_images, mixed_labels, lam


# --- Loss functions ---

def masked_focal_bce(logits, targets, mask, pos_weight=None, gamma=2.0):
    p = torch.sigmoid(logits)
    if pos_weight is not None:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=pos_weight)
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    p_t = targets * p + (1 - targets) * (1 - p)
    focal_weight = (1 - p_t) ** gamma
    loss = focal_weight * bce * mask
    return loss.sum() / (mask.sum() + 1e-6)


def masked_asl(logits, targets, mask, gamma_neg=4, gamma_pos=0, clip=0.05):
    """Asymmetric Loss with view-mask."""
    prob = torch.sigmoid(logits)
    if clip > 0:
        prob_neg = (prob - clip).clamp(min=0)
        prob = prob * targets + prob_neg * (1 - targets)

    los_pos = -targets * torch.log(prob.clamp(min=1e-8))
    los_neg = -(1 - targets) * torch.log((1 - prob).clamp(min=1e-8))

    if gamma_pos > 0:
        los_pos = los_pos * ((1 - prob) ** gamma_pos)
    if gamma_neg > 0:
        pt_neg = 1 - prob
        los_neg = los_neg * ((1 - pt_neg) ** gamma_neg)

    loss = los_pos + los_neg
    return (loss * mask).sum() / (mask.sum() + 1e-6)
