import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_dilation
from sklearn.metrics import f1_score, average_precision_score


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class DiceLoss(nn.Module):
    def __init__(self, n_classes, smooth=1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, pred, target, ignore=None):
        if target.dim() == 4:
            target = target.squeeze(1)

        target_oh = F.one_hot(target.long(), self.n_classes)
        target_oh = target_oh.permute(0, 3, 1, 2).float()

        if ignore is not None:
            if ignore.dim() == 3:
                ignore = ignore.unsqueeze(1)
            weight = 1.0 - ignore
            pred = pred * weight
            target_oh = target_oh * weight

        dice_total = 0.0
        valid_classes = 0

        for c in range(1, self.n_classes):
            p = pred[:, c]
            g = target_oh[:, c]
            inter = (p * g).sum()
            union = p.sum() + g.sum()
            if union > 0:
                dice_total += (2.0 * inter + self.smooth) / (union + self.smooth)
                valid_classes += 1

        if valid_classes == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return 1.0 - dice_total / valid_classes


class BoundaryLoss(nn.Module):
    """Boundary loss (Kervadec et al. 2019)."""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, pred_softmax, mask):
        B, C, H, W = pred_softmax.shape
        loss = torch.tensor(0.0, device=pred_softmax.device)

        for c in range(1, C):
            gt_c = (mask == c).float().cpu().numpy()
            dist_maps = []
            for b in range(B):
                if gt_c[b].sum() == 0:
                    dist_maps.append(np.zeros((H, W), dtype=np.float32))
                    continue
                dist = distance_transform_edt(gt_c[b]).astype(np.float32)
                dist = dist / (np.sqrt(H**2 + W**2) + 1e-6)
                dist_maps.append(dist)

            dist_t = torch.from_numpy(
                np.stack(dist_maps, axis=0)
            ).to(pred_softmax.device)

            loss = loss + (pred_softmax[:, c] * dist_t).mean()

        return loss / max(C - 1, 1)


def compute_boundary_gt(mask, num_classes=15):
    """Generate boundary GT from segmentation mask."""
    B, H, W = mask.shape
    boundary_map = torch.zeros(B, 1, H, W, device=mask.device)

    for c in range(1, num_classes):
        gt_c = (mask == c).float().unsqueeze(1)
        if gt_c.sum() == 0:
            continue
        dilated = F.max_pool2d(gt_c, kernel_size=3, stride=1, padding=1)
        eroded  = -F.max_pool2d(-gt_c, kernel_size=3, stride=1, padding=1)
        border  = dilated - eroded
        boundary_map = torch.clamp(boundary_map + border, 0, 1)

    return boundary_map


# --- View-based masking ---

def build_allowed_mat(device, allowed_dict, num_views, num_classes):
    mat = torch.zeros((num_views, num_classes), dtype=torch.bool, device=device)
    for v in range(num_views):
        if v in allowed_dict:
            for c in allowed_dict[v]:
                mat[v, c] = True
    return mat


def apply_view_mask_logits(logits, views, allowed_mat):
    B, C, H, W = logits.shape
    mask = allowed_mat[views]
    mask = mask.float().unsqueeze(-1).unsqueeze(-1)
    return logits * mask + (1 - mask) * (-1e9)


# --- Classification losses ---

def masked_bce_with_logits(logits, targets, mask, pos_weight=None):
    mask = mask.float()
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none', pos_weight=pos_weight
    )
    return (bce * mask).sum() / (mask.sum() + 1e-6)


def masked_mse(pred, target, mask):
    mask = mask.float()
    return ((pred - target) ** 2 * mask).sum() / (mask.sum() + 1e-6)


def masked_focal_bce_with_logits(logits, targets, mask, pos_weight=None,
                                  gamma=2.0, alpha=None, eps=1e-8):
    mask = mask.float()
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none', pos_weight=pos_weight
    )
    p = torch.sigmoid(logits)
    pt = p * targets + (1.0 - p) * (1.0 - targets)
    focal = (1.0 - pt).pow(gamma)

    if alpha is not None:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        focal = focal * alpha_t

    loss = bce * focal * mask
    return loss.sum() / (mask.sum() + eps)


# --- NSD ---

def nsd_binary(pred, gt, tol=2.0):
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    struct = np.ones((3, 3), dtype=bool)
    pred_surface = binary_dilation(pred, struct) & ~pred
    gt_surface   = binary_dilation(gt, struct) & ~gt

    n_pred = pred_surface.sum()
    n_gt   = gt_surface.sum()

    if n_pred == 0 and n_gt == 0:
        return 1.0

    dt_gt   = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)

    match_pred = (dt_gt[pred_surface] <= tol).sum() if n_pred > 0 else 0
    match_gt   = (dt_pred[gt_surface] <= tol).sum() if n_gt > 0 else 0

    return float(match_pred + match_gt) / float(n_pred + n_gt + 1e-8)


# --- Classification metrics ---

def masked_metrics_with_threshold_search(y_true, y_prob, views, cls_allowed, thresholds=None):
    N, K = y_true.shape

    valid = np.zeros((N, K), dtype=bool)
    for v, ks in cls_allowed.items():
        v_mask = views == v
        for k in ks:
            valid[v_mask, k] = True

    f1_at_05 = np.zeros(K)
    f1_at_best = np.zeros(K)
    best_thr = np.ones(K) * 0.5
    auprc = np.zeros(K)
    support = np.zeros(K, dtype=int)

    for k in range(K):
        mask_k = valid[:, k]
        if mask_k.sum() == 0:
            continue

        y_t = y_true[mask_k, k]
        y_p = y_prob[mask_k, k]
        support[k] = int(mask_k.sum())

        y_pred_05 = (y_p >= 0.5).astype(int)
        if y_t.sum() > 0:
            f1_at_05[k] = f1_score(y_t, y_pred_05, zero_division=0)
        else:
            f1_at_05[k] = float((y_pred_05 == 0).all())

        try:
            if y_t.sum() > 0 and y_t.sum() < len(y_t):
                auprc[k] = average_precision_score(y_t, y_p)
            else:
                auprc[k] = 0.0
        except Exception:
            auprc[k] = 0.0

        if thresholds is not None:
            best_thr[k] = thresholds[k]
            y_pred_best = (y_p >= thresholds[k]).astype(int)
            f1_at_best[k] = f1_score(y_t, y_pred_best, zero_division=0)
        else:
            best_f1_k = f1_at_05[k]
            best_thr_k = 0.5
            for thr in np.arange(0.1, 0.95, 0.05):
                y_pred_thr = (y_p >= thr).astype(int)
                f1_thr = f1_score(y_t, y_pred_thr, zero_division=0)
                if f1_thr > best_f1_k:
                    best_f1_k = f1_thr
                    best_thr_k = thr
            f1_at_best[k] = best_f1_k
            best_thr[k] = best_thr_k

    return {
        "per_class_f1@0.5": f1_at_05,
        "macro_f1@0.5": float(f1_at_05.mean()),
        "per_class_f1@best": f1_at_best,
        "macro_f1@best": float(f1_at_best.mean()),
        "per_class_best_thr": best_thr,
        "per_class_auprc": auprc,
        "support": support,
    }


def compute_pos_weight_from_loader(loader, allowed_cls_mat, num_classes, device):
    pos_counts = torch.zeros(num_classes)
    neg_counts = torch.zeros(num_classes)

    for batch in loader:
        label = batch[3] if len(batch) == 4 else batch[2]
        view = batch[1]

        view = view.long().view(-1)
        mask = allowed_cls_mat[view].cpu().float()

        for k in range(num_classes):
            valid = mask[:, k] > 0
            if valid.sum() == 0:
                continue
            pos_counts[k] += (label[valid, k] > 0).float().sum()
            neg_counts[k] += (label[valid, k] == 0).float().sum()

    pos_weight = (neg_counts + 1) / (pos_counts + 1)
    return pos_weight.to(device)


def build_same_view_perm(views_a, views_b):
    B = views_a.shape[0]
    device = views_a.device
    perm = torch.arange(B, device=device)

    for v in views_a.unique():
        idx_a = (views_a == v).nonzero(as_tuple=True)[0]
        idx_b = (views_b == v).nonzero(as_tuple=True)[0]

        n = min(len(idx_a), len(idx_b))
        if n > 0:
            shuffled = idx_b[torch.randperm(len(idx_b), device=device)]
            perm[idx_a[:n]] = shuffled[:n]

    return perm


def load_pretrained_flexible(model, checkpoint_path, strict=False, **kwargs):
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    msg = model.load_state_dict(state_dict, strict=strict)
    print(f"[load_pretrained] missing={len(msg.missing_keys)}, "
          f"unexpected={len(msg.unexpected_keys)}")
    if msg.missing_keys:
        print(f"  Missing (first 10): {msg.missing_keys[:10]}")
    if msg.unexpected_keys:
        print(f"  Unexpected (first 10): {msg.unexpected_keys[:10]}")

    return ckpt
