import argparse
import os
import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader

from scipy.ndimage import maximum_filter
from skimage.draw import polygon

from models import StarDist
from my_datasets import DSB2018Datasets
import warnings

# Ẩn mọi DeprecationWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================
# Args
# ==========================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--base-filters", type=int, default=32)
    parser.add_argument("--shared-channels", type=int, default=128)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


# ==========================
# StarDist post-processing helpers
# ==========================
def find_peaks(prob_map, prob_thresh=0.5, min_distance=2):
    H, W = prob_map.shape
    mask = prob_map >= prob_thresh
    if not np.any(mask):
        return []

    size = 2 * min_distance + 1
    footprint = np.ones((size, size), dtype=bool)
    max_filt = maximum_filter(prob_map, footprint=footprint, mode="nearest")
    peaks = (prob_map == max_filt) & mask

    ys, xs = np.nonzero(peaks)
    return list(zip(ys.tolist(), xs.tolist()))


def rays_to_polygon_mask(center_y, center_x, rays, H, W):
    n_rays = rays.shape[0]
    max_radius = max(H, W)
    rays = np.clip(rays, 0.0, float(max_radius))

    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    ys = center_y + rays * np.sin(angles)
    xs = center_x + rays * np.cos(angles)

    ys = np.clip(ys, 0, H - 1)
    xs = np.clip(xs, 0, W - 1)

    rr, cc = polygon(ys, xs, (H, W))
    mask = np.zeros((H, W), dtype=bool)
    mask[rr, cc] = True
    return mask


def nms_on_masks(masks, scores, iou_thresh=0.3, min_size=0):
    if len(masks) == 0:
        return []

    areas = np.array([m.sum() for m in masks], dtype=np.float32)
    keep_initial = [i for i, a in enumerate(areas) if a >= min_size]
    if len(keep_initial) == 0:
        return []

    masks = [masks[i] for i in keep_initial]
    scores = np.asarray(scores, dtype=np.float32)[keep_initial]
    areas = areas[keep_initial]

    order = np.argsort(scores)[::-1]
    keep = []
    used = np.zeros(len(masks), dtype=bool)

    for idx in order:
        if used[idx]:
            continue
        keep.append(idx)
        used[idx] = True
        m_i = masks[idx]
        area_i = areas[idx]

        for j in order:
            if used[j]:
                continue
            m_j = masks[j]
            inter = np.logical_and(m_i, m_j).sum()
            if inter == 0:
                continue
            union = area_i + areas[j] - inter
            iou = inter / union
            if iou > iou_thresh:
                used[j] = True

    kept_global = [keep_initial[i] for i in keep]
    return kept_global


def stardist_postprocess(prob_map, dist_map, prob_thresh=0.5,
                         peak_min_distance=2, nms_iou_thresh=0.3,
                         min_size=0):
    """
    Trả về:
      - labeled_mask: (H, W) int32
      - inst_scores: dict {instance_id: score}
    """
    H, W = prob_map.shape

    centers = find_peaks(prob_map, prob_thresh=prob_thresh,
                         min_distance=peak_min_distance)
    if len(centers) == 0:
        return np.zeros((H, W), dtype=np.int32), {}

    masks = []
    scores = []

    for (y, x) in centers:
        rays = dist_map[:, y, x]
        if np.all(rays <= 0):
            continue
        mask = rays_to_polygon_mask(y, x, rays, H, W)
        if mask.sum() == 0:
            continue
        masks.append(mask)
        scores.append(prob_map[y, x])

    keep_idxs = nms_on_masks(masks, scores,
                             iou_thresh=nms_iou_thresh,
                             min_size=min_size)

    labeled = np.zeros((H, W), dtype=np.int32)
    inst_scores = {}
    next_id = 1
    for ki in keep_idxs:
        labeled[masks[ki]] = next_id
        inst_scores[next_id] = float(scores[ki])
        next_id += 1

    return labeled, inst_scores


# ==========================
# Metrics
# ==========================
def compute_instance_metrics(gt_mask, pred_mask, iou_thresh=0.5):
    gt_ids = [i for i in np.unique(gt_mask) if i != 0]
    pred_ids = [i for i in np.unique(pred_mask) if i != 0]

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return dict(tp=0, fp=0, fn=0,
                    precision=1.0, recall=1.0, f1=1.0)
    if len(gt_ids) == 0:
        return dict(tp=0, fp=len(pred_ids), fn=0,
                    precision=0.0, recall=0.0, f1=0.0)
    if len(pred_ids) == 0:
        return dict(tp=0, fp=0, fn=len(gt_ids),
                    precision=0.0, recall=0.0, f1=0.0)

    iou_mat = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    for i, gid in enumerate(gt_ids):
        g = gt_mask == gid
        g_area = g.sum()
        for j, pid in enumerate(pred_ids):
            p = pred_mask == pid
            inter = np.logical_and(g, p).sum()
            if inter == 0:
                continue
            union = g_area + p.sum() - inter
            iou_mat[i, j] = inter / union

    matched_gt = set()
    matched_pred = set()
    tp = 0

    while True:
        if iou_mat.size == 0:
            break
        idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        max_iou = iou_mat[idx]
        if max_iou < iou_thresh:
            break
        i, j = idx
        tp += 1
        matched_gt.add(gt_ids[i])
        matched_pred.add(pred_ids[j])
        iou_mat[i, :] = 0
        iou_mat[:, j] = 0

    fp = len(pred_ids) - len(matched_pred)
    fn = len(gt_ids) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall /
          (precision + recall)) if (precision + recall) > 0 else 0.0

    return dict(tp=tp, fp=fp, fn=fn,
                precision=precision, recall=recall, f1=f1)


def compute_pixel_metrics(gt_binary, pred_binary):
    inter = np.logical_and(gt_binary, pred_binary).sum()
    union = np.logical_or(gt_binary, pred_binary).sum()
    iou = inter / union if union > 0 else 1.0
    dice = (2 * inter / (gt_binary.sum() + pred_binary.sum())
            if (gt_binary.sum() + pred_binary.sum()) > 0 else 1.0)
    return iou, dice


def collect_ap_data_for_image(gt_mask, pred_mask, inst_scores, iou_thresh=0.5):
    gt_ids = [i for i in np.unique(gt_mask) if i != 0]
    pred_ids = [i for i in np.unique(pred_mask) if i != 0]

    num_gt = len(gt_ids)
    if num_gt == 0 or len(pred_ids) == 0:
        return [], [], num_gt

    pred_ids_sorted = sorted(
        pred_ids,
        key=lambda pid: inst_scores.get(pid, 0.0),
        reverse=True,
    )

    gt_used = set()
    scores = []
    tp_flags = []

    for pid in pred_ids_sorted:
        p_mask = (pred_mask == pid)
        best_iou = 0.0
        best_gid = None

        for gid in gt_ids:
            if gid in gt_used:
                continue
            g_mask = (gt_mask == gid)

            inter = np.logical_and(p_mask, g_mask).sum()
            if inter == 0:
                continue
            union = p_mask.sum() + g_mask.sum() - inter
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_gid = gid

        scores.append(inst_scores.get(pid, 0.0))
        if best_iou >= iou_thresh and best_gid is not None:
            tp_flags.append(1)
            gt_used.add(best_gid)
        else:
            tp_flags.append(0)

    return scores, tp_flags, num_gt


def compute_ap_from_scores(all_scores, all_tp_flags, total_gt):
    if total_gt == 0 or len(all_scores) == 0:
        return 0.0

    scores = np.asarray(all_scores, dtype=np.float32)
    tp_flags = np.asarray(all_tp_flags, dtype=np.int32)

    order = np.argsort(-scores)
    tp = tp_flags[order]
    fp = 1 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / float(total_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


# ==========================
# Evaluate 1 cấu hình
# ==========================
def evaluate_config(model, val_loader, device,
                    prob_threshold, peak_min_distance,
                    nms_iou_thresh, min_size):
    model.eval()

    total_tp = total_fp = total_fn = 0
    total_iou = total_dice = 0.0
    num_images = 0

    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    all_scores = {t: [] for t in iou_thresholds}
    all_tp_flags = {t: [] for t in iou_thresholds}
    total_gt_for_ap = 0

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 4:
                images, prob_gt, dist_gt, labeled_mask = batch
            else:
                images, prob_gt, dist_gt = batch
                labeled_mask = None

            images = images.to(device)
            prob_gt = prob_gt.to(device)

            pred_prob_logits, pred_dists = model(images)
            pred_probs = torch.sigmoid(pred_prob_logits).cpu().numpy()
            pred_dists_np = pred_dists.cpu().numpy()
            prob_gt_np = prob_gt.cpu().numpy()

            if labeled_mask is not None:
                labeled_mask_np = labeled_mask.numpy()
            else:
                labeled_mask_np = None

            B = images.size(0)
            for i in range(B):
                prob_pred_i = pred_probs[i, 0]
                dist_pred_i = pred_dists_np[i]
                prob_gt_i = prob_gt_np[i, 0]

                if labeled_mask_np is not None:
                    gt_mask_i = labeled_mask_np[i]
                else:
                    from scipy.ndimage import label as cc_label
                    binary = prob_gt_i > 0.5
                    gt_mask_i, _ = cc_label(binary)

                pred_mask_i, inst_scores = stardist_postprocess(
                    prob_pred_i,
                    dist_pred_i,
                    prob_thresh=prob_threshold,
                    peak_min_distance=peak_min_distance,
                    nms_iou_thresh=nms_iou_thresh,
                    min_size=min_size,
                )

                # instance metrics @ IoU=0.5
                inst_metrics = compute_instance_metrics(
                    gt_mask_i, pred_mask_i, iou_thresh=0.5
                )
                total_tp += inst_metrics["tp"]
                total_fp += inst_metrics["fp"]
                total_fn += inst_metrics["fn"]

                # pixel metrics
                gt_bin = gt_mask_i > 0
                pred_bin = pred_mask_i > 0
                iou, dice = compute_pixel_metrics(gt_bin, pred_bin)
                total_iou += iou
                total_dice += dice
                num_images += 1

                # AP data
                num_gt_i = max(int(gt_mask_i.max()), 0)
                total_gt_for_ap += num_gt_i
                for t in iou_thresholds:
                    scores_img, tp_flags_img, _ = collect_ap_data_for_image(
                        gt_mask_i, pred_mask_i, inst_scores, iou_thresh=t
                    )
                    all_scores[t].extend(scores_img)
                    all_tp_flags[t].extend(tp_flags_img)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall /
          (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = total_iou / num_images if num_images > 0 else 0.0
    mean_dice = total_dice / num_images if num_images > 0 else 0.0

    ap_values = []
    for t in iou_thresholds:
        ap_t = compute_ap_from_scores(
            all_scores[t],
            all_tp_flags[t],
            total_gt_for_ap,
        )
        ap_values.append(ap_t)
    mAP = float(np.mean(ap_values)) if len(ap_values) > 0 else 0.0

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "mAP": mAP,
    }


# ==========================
# Main (grid search)
# ==========================
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # dataset & loader
    val_dataset = DSB2018Datasets(
        root=args.root,
        img_size=(args.img_size, args.img_size),
        split="val",
        n_rays=args.n_rays,
        augment=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )
    print(f"Val size: {len(val_dataset)} images")

    # model
    model = StarDist(
        n_channels=3,
        n_rays=args.n_rays,
        base_filters=args.base_filters,
        shared_channels=args.shared_channels,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    print("Loaded checkpoint.")

    # grid search ranges
    prob_threshold_list = [0.4, 0.5, 0.6]
    peak_min_distance_list = [1, 2, 3]
    nms_iou_thresh_list = [0.2, 0.3, 0.4]
    min_size_list = [10, 20, 40]

    combos = list(itertools.product(
        prob_threshold_list,
        peak_min_distance_list,
        nms_iou_thresh_list,
        min_size_list,
    ))
    print(f"Total configs: {len(combos)}")

    results = []

    for idx, (p_th, peak_md, nms_th, ms) in enumerate(combos, start=1):
        print(f"\n[{idx}/{len(combos)}] "
              f"prob_th={p_th}, peak_min_dist={peak_md}, "
              f"nms_iou={nms_th}, min_size={ms}")

        metrics = evaluate_config(
            model,
            val_loader,
            device,
            prob_threshold=p_th,
            peak_min_distance=peak_md,
            nms_iou_thresh=nms_th,
            min_size=ms,
        )

        print(f"  F1@0.5 = {metrics['f1']:.4f}, "
              f"mAP@[0.5:0.95] = {metrics['mAP']:.4f}, "
              f"Prec = {metrics['precision']:.4f}, "
              f"Rec = {metrics['recall']:.4f}")

        results.append({
            "prob_th": p_th,
            "peak_min_dist": peak_md,
            "nms_iou": nms_th,
            "min_size": ms,
            **metrics,
        })

    # sort kết quả theo mAP (giảm dần)
    results_sorted_map = sorted(results, key=lambda r: r["mAP"], reverse=True)
    best_map = results_sorted_map[0]

    # sort theo F1
    results_sorted_f1 = sorted(results, key=lambda r: r["f1"], reverse=True)
    best_f1 = results_sorted_f1[0]

    print("\n===== Top 5 configs by mAP =====")
    for r in results_sorted_map[:5]:
        print(
            f"mAP={r['mAP']:.4f}, F1={r['f1']:.4f} | "
            f"prob_th={r['prob_th']}, peak_min_dist={r['peak_min_dist']}, "
            f"nms_iou={r['nms_iou']}, min_size={r['min_size']}"
        )

    print("\n===== Top 5 configs by F1 =====")
    for r in results_sorted_f1[:5]:
        print(
            f"F1={r['f1']:.4f}, mAP={r['mAP']:.4f} | "
            f"prob_th={r['prob_th']}, peak_min_dist={r['peak_min_dist']}, "
            f"nms_iou={r['nms_iou']}, min_size={r['min_size']}"
        )

    print("\nBest (by mAP):", best_map)
    print("Best (by F1) :", best_f1)


if __name__ == "__main__":
    main()
