import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from scipy.ndimage import maximum_filter
from skimage.draw import polygon

import matplotlib.pyplot as plt

from models import StarDist
from my_datasets import DSB2018Datasets


# ==========================
# Arguments
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

    parser.add_argument("--prob-threshold", type=float, default=0.5)
    parser.add_argument("--min-size", type=int, default=20,
                        help="minimum instance area (pixels)")
    parser.add_argument("--peak-min-distance", type=int, default=2,
                        help="radius for local maxima detection in pixels")
    parser.add_argument("--nms-iou-thresh", type=float, default=0.3,
                        help="IoU threshold for mask-level NMS")

    parser.add_argument("--save-vis-dir", type=str,
                        default="vis_stardist_eval")
    parser.add_argument("--num-vis", type=int, default=8,
                        help="number of qualitative segmentation images to save")

    return parser.parse_args()


# ==========================
# StarDist post-processing
# ==========================
def find_peaks(prob_map, prob_thresh=0.5, min_distance=2):
    """
    Tìm local maxima trên probability map.
    prob_map: (H, W) numpy
    Return: list[(y, x)]
    """
    H, W = prob_map.shape
    mask = prob_map >= prob_thresh
    if not np.any(mask):
        return []

    size = 2 * min_distance + 1
    footprint = np.ones((size, size), dtype=bool)
    max_filt = maximum_filter(prob_map, footprint=footprint, mode="nearest")
    peaks = (prob_map == max_filt) & mask

    ys, xs = np.nonzero(peaks)
    coords = list(zip(ys.tolist(), xs.tolist()))
    return coords


def rays_to_polygon_mask(center_y, center_x, rays, H, W):
    """
    center_y, center_x: int
    rays: (n_rays,) distances in pixels
    Trả về mask bool (H, W) cho đa giác star-convex.
    """
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
    """
    masks: list[bool(H,W)]
    scores: list/array float
    Trả về index các mask được giữ lại sau NMS.
    """
    if len(masks) == 0:
        return []

    areas = np.array([m.sum() for m in masks], dtype=np.float32)
    # Lọc theo min_size
    keep_initial = [i for i, a in enumerate(areas) if a >= min_size]
    if len(keep_initial) == 0:
        return []

    masks = [masks[i] for i in keep_initial]
    scores = np.asarray(scores, dtype=np.float32)[keep_initial]
    areas = areas[keep_initial]

    order = np.argsort(scores)[::-1]  # sort theo score giảm dần
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
    StarDist-style:
      1) local maxima trên prob_map
      2) từ mỗi center + rays → đa giác star-convex
      3) mask-level NMS

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
        rays = dist_map[:, y, x]  # (n_rays,)
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
    """
    gt_mask, pred_mask: (H, W) int labels, 0=background
    """
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
    """
    Tính TP/FP theo thứ tự score để dùng cho AP.
    Trả về:
      - scores: list[float]
      - tp_flags: list[int]  (1 nếu TP, 0 nếu FP)
      - num_gt: int (số GT instances trong ảnh này)
    """
    gt_ids = [i for i in np.unique(gt_mask) if i != 0]
    pred_ids = [i for i in np.unique(pred_mask) if i != 0]

    num_gt = len(gt_ids)
    if num_gt == 0 or len(pred_ids) == 0:
        return [], [], num_gt

    # sort pred theo score giảm dần
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
    """
    all_scores: list[float] cho tất cả prediction (mọi ảnh)
    all_tp_flags: list[int] (1 nếu TP, 0 nếu FP)
    total_gt: tổng số GT instances
    """
    if total_gt == 0 or len(all_scores) == 0:
        return 0.0

    scores = np.asarray(all_scores, dtype=np.float32)
    tp_flags = np.asarray(all_tp_flags, dtype=np.int32)

    # sort theo score giảm dần
    order = np.argsort(-scores)
    tp = tp_flags[order]
    fp = 1 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / float(total_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # thêm điểm (0,1) và (1,0)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    # làm trơn precision (monotonic decreasing)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # tích phân theo từng đoạn recall thay đổi
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


# ==========================
# Visualization (segmentation sample)
# ==========================
def visualize_sample(save_path, image, gt_mask, pred_mask,
                     prob_gt=None, prob_pred=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if image.dtype != np.uint8:
        img_vis = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        img_vis = image

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0, 0].imshow(img_vis, cmap="gray")
    axs[0, 0].set_title("Image")
    axs[0, 0].axis("off")

    im1 = axs[0, 1].imshow(gt_mask, cmap="tab20")
    axs[0, 1].set_title(f"GT instances (n={gt_mask.max()})")
    axs[0, 1].axis("off")
    fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    im2 = axs[0, 2].imshow(pred_mask, cmap="tab20")
    axs[0, 2].set_title(f"Pred instances (n={pred_mask.max()})")
    axs[0, 2].axis("off")
    fig.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)

    if prob_gt is not None:
        im3 = axs[1, 0].imshow(prob_gt, cmap="hot", vmin=0, vmax=1)
        axs[1, 0].set_title("GT prob_map")
        axs[1, 0].axis("off")
        fig.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)
    else:
        axs[1, 0].axis("off")

    if prob_pred is not None:
        im4 = axs[1, 1].imshow(prob_pred, cmap="hot", vmin=0, vmax=1)
        axs[1, 1].set_title("Pred prob_map")
        axs[1, 1].axis("off")
        fig.colorbar(im4, ax=axs[1, 1], fraction=0.046, pad=0.04)
    else:
        axs[1, 1].axis("off")

    gt_binary = gt_mask > 0
    pred_binary = pred_mask > 0

    overlay = img_vis.copy().astype(np.float32)
    # green = GT, red = Pred
    overlay[gt_binary] = overlay[gt_binary] * 0.5 + np.array([0, 255, 0]) * 0.5
    overlay[pred_binary] = overlay[pred_binary] * 0.5 + np.array([255, 0, 0]) * 0.5

    axs[1, 2].imshow(overlay.astype(np.uint8))
    axs[1, 2].set_title("Overlay (green=GT, red=Pred)")
    axs[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ==========================
# Main
# ==========================
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & loader (val split)
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

    # Model
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
    model.eval()

    # Accumulators
    total_tp = total_fp = total_fn = 0
    total_iou = total_dice = 0.0
    num_images = 0

    # IoU thresholds cho AP@[0.5:0.95]
    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # 0.50, 0.55, ..., 0.95
    all_scores = {t: [] for t in iou_thresholds}
    all_tp_flags = {t: [] for t in iou_thresholds}
    total_gt_for_ap = 0

    os.makedirs(args.save_vis_dir, exist_ok=True)
    vis_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if len(batch) == 4:
                images, prob_gt, dist_gt, labeled_mask = batch
            else:
                images, prob_gt, dist_gt = batch
                labeled_mask = None

            images = images.to(device)
            prob_gt = prob_gt.to(device)

            pred_prob_logits, pred_dists = model(images)
            pred_probs = torch.sigmoid(pred_prob_logits).cpu().numpy()  # (B,1,H,W)
            pred_dists_np = pred_dists.cpu().numpy()  # (B,n_rays,H,W)
            prob_gt_np = prob_gt.cpu().numpy()  # (B,1,H,W)

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
                    # fallback nếu không có instance GT
                    from scipy.ndimage import label as cc_label
                    binary = prob_gt_i > 0.5
                    gt_mask_i, _ = cc_label(binary)

                pred_mask_i, inst_scores = stardist_postprocess(
                    prob_pred_i,
                    dist_pred_i,
                    prob_thresh=args.prob_threshold,
                    peak_min_distance=args.peak_min_distance,
                    nms_iou_thresh=args.nms_iou_thresh,
                    min_size=args.min_size,
                )

                # Instance metrics tại IoU=0.5
                inst_metrics = compute_instance_metrics(
                    gt_mask_i, pred_mask_i, iou_thresh=0.5
                )
                total_tp += inst_metrics["tp"]
                total_fp += inst_metrics["fp"]
                total_fn += inst_metrics["fn"]

                # Pixel metrics
                gt_bin = gt_mask_i > 0
                pred_bin = pred_mask_i > 0
                iou, dice = compute_pixel_metrics(gt_bin, pred_bin)
                total_iou += iou
                total_dice += dice
                num_images += 1

                # AP data cho nhiều IoU thresholds
                num_gt_i = max(int(gt_mask_i.max()), 0)
                total_gt_for_ap += num_gt_i
                for t in iou_thresholds:
                    scores_img, tp_flags_img, _ = collect_ap_data_for_image(
                        gt_mask_i, pred_mask_i, inst_scores, iou_thresh=t
                    )
                    all_scores[t].extend(scores_img)
                    all_tp_flags[t].extend(tp_flags_img)

                # Qualitative visualization (segmentation)
                if vis_count < args.num_vis:
                    img_np = images[i].cpu().permute(1, 2, 0).numpy()
                    save_path = os.path.join(
                        args.save_vis_dir,
                        f"val_{batch_idx * B + i:04d}.png",
                    )
                    visualize_sample(
                        save_path,
                        img_np,
                        gt_mask_i,
                        pred_mask_i,
                        prob_gt=prob_gt_i,
                        prob_pred=prob_pred_i,
                    )
                    vis_count += 1

    # Global metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall /
          (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = total_iou / num_images if num_images > 0 else 0.0
    mean_dice = total_dice / num_images if num_images > 0 else 0.0

    # AP per IoU threshold & mAP
    ap_values = []
    for t in iou_thresholds:
        ap_t = compute_ap_from_scores(
            all_scores[t],
            all_tp_flags[t],
            total_gt_for_ap,
        )
        ap_values.append(ap_t)

    mAP = float(np.mean(ap_values)) if len(ap_values) > 0 else 0.0

    # Print results
    print("==== StarDist Evaluation (val set) ====")
    print(f"Instance-level (IoU=0.5): TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print("Pixel-level:")
    print(f"  mean IoU:  {mean_iou:.4f}")
    print(f"  mean Dice: {mean_dice:.4f}")
    print("AP per IoU threshold:")
    for t, ap in zip(iou_thresholds, ap_values):
        print(f"  AP@{t:.2f}: {ap:.4f}")
    print(f"mAP@[0.50:0.95]: {mAP:.4f}")
    print(f"Saved {vis_count} qualitative segmentation images to: {args.save_vis_dir}")

    # Plot AP vs IoU
    plt.figure(figsize=(10, 5))
    plt.plot(iou_thresholds, ap_values, marker="o", linestyle="-", label="AP")
    plt.xlabel("IoU Threshold")
    plt.ylabel("Average Precision (AP)")
    plt.title("AP vs IoU Threshold")
    plt.ylim(0.0, 1.0)
    plt.xlim(0.45, 0.95)
    plt.axhline(y=mAP, color="red", linestyle="--",
                label=f"mAP = {mAP:.4f}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plot_path = os.path.join(args.save_vis_dir, "ap_vs_iou.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved AP vs IoU plot to: {plot_path}")


if __name__ == "__main__":
    main()
