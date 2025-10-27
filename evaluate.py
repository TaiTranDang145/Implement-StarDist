import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models import StarDist
from my_datasets import DSB2018Datasets
from post_processing import postprocess_batch, calculate_iou
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import os
import json

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate StarDist model')
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--n-rays', type=int, default=32)
    parser.add_argument('--base-filters', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--prob-thresh', type=float, default=0.5)
    parser.add_argument('--nms-thresh', type=float, default=0.4)
    parser.add_argument('--iou-thresholds', nargs='+', type=float,
                        default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                        help='IoU thresholds for AP calculation')
    parser.add_argument('--save-dir', type=str, default='evaluation_results')
    return parser.parse_args()


def load_model(checkpoint_path, device, n_channels=3, n_rays=32, base_filters=64):
    """Load trained model"""
    model = StarDist(
        n_channels=n_channels,
        n_rays=n_rays,
        base_filter=base_filters
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def extract_instances_from_labeled_mask(labeled_mask):
    """
    Trích xuất các instance riêng lẻ từ labeled mask

    Args:
        labeled_mask: array shape (H, W) với các ID khác nhau cho mỗi instance

    Returns:
        instances: List of binary masks (H, W)
    """
    unique_ids = np.unique(labeled_mask)
    unique_ids = unique_ids[unique_ids > 0]  # Loại bỏ background (0)

    instances = []
    for instance_id in unique_ids:
        mask = (labeled_mask == instance_id).astype(np.uint8)
        instances.append(mask)

    return instances


def match_instances(pred_instances, gt_instances, iou_threshold=0.5):
    """
    Matching predictions với ground truth sử dụng Hungarian algorithm

    Args:
        pred_instances: List of predicted binary masks
        gt_instances: List of ground truth binary masks
        iou_threshold: IoU threshold để coi là match

    Returns:
        tp: Số lượng True Positives
        fp: Số lượng False Positives
        fn: Số lượng False Negatives
        matched_ious: List of IoU values cho matched pairs
    """
    if len(pred_instances) == 0 and len(gt_instances) == 0:
        return 0, 0, 0, []

    if len(pred_instances) == 0:
        return 0, 0, len(gt_instances), []

    if len(gt_instances) == 0:
        return 0, len(pred_instances), 0, []

    # Tính IoU matrix giữa tất cả predictions và ground truths
    iou_matrix = np.zeros((len(pred_instances), len(gt_instances)))

    for i, pred_mask in enumerate(pred_instances):
        for j, gt_mask in enumerate(gt_instances):
            iou_matrix[i, j] = calculate_iou(pred_mask, gt_mask)

    # Hungarian algorithm để tìm optimal matching
    pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)

    # Đếm TP, FP, FN
    matched_ious = []
    tp = 0

    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        iou = iou_matrix[pred_idx, gt_idx]
        if iou >= iou_threshold:
            tp += 1
            matched_ious.append(iou)

    fp = len(pred_instances) - tp
    fn = len(gt_instances) - tp

    return tp, fp, fn, matched_ious


def calculate_ap_at_iou(pred_instances_list, gt_instances_list, scores_list, iou_threshold=0.5):
    """
    Tính Average Precision tại một IoU threshold cụ thể

    Args:
        pred_instances_list: List of lists of predicted masks (per image)
        gt_instances_list: List of lists of ground truth masks (per image)
        scores_list: List of lists of confidence scores (per image)
        iou_threshold: IoU threshold

    Returns:
        ap: Average Precision
        precision: array of precision values
        recall: array of recall values
    """
    # Flatten tất cả predictions và scores từ tất cả các ảnh
    all_predictions = []
    all_gt_matched = []

    for pred_instances, gt_instances, scores in zip(pred_instances_list, gt_instances_list, scores_list):
        if len(pred_instances) == 0:
            continue

        # Tính IoU giữa predictions và ground truths
        for pred_idx, (pred_mask, score) in enumerate(zip(pred_instances, scores)):
            best_iou = 0
            matched = False

            for gt_mask in gt_instances:
                iou = calculate_iou(pred_mask, gt_mask)
                if iou > best_iou:
                    best_iou = iou
                    if iou >= iou_threshold:
                        matched = True

            all_predictions.append({
                'score': score,
                'matched': matched,
                'iou': best_iou
            })

    if len(all_predictions) == 0:
        return 0.0, np.array([]), np.array([])

    # Sort by confidence score (descending)
    all_predictions.sort(key=lambda x: x['score'], reverse=True)

    # Tính total ground truths
    total_gt = sum(len(gt) for gt in gt_instances_list)

    if total_gt == 0:
        return 0.0, np.array([]), np.array([])

    # Tính precision và recall tại mỗi threshold
    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []

    for pred in all_predictions:
        if pred['matched']:
            tp_cumsum += 1
        else:
            fp_cumsum += 1

        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / total_gt

        precisions.append(precision)
        recalls.append(recall)

    precisions = np.array(precisions)
    recalls = np.array(recalls)

    # Tính AP sử dụng 11-point interpolation hoặc all-point
    # Sử dụng all-point interpolation (COCO style)
    # Add sentinel values at the end
    precisions = np.concatenate(([0], precisions, [0]))
    recalls = np.concatenate(([0], recalls, [1]))

    # Compute the precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Tính AP
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])

    return ap, precisions[1:-1], recalls[1:-1]


def evaluate_model(model, dataloader, device, args):
    """
    Evaluate model trên toàn bộ dataset
    """
    all_pred_instances = []
    all_gt_instances = []
    all_scores = []

    # Collect predictions và ground truths
    with torch.no_grad():
        for images, prob_gt, rays_gt in tqdm(dataloader, desc="Inference"):
            images = images.to(device)

            # Forward
            pred_probs, pred_dists = model(images)

            # Post-processing
            pred_masks, centers, scores = postprocess_batch(
                pred_probs, pred_dists,
                prob_thresh=args.prob_thresh,
                nms_thresh=args.nms_thresh
            )

            # Extract instances from predictions và ground truth
            batch_size = images.shape[0]
            for i in range(batch_size):
                # Predicted instances
                pred_instances = extract_instances_from_labeled_mask(pred_masks[i])
                all_pred_instances.append(pred_instances)
                all_scores.append(scores[i])

                # Ground truth instances từ prob_gt
                # prob_gt shape: (1, H, W), cần chuyển thành labeled mask
                gt_prob = prob_gt[i, 0].cpu().numpy()

                # Label connected components
                from scipy.ndimage import label
                gt_labeled, _ = label(gt_prob > 0.5)
                gt_instances = extract_instances_from_labeled_mask(gt_labeled)
                all_gt_instances.append(gt_instances)


    # Calculate metrics
    results = {}

    aps = []

    for iou_thresh in args.iou_thresholds:
        ap, precisions, recalls = calculate_ap_at_iou(
            all_pred_instances, all_gt_instances, all_scores,
            iou_threshold=iou_thresh
        )
        aps.append(ap)
        results[f'AP@{iou_thresh:.2f}'] = ap
        print(f"  AP @ IoU {iou_thresh:.2f}: {ap:.4f}")

    # Calculate mAP
    mAP = np.mean(aps)
    results['mAP'] = mAP

    print(f"\n{'=' * 80}")
    print(f"Final Results:")
    print(f"{'=' * 80}")
    print(f"  mAP (IoU 0.5:0.95): {mAP:.4f}")
    print(f"  AP50 (IoU 0.5):     {results['AP@0.50']:.4f}")
    print(f"  AP75 (IoU 0.75):    {results['AP@0.75']:.4f}")
    print(f"{'=' * 80}\n")

    # Calculate detection metrics at IoU=0.5
    print(f"Detection Metrics (IoU threshold = 0.5):")
    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious = []

    for pred_instances, gt_instances in zip(all_pred_instances, all_gt_instances):
        tp, fp, fn, matched_ious = match_instances(pred_instances, gt_instances, iou_threshold=0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(matched_ious)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  True Positives:  {total_tp}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    if len(all_ious) > 0:
        print(f"  Mean IoU (matched): {np.mean(all_ious):.4f}")

    results['precision@0.5'] = precision
    results['recall@0.5'] = recall
    results['f1@0.5'] = f1
    results['TP'] = total_tp
    results['FP'] = total_fp
    results['FN'] = total_fn

    return results


def plot_ap_curve(iou_thresholds, aps, save_path):
    """Plot AP vs IoU threshold curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(iou_thresholds, aps, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('IoU Threshold', fontsize=12)
    plt.ylabel('Average Precision (AP)', fontsize=12)
    plt.title('AP vs IoU Threshold', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim([0.45, 1.0])
    plt.ylim([0, 1.0])

    # Add mAP line
    mAP = np.mean(aps)
    plt.axhline(y=mAP, color='r', linestyle='--', label=f'mAP = {mAP:.4f}')
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved AP curve to {save_path}")


def save_results(results, args):
    """Save evaluation results"""
    os.makedirs(args.save_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(args.save_dir, 'evaluation_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✓ Saved results to {json_path}")

    # Plot AP curve
    aps = [results[f'AP@{t:.2f}'] for t in args.iou_thresholds]
    plot_path = os.path.join(args.save_dir, 'ap_curve.png')
    plot_ap_curve(args.iou_thresholds, aps, plot_path)

    # Save text report
    report_path = os.path.join(args.save_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("StarDist Model Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset Split: {args.split}\n")
        f.write(f"Probability Threshold: {args.prob_thresh}\n")
        f.write(f"NMS Threshold: {args.nms_thresh}\n\n")
        f.write("=" * 80 + "\n")
        f.write("Average Precision Metrics:\n")
        f.write("=" * 80 + "\n")
        f.write(f"mAP (IoU 0.5:0.95): {results['mAP']:.4f}\n")
        f.write(f"AP50 (IoU 0.5):     {results['AP@0.50']:.4f}\n")
        f.write(f"AP75 (IoU 0.75):    {results['AP@0.75']:.4f}\n\n")

        f.write("AP at different IoU thresholds:\n")
        for iou_thresh in args.iou_thresholds:
            f.write(f"  AP @ {iou_thresh:.2f}: {results[f'AP@{iou_thresh:.2f}']:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Detection Metrics (IoU = 0.5):\n")
        f.write("=" * 80 + "\n")
        f.write(f"True Positives:  {results['TP']}\n")
        f.write(f"False Positives: {results['FP']}\n")
        f.write(f"False Negatives: {results['FN']}\n")
        f.write(f"Precision: {results['precision@0.5']:.4f}\n")
        f.write(f"Recall:    {results['recall@0.5']:.4f}\n")
        f.write(f"F1-Score:  {results['f1@0.5']:.4f}\n")

    print(f"✓ Saved report to {report_path}")


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(
        args.checkpoint, device,
        n_channels=3,
        n_rays=args.n_rays,
        base_filters=args.base_filters
    )

    # Load dataset
    dataset = DSB2018Datasets(
        root=args.root,
        img_size=(args.img_size, args.img_size),
        split=args.split,
        n_rays=args.n_rays,
        augment=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    # Evaluate
    results = evaluate_model(model, dataloader, device, args)

    # Save results
    save_results(results, args)


if __name__ == '__main__':
    main()