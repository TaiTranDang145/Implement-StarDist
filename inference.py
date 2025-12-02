import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from models import StarDist
from my_datasets import DSB2018Datasets
import argparse
import os
from scipy.ndimage import label
from skimage.measure import regionprops


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to model checkpoint')
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--n-rays', type=int, default=32)
    parser.add_argument('--base-filters', type=int, default=32)
    parser.add_argument('--shared-channels', type=int, default=128)
    parser.add_argument('--prob-thresh', type=float, default=0.5,
                        help='Probability threshold for detection')
    parser.add_argument('--nms-thresh', type=float, default=0.5,
                        help='NMS IoU threshold')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--output-dir', type=str, default='inference_results')
    return parser.parse_args()


def polygon_to_mask(center, distances, shape):
    """
    Convert star-convex polygon (center + radial distances) to binary mask
    
    Args:
        center: (y, x) tuple
        distances: array of n_rays distances
        shape: (H, W) image shape
    
    Returns:
        Binary mask (H, W)
    """
    n_rays = len(distances)
    points = []
    
    for k in range(n_rays):
        angle = 2 * np.pi * k / n_rays
        dist = distances[k]
        
        # Compute point on boundary
        x = center[1] + dist * np.sin(angle)
        y = center[0] + dist * np.cos(angle)
        
        # Clip to image bounds
        x = np.clip(x, 0, shape[1] - 1)
        y = np.clip(y, 0, shape[0] - 1)
        
        points.append([int(x), int(y)])
    
    # Create mask from polygon
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 1)
    
    return mask


def non_maximum_suppression(prob_map, dist_map, prob_thresh=0.5, nms_thresh=0.3):
    """
    Perform Non-Maximum Suppression to get final instance segmentation
    
    Args:
        prob_map: (H, W) probability map
        dist_map: (n_rays, H, W) distance map
        prob_thresh: threshold for probability
        nms_thresh: IoU threshold for NMS
    
    Returns:
        instance_mask: (H, W) with each cell having unique ID
    """
    H, W = prob_map.shape
    n_rays = dist_map.shape[0]
    
    # Find candidate points (prob > threshold)
    candidates = np.where(prob_map > prob_thresh)
    
    if len(candidates[0]) == 0:
        return np.zeros((H, W), dtype=np.int32)
    
    # Sort by probability (descending)
    probs = prob_map[candidates]
    sorted_indices = np.argsort(probs)[::-1]
    
    # NMS
    instance_mask = np.zeros((H, W), dtype=np.int32)
    instance_id = 1
    suppressed = set()
    
    for idx in sorted_indices:
        if idx in suppressed:
            continue
        
        y, x = candidates[0][idx], candidates[1][idx]
        
        # Get distances for this point
        distances = dist_map[:, y, x]
        
        # Create polygon mask
        polygon_mask = polygon_to_mask((y, x), distances, (H, W))
        
        # Check overlap with existing instances
        overlap = (polygon_mask > 0) & (instance_mask > 0)
        
        if np.any(overlap):
            # Calculate IoU with overlapping instances
            overlapping_ids = np.unique(instance_mask[overlap])
            
            max_iou = 0
            for other_id in overlapping_ids:
                if other_id == 0:
                    continue
                other_mask = (instance_mask == other_id)
                intersection = np.sum(polygon_mask & other_mask)
                union = np.sum(polygon_mask | other_mask)
                iou = intersection / (union + 1e-8)
                max_iou = max(max_iou, iou)
            
            # Suppress if IoU too high
            if max_iou > nms_thresh:
                suppressed.add(idx)
                continue
        
        # Add this instance
        instance_mask[polygon_mask > 0] = instance_id
        instance_id += 1
    
    return instance_mask


def predict_image(model, image, device, prob_thresh=0.5, nms_thresh=0.3):
    """
    Predict segmentation for a single image
    
    Args:
        model: trained StarDist model
        image: (C, H, W) tensor
        device: cuda or cpu
        prob_thresh: probability threshold
        nms_thresh: NMS threshold
    
    Returns:
        instance_mask: (H, W) numpy array with instance IDs
        prob_map: (H, W) probability map
    """
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension
        image_batch = image.unsqueeze(0).to(device)  # (1, C, H, W)
        
        # Predict
        pred_prob, pred_dist = model(image_batch)  # (1, 1, H, W), (1, n_rays, H, W)
        
        # Apply sigmoid to probability
        pred_prob = torch.sigmoid(pred_prob).squeeze().cpu().numpy()  # (H, W)
        pred_dist = pred_dist.squeeze().cpu().numpy()  # (n_rays, H, W)
    
    # Perform NMS
    instance_mask = non_maximum_suppression(pred_prob, pred_dist, prob_thresh, nms_thresh)
    
    return instance_mask, pred_prob


def visualize_prediction(image, pred_mask, gt_mask=None, save_path=None):
    """
    Visualize prediction results
    
    Args:
        image: (C, H, W) tensor
        pred_mask: (H, W) predicted instance mask
        gt_mask: (H, W) ground truth mask (optional)
        save_path: path to save figure
    """
    image_np = image.permute(1, 2, 0).cpu().numpy()
    
    if gt_mask is not None:
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    else:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axs[0].imshow(image_np)
    axs[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axs[0].axis('off')
    
    # Predicted mask
    axs[1].imshow(pred_mask, cmap='tab20')
    axs[1].set_title(f'Predicted Mask\n({pred_mask.max()} cells)', fontsize=14, fontweight='bold')
    axs[1].axis('off')
    
    # Overlay
    overlay = image_np.copy()
    if pred_mask.max() > 0:
        # Create colored mask
        colored_mask = plt.cm.tab20(pred_mask / (pred_mask.max() + 1))[:, :, :3]
        mask_binary = (pred_mask > 0).astype(float)[..., None]
        overlay = overlay * (1 - mask_binary * 0.5) + colored_mask * mask_binary * 0.5
    
    axs[2].imshow(overlay)
    axs[2].set_title('Overlay', fontsize=14, fontweight='bold')
    axs[2].axis('off')
    
    # Ground truth (if provided)
    if gt_mask is not None:
        axs[3].imshow(gt_mask, cmap='tab20')
        axs[3].set_title(f'Ground Truth\n({gt_mask.max()} cells)', fontsize=14, fontweight='bold')
        axs[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def calculate_metrics(pred_mask, gt_mask, iou_threshold=0.5):
    """
    Calculate detection metrics (precision, recall, F1)
    
    Args:
        pred_mask: (H, W) predicted instance mask
        gt_mask: (H, W) ground truth instance mask
        iou_threshold: IoU threshold for matching
    
    Returns:
        metrics: dict with precision, recall, F1
    """
    # Get unique cell IDs
    pred_ids = np.unique(pred_mask)[1:]  # Skip background (0)
    gt_ids = np.unique(gt_mask)[1:]
    
    # Match predictions to ground truth
    matched_pred = set()
    matched_gt = set()
    
    for pred_id in pred_ids:
        pred_region = (pred_mask == pred_id)
        
        best_iou = 0
        best_gt_id = None
        
        for gt_id in gt_ids:
            if gt_id in matched_gt:
                continue
            
            gt_region = (gt_mask == gt_id)
            
            intersection = np.sum(pred_region & gt_region)
            union = np.sum(pred_region | gt_region)
            iou = intersection / (union + 1e-8)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id
        
        if best_iou >= iou_threshold and best_gt_id is not None:
            matched_pred.add(pred_id)
            matched_gt.add(best_gt_id)
    
    # Calculate metrics
    tp = len(matched_pred)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - len(matched_gt)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'n_pred': len(pred_ids),
        'n_gt': len(gt_ids)
    }


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    model = StarDist(
        n_channels=3,
        n_rays=args.n_rays,
        base_filters=args.base_filters,
        shared_channels=args.shared_channels
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"✓ Best val loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataset = DSB2018Datasets(
        root=args.root,
        img_size=(args.img_size, args.img_size),
        split='val',
        n_rays=args.n_rays,
        augment=False
    )
    print(f"✓ Loaded {len(val_dataset)} validation images")
    
    # Predict and visualize
    print(f"\n{'='*80}")
    print("RUNNING INFERENCE")
    print(f"{'='*80}\n")
    
    all_metrics = []
    
    for idx in range(min(args.num_samples, len(val_dataset))):
        print(f"\nProcessing image {idx + 1}/{args.num_samples}...")
        
        # Get data
        image, prob_gt, rays_gt, labeled_mask = val_dataset[idx]
        
        gt_mask = labeled_mask.numpy() if hasattr(labeled_mask, 'numpy') else labeled_mask
        
        # Predict
        pred_mask, pred_prob = predict_image(
            model, image, device, 
            prob_thresh=args.prob_thresh,
            nms_thresh=args.nms_thresh
        )
        
        # Calculate metrics
        metrics = calculate_metrics(pred_mask, gt_mask, iou_threshold=0.5)
        all_metrics.append(metrics)
        
        print(f"  Predicted: {metrics['n_pred']} cells")
        print(f"  Ground truth: {metrics['n_gt']} cells")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1: {metrics['f1']:.3f}")
        
        # Visualize
        save_path = os.path.join(args.output_dir, f'prediction_{idx}.png')
        visualize_prediction(image, pred_mask, gt_mask, save_path)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    avg_precision = np.mean([m['precision'] for m in all_metrics])
    avg_recall = np.mean([m['recall'] for m in all_metrics])
    avg_f1 = np.mean([m['f1'] for m in all_metrics])
    
    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall: {avg_recall:.3f}")
    print(f"Average F1: {avg_f1:.3f}")
    
    print(f"\n✓ All results saved to {args.output_dir}/")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()