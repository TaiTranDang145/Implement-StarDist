import numpy as np
import torch
from scipy.ndimage import label, center_of_mass
from skimage.draw import polygon
from typing import Tuple, List
import cv2

def rays_to_polygon(center, distance, n_rays = 32):
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    cy, cx = center
    vertices_y = cy + distance * np.sin(angles)
    vertices_x = cx + distance * np.cos(angles)
    vertices = np.stack([vertices_y, vertices_x], axis=1)
    return vertices

def polygon_to_mask(vertices, shape):
    mask = np.zeros(shape, dtype=np.uint8)

    vertices[:, 0] = np.clip(vertices[:, 0], 0, shape[0] - 1)
    vertices[:, 1] = np.clip(vertices[:, 1], 0, shape[1] - 1)
    rr, cc = polygon(vertices[:, 0], vertices[:, 1], shape)
    mask[rr, cc] = 1
    return mask


def non_maximum_suppression(prob_map, dist_map, prob_thresh=0.5, nms_thresh=0.4):
    candidates = prob_map > prob_thresh

    if not np.any(candidates):
        return [], [], []

    labeled_mask, num_features = label(candidates)

    if num_features == 0:
        return [], [], []


    centers = []
    scores = []
    distances_list = []

    for i in range(1, num_features + 1):
        mask_i = labeled_mask == i


        coords = np.argwhere(mask_i)
        probs = prob_map[mask_i]

        if len(coords) == 0:
            continue

        cy = int(np.average(coords[:, 0], weights=probs))
        cx = int(np.average(coords[:, 1], weights=probs))


        cy = np.clip(cy, 0, prob_map.shape[0] - 1)
        cx = np.clip(cx, 0, prob_map.shape[1] - 1)

        score = prob_map[cy, cx]
        distances = dist_map[:, cy, cx]

        centers.append((cy, cx))
        scores.append(score)
        distances_list.append(distances)

    if len(centers) == 0:
        return [], [], []

    indices = np.argsort(scores)[::-1]
    centers = [centers[i] for i in indices]
    scores = [scores[i] for i in indices]
    distances_list = [distances_list[i] for i in indices]

    # NMS
    keep = []
    masks = []

    for i, (center, dist, score) in enumerate(zip(centers, distances_list, scores)):
        vertices = rays_to_polygon(center, dist, n_rays=len(dist))
        mask = polygon_to_mask(vertices, prob_map.shape)

        overlap = False
        for kept_mask in masks:
            intersection = np.logical_and(mask, kept_mask).sum()
            union = np.logical_or(mask, kept_mask).sum()

            if union > 0:
                iou = intersection / union
                if iou > nms_thresh:
                    overlap = True
                    break

        if not overlap:
            keep.append(i)
            masks.append(mask)

            
    final_centers = [centers[i] for i in keep]
    final_distances = [distances_list[i] for i in keep]
    final_scores = [scores[i] for i in keep]

    return final_centers, final_distances, final_scores

def reconstruct_instances(prob_map, dist_map, prob_thresh = 0.5, nms_thresh = 0.4):
    centers, distances_list, scores = non_maximum_suppression(prob_map, dist_map, prob_thresh, nms_thresh)

    if len(centers) == 0:
        return np.zeros_like(prob_map.shape, dtype=np.int32), [], []

    instance_mask = np.zeros(prob_map.shape, dtype=np.int32)

    for instance_id, (center, distances) in enumerate(zip(centers, distances_list), start=1):
        vertices = rays_to_polygon(center, distances, n_rays=len(distances))
        mask = polygon_to_mask(vertices, prob_map.shape)
        instance_mask[mask == 1] = instance_id

    return instance_mask, centers, scores

def postprocess_batch(pred_probs, pred_dists, prob_thresh=0.5, nms_thresh=0.4):
    if isinstance(pred_probs, torch.Tensor):
        pred_probs = torch.sigmoid(pred_probs).cpu().numpy()
    if isinstance(pred_dists, torch.Tensor):
        pred_dists = pred_dists.cpu().numpy()
    batch_size = pred_probs.shape[0]
    instance_masks = []
    all_centers = []
    all_scores = []
    for i in range(batch_size):
        prob_map = pred_probs[i, 0]
        dist_map = pred_dists[i]
        instance_mask, centers, scores = reconstruct_instances(prob_map, dist_map, prob_thresh, nms_thresh)
        instance_masks.append(instance_mask)
        all_centers.append(centers)
        all_scores.append(scores)
    return instance_masks, all_centers, all_scores

def visualize_instances(image, instance_mask, centers=None, alpha=0.5):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image[:,:,0], cv2.COLOR_GRAY2BGR)

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    vis_image = image.copy()
    num_instances = instance_mask.max()

    if num_instances > 0:
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(num_instances + 1, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]

        colored_mask = colors[instance_mask]
        vis_image = cv2.addWeighted(vis_image, 1 - alpha, colored_mask, alpha, 0)

        for i in range(1, num_instances + 1):
            mask_i = (instance_mask == i).astype(np.uint8)
            contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, (255, 255, 255), 1)

    if centers is not None:
        for cy, cx in centers:
            cv2.circle(vis_image, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    return vis_image

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    iou = intersection / union
    return iou

if __name__ == "__main__":
    # Test code
    print("Testing post-processing functions")

    # Create dummy predictions
    H, W = 256, 256
    n_rays = 32

    # Fake probability map with 3 objects
    prob_map = np.zeros((H, W), dtype=np.float32)
    prob_map[50:80, 50:80] = 0.9
    prob_map[150:180, 150:180] = 0.85
    prob_map[100:130, 200:230] = 0.8

    # Fake distance map
    dist_map = np.random.rand(n_rays, H, W).astype(np.float32) * 20 + 10

    # Post-process
    instance_mask, centers, scores = reconstruct_instances(
        prob_map, dist_map, prob_thresh=0.5, nms_thresh=0.4
    )

    print(f"Number of instances detected: {instance_mask.max()}")
    print(f"Centers: {centers}")
    print(f"Scores: {scores}")
    print(f"Instance mask shape: {instance_mask.shape}")
    print(f"Unique IDs in mask: {np.unique(instance_mask)}")

    print("\n✓ Post-processing test completed!")