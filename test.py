import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models import StarDist
from my_datasets import DSB2018Datasets
from post_processing import postprocess_batch, visualize_instances
from tqdm import tqdm
import os
import cv2

def get_args():
    parser = argparse.ArgumentParser(description='Test StarDist model with post-processing')
    parser.add_argument('--root', type=str, default='data', help='Data root directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--n-rays', type=int, default=32)
    parser.add_argument('--base-filters', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--prob-thresh', type=float, default=0.5, help='Probability threshold')
    parser.add_argument('--nms-thresh', type=float, default=0.4, help='NMS IoU threshold')
    parser.add_argument('--save-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to visualize')
    return parser.parse_args()

def load_model(checkpoint_path, n_channels, n_rays, base_filters, device):
    model = StarDist(n_channels=n_channels, n_rays=n_rays, base_filter=base_filters)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def test_model(model, dataloader, device, args):
    all_prediction = []
    all_images = []
    all_gt_masks = []
    all_centers = []
    all_scores = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader)):
            if args.split in ['train', 'val']:
                images, prob_gt, rays_gt = batch_data
                images = images.to(device)
                gt_masks = prob_gt.cpu().numpy()
            else:
                images, image_ids = batch_data
                images = images.to(device)
                gt_masks = None
            pred_probs, pred_dists = model(images)
            instance_masks, centers_batch, scores_batch = postprocess_batch(pred_probs, pred_dists,
                                                                prob_thresh=args.prob_thresh,
                                                                nms_thresh=args.nms_thresh)

            images_np = images.cpu().numpy()
            for i in range(images.shape[0]):
                img = images_np[i].transpose(1, 2, 0)
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                all_images.append(img)
                all_prediction.append(instance_masks[i])
                all_centers.append(centers_batch[i])
                all_scores.append(scores_batch[i])
                if gt_masks is not None:
                    all_gt_masks.append(gt_masks[i, 0])

    return all_prediction, all_images, all_gt_masks, all_centers, all_scores

def calculate_metrics(prediction, ground_truth):
    pred_counts = []
    gt_counts = []
    for pred, gt in zip(prediction, ground_truth):
        pred_counts.append(pred.max())
        gt_counts.append((gt > 0).sum() if gt is not None else 0)
    return pred_counts, gt_counts


def save_visualizations(images, predictions, centers_list, save_dir, num_samples=10):
    os.makedirs(save_dir, exist_ok=True)
    num_samples = min(num_samples, len(images))
    for i in range(num_samples):
        fig, axes = plt.subplots(1,2, figsize=(12, 6))

        # original image
        axes[0].imshow(images[i])
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # predicted with overlay
        vis_img = visualize_instances(images[i], predictions[i], centers = centers_list[i])
        axes[1].imshow(vis_img)
        axes[1].set_title('Predicted Instances')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'result_{i:03d}.png'), dpi = 150, bbox_inches='tight')
        plt.close()


def save_comparison(images, predictions, ground_truths, centers_list, save_dir, num_samples=20):
    os.makedirs(save_dir, exist_ok=True)
    num_samples = min(num_samples, len(images))

    for i in range(num_samples):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(images[i])
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Ground truth
        if ground_truths[i] is not None:
            gt_vis = (ground_truths[i] > 0).astype(np.uint8) * 255
            axes[1].imshow(images[i])
            axes[1].imshow(gt_vis, alpha=0.5, cmap='Reds')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')

        # Prediction
        vis_img = visualize_instances(images[i], predictions[i], centers_list[i])
        axes[2].imshow(vis_img)
        axes[2].set_title(f'Predictions ({predictions[i].max()} instances)')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'comparison_{i:03d}.png'), dpi=150, bbox_inches='tight')
        plt.close()
def visualize_sample(dataset, idx):
    sample = dataset.__getitem__(idx)
    if isinstance(sample, tuple) and len(sample) == 3:
        image, prob_gt, rays_gt = sample
        img = image.permute(1, 2, 0).numpy()
        gt_mask = prob_gt.squeeze().numpy()
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.title(f"Image index {idx}")
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(gt_mask, cmap='gray')
        plt.title("Ground truth mask")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        image, image_id = sample
        img = image.permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(f"Test image index {idx}")
        plt.axis('off')
        plt.show()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DSB2018Datasets(
        root=args.root,
        img_size=(args.img_size, args.img_size),
        split=args.split,
        n_rays=args.n_rays,
        augment=False
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = load_model(args.checkpoint, n_channels=3, n_rays=args.n_rays, base_filters=args.base_filters, device=device)

    predictions, images, gt_masks, centers_list, scores_list = test_model(model, dataloader, device, args)

    num_instances_per_image = [pred.max() for pred in predictions]
    print(f"Total images processed: {len(predictions)}")
    print(f"Average instances per image: {np.mean(num_instances_per_image):.2f}")
    print(f"Min instances: {np.min(num_instances_per_image)}")
    print(f"Max instances: {np.max(num_instances_per_image)}")

    #if args.split in ['train', 'val']:
        #pred_counts, gt_counts = calculate_metrics(predictions, gt_masks)
        # for i in range(len(pred_counts)):
        #     print(f"\nGround truth comparison:")
        #     print(f"Average predicted instances: {np.mean(pred_counts):.2f}")
        #     print(f"Average ground truth regions: {np.mean(gt_counts):.2f}")

    save_visualizations(images, predictions, centers_list, os.path.join(args.save_dir, 'visualizations'), args.num_samples)

    if args.split in ['train', 'val']:
        save_comparison(images, predictions, gt_masks, centers_list, args.save_dir, args.num_samples)
    else:
        save_visualizations(images, predictions, centers_list, args.save_dir, args.num_samples)
    #visualize_sample(dataset, 36)

if __name__ == '__main__':
    main()