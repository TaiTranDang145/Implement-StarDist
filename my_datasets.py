import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
import matplotlib.pyplot as plt
from stardist.geometry import star_dist
from scipy.ndimage import distance_transform_edt

class DSB2018Datasets(Dataset):
    def __init__(self, root="data", img_size=(256, 256), split='train', val_ratio=0.1,
                 seed=42, n_rays=32, augment=False):
        self.split = split
        self.img_size = img_size
        self.n_rays = n_rays
        self.augment = augment

        # lấy thư mục train/val
        data_folder = os.path.join(root, 'stage1_train') if split in ['train', 'val'] else os.path.join(root, 'stage1_test')
        self.data_folder = data_folder

        # lấy danh sách image ID
        image_ids = next(os.walk(self.data_folder))[1]

        # chia dữ liệu train/val
        if split in ['train', 'val']:
            train_ids, val_ids = train_test_split(image_ids, test_size=val_ratio, random_state=seed)
            self.image_ids = train_ids if split == 'train' else val_ids
        else:
            self.image_ids = image_ids

        # data augmentation - QUAN TRỌNG: phải đảm bảo mask dùng nearest interpolation
        if self.augment:
            self.transform = A.Compose([
                A.Rotate(limit=180, p=0.5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3, 
                                   interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT)
            ], additional_targets={'mask': 'mask'})

    def compute_prob_map_distance(self, labeled_mask):
        """
        Tính object probability map theo IMPLEMENTATION GỐC của StarDist:
        "normalized Euclidean distance to the nearest background pixel"
        
        QUAN TRỌNG: Normalize TỪNG CELL về [0, 1] để mọi cell đều có prob max = 1.0
        """
        prob_map = np.zeros_like(labeled_mask, dtype=np.float32)
        
        # Lấy số lượng objects
        num_objects = labeled_mask.max()
        
        if num_objects == 0:
            return prob_map
        
        # Tính prob cho TỪNG object riêng biệt
        for obj_id in range(1, num_objects + 1):
            obj_mask = (labeled_mask == obj_id)
            
            if not np.any(obj_mask):
                continue
            
            # Distance transform CHO object này
            dist = distance_transform_edt(obj_mask)
            
            # Normalize bằng MAX của OBJECT NÀY (không phải toàn ảnh)
            dmax = dist.max()
            if dmax > 0:
                # Mỗi object sẽ có center = 1.0, edge = 0.0
                prob_map[obj_mask] = (dist[obj_mask] / dmax).astype(np.float32)
        
        return prob_map

    def compute_star_dist(self, labeled_mask):
        """
        Tính probability map và radial distances
        """
        prob_map = self.compute_prob_map_distance(labeled_mask)

        if np.max(labeled_mask) == 0:
            rays = np.zeros((self.n_rays, *labeled_mask.shape), dtype=np.float32)
            return prob_map, rays

        # star_dist trả về (H, W, n_rays)
        # mode='cpp' nhanh hơn và chính xác hơn python version
        rays = star_dist(labeled_mask, n_rays=self.n_rays, mode='cpp')
        rays = np.moveaxis(rays, -1, 0).astype(np.float32)  # (n_rays, H, W)

        return prob_map, rays

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        path = os.path.join(self.data_folder, image_id)
        
        image_path = os.path.join(path, 'images', image_id + '.png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)

        if self.split in ['train', 'val']:
            mask_dir = os.path.join(path, 'masks')
            mask_files = next(os.walk(mask_dir))[2]

            # Gộp các label mask
            labeled_mask = np.zeros(self.img_size, dtype=np.int32)
            current_id = 1
            for mask_file in mask_files:
                mask_i = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
                mask_i = cv2.resize(mask_i, self.img_size, interpolation=cv2.INTER_NEAREST)
                mask_bin = (mask_i > 0)
                # Gán ID mới cho vùng chưa được gán
                labeled_mask[(labeled_mask == 0) & mask_bin] = current_id
                current_id += 1

            if self.augment:
                # Albumentations tự động xử lý mask với nearest interpolation
                augmented = self.transform(image=image, mask=labeled_mask.astype(np.uint16))
                image = augmented['image']
                labeled_mask = augmented['mask'].astype(np.int32)

            # Tính probability map và rays
            prob_map, rays = self.compute_star_dist(labeled_mask)

            # Chuẩn hóa và chuyển sang tensor
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # (C,H,W)
            prob_map = torch.from_numpy(prob_map).unsqueeze(0)  # (1,H,W)
            rays = torch.from_numpy(rays)  # (n_rays,H,W)
            labeled_mask = torch.from_numpy(labeled_mask).long()  # (H,W) - THÊM labeled_mask

            return image, prob_map, rays, labeled_mask  # TRẢ VỀ 4 items

        else:
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            return image, image_id


def visualize_augmentation(dataset, idx=0):
    """
    Visualize original vs augmented image WITH MASK overlay
    """
    image_id = dataset.image_ids[idx]
    path = os.path.join(dataset.data_folder, image_id)
    
    # Load image
    img_path = os.path.join(path, 'images', image_id + '.png')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dataset.img_size, interpolation=cv2.INTER_LINEAR)
    
    # Load and merge masks
    mask_dir = os.path.join(path, 'masks')
    mask_files = next(os.walk(mask_dir))[2]
    
    labeled_mask = np.zeros(dataset.img_size, dtype=np.int32)
    current_id = 1
    for mask_file in mask_files:
        mask_i = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        mask_i = cv2.resize(mask_i, dataset.img_size, interpolation=cv2.INTER_NEAREST)
        mask_bin = (mask_i > 0)
        labeled_mask[(labeled_mask == 0) & mask_bin] = current_id
        current_id += 1
    
    # Apply augmentation if enabled
    if dataset.augment:
        aug = dataset.transform(image=img, mask=labeled_mask.astype(np.uint16))
        img_aug = aug['image']
        mask_aug = aug['mask'].astype(np.int32)
    else:
        img_aug = img.copy()
        mask_aug = labeled_mask.copy()
    
    # Create visualization
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image and mask
    axs[0, 0].imshow(img)
    axs[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(labeled_mask, cmap='tab20')
    axs[0, 1].set_title(f'Original Mask ({labeled_mask.max()} cells)', fontsize=12, fontweight='bold')
    axs[0, 1].axis('off')
    
    # Augmented image and mask
    axs[1, 0].imshow(img_aug)
    axs[1, 0].set_title('Augmented Image', fontsize=12, fontweight='bold')
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(mask_aug, cmap='tab20')
    axs[1, 1].set_title(f'Augmented Mask ({mask_aug.max()} cells)', fontsize=12, fontweight='bold')
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved to augmentation.png")
    plt.show()


def visualize_dataset(dataset, idx=0):
    """
    Visualize one sample from dataset: image, prob_map, and all 32 distance rays
    """
    image, prob_map, rays = dataset[idx]
    n_rays = rays.shape[0]  # Should be 32
    
    # Convert to numpy
    image_np = image.permute(1, 2, 0).numpy()
    prob_np = prob_map.squeeze().numpy()
    rays_np = rays.numpy()  # (n_rays, H, W)
    
    # Tạo composite visualization: vẽ tất cả 32 rays lên một ảnh
    H, W = rays_np.shape[1], rays_np.shape[2]
    rays_composite = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Tìm CENTER của mỗi cell bằng cách tìm local maxima trong prob_map
    from scipy.ndimage import label, maximum_filter
    
    # Threshold thấp hơn để bắt được nhiều cells hơn
    binary_mask = prob_np > 0.1  # Giảm từ 0.5 xuống 0.1
    
    # Label connected components
    labeled_mask, num_cells = label(binary_mask)
    
    # Tìm center của mỗi cell (pixel có prob cao nhất trong mỗi cell)
    centers = []
    for cell_id in range(1, num_cells + 1):
        cell_mask = (labeled_mask == cell_id)
        if not np.any(cell_mask):
            continue
        
        # Tìm pixel có prob max trong cell này
        cell_probs = prob_np * cell_mask
        center_idx = np.unravel_index(np.argmax(cell_probs), prob_np.shape)
        centers.append(center_idx)
    
    print(f"Found {len(centers)} cell centers to visualize")
    
    # Vẽ rays từ CENTER của mỗi cell
    for y, x in centers:
        for k in range(n_rays):
            angle = 2 * np.pi * k / n_rays
            distance = rays_np[k, y, x]
            
            if distance > 0:
                # Tính endpoint của ray
                end_x = int(x + distance * np.sin(angle))
                end_y = int(y + distance * np.cos(angle))
                
                # Clip về trong bounds
                end_x = np.clip(end_x, 0, W - 1)
                end_y = np.clip(end_y, 0, H - 1)
                
                # Vẽ line từ center tới boundary
                color = plt.cm.hsv(k / n_rays)[:3]  # Mỗi ray một màu khác nhau
                color = tuple([int(c * 255) for c in color])
                cv2.line(rays_composite, (x, y), (end_x, end_y), color, 1)
    
    # Overlay rays lên image gốc
    overlay = (image_np * 255).astype(np.uint8).copy()
    mask = rays_composite.sum(axis=2) > 0
    overlay[mask] = overlay[mask] * 0.3 + rays_composite[mask] * 0.7
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    
    # Image
    axs[0, 0].imshow(image_np)
    axs[0, 0].set_title('Image', fontsize=14, fontweight='bold')
    axs[0, 0].axis('off')
    
    # Probability Map
    im1 = axs[0, 1].imshow(prob_np, cmap='hot', vmin=0, vmax=1)
    axs[0, 1].set_title(f'Probability Map\n(min={prob_np.min():.3f}, max={prob_np.max():.3f})', 
                     fontsize=14, fontweight='bold')
    axs[0, 1].axis('off')
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)
    
    # All 32 Distance Rays visualized
    axs[1, 0].imshow(overlay)
    axs[1, 0].set_title(f'All {n_rays} Distance Rays from {len(centers)} cells\n(Color = ray direction)', 
                        fontsize=14, fontweight='bold')
    axs[1, 0].axis('off')
    
    # Mean distance across all rays (để xem pattern tổng thể)
    mean_dist = rays_np.mean(axis=0)
    im2 = axs[1, 1].imshow(mean_dist, cmap='viridis')
    axs[1, 1].set_title(f'Mean Distance (all rays)\n(min={mean_dist.min():.1f}, max={mean_dist.max():.1f})', 
                     fontsize=14, fontweight='bold')
    axs[1, 1].axis('off')
    plt.colorbar(im2, ax=axs[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('dataset_visualization.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved to dataset_visualization.png")
    print(f"✓ Visualized {n_rays} rays from {len(centers)} cell centers")
    plt.show()


def main():
    """Test dataset"""
    dataset = DSB2018Datasets(
        root="data",
        img_size=(256, 256),
        split='train',
        n_rays=32,
        augment=True
    )

    print(f"Dataset size: {len(dataset)}")
    
    # Test one sample
    image, prob_map, rays = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Prob map shape: {prob_map.shape}")
    print(f"  Rays shape: {rays.shape}")
    print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Prob map range: [{prob_map.min():.3f}, {prob_map.max():.3f}]")
    print(f"  Rays range: [{rays.min():.3f}, {rays.max():.3f}]")

    # Visualizations
    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)
    
    # 1. Visualize dataset sample
    print("\n1. Visualizing dataset sample (image, prob_map, rays)...")
    visualize_dataset(dataset, idx=0)
    
    # 2. Visualize augmentation
    print("\n2. Visualizing augmentation (original vs augmented)...")
    visualize_augmentation(dataset, idx=0)
    
    print("\n" + "="*60)
    print("✓ All visualizations completed!")
    print("="*60)


if __name__ == '__main__':
    main()