import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A # data augmentation
import matplotlib.pyplot as plt
from stardist.geometry import star_dist
import matplotlib.pyplot as plt
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

        # data augmentation
        if self.augment:
            self.transform = A.Compose([
                A.Rotate(limit=180, p=0.5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT), # xoay ngau nhien -180 do -> 180 do, 50% anh duoc xoay
                A.HorizontalFlip(p=0.5), # lat trai, phai
                A.VerticalFlip(p=0.5), # lap tren, duoi
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT) # random uon cong
            ], additional_targets={'mask' : 'mask'})

    # tinh xac suat 1 pixel co phai la object hay khong
    def compute_prob_map_distance(self, labeled_mask):
        prob_map = np.zeros_like(labeled_mask, dtype = np.float32)
        all_object_mask = (labeled_mask > 0).astype(np.uint8)
        if not np.any(all_object_mask):
            return prob_map
        dist = distance_transform_edt(all_object_mask)
        dmax = dist.max()
        if dmax > 0:
            prob_map = (dist/dmax).astype(np.float32)
        
        return prob_map

    # tinh xac suat va khoang cach cua moi pixel
    def compute_star_dist(self, labeled_mask):
        
        prob_map = self.compute_prob_map_distance(labeled_mask)

        if np.max(labeled_mask) == 0:
            rays = np.zeros((self.n_rays, *labeled_mask.shape), dtype=np.float32)
            return prob_map, rays

        # star_dist trả về (H, W, n_rays)
        rays = star_dist(labeled_mask, n_rays=self.n_rays, mode = 'cpp')
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

            # gop cac label mask
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
                augmented = self.transform(image=image, mask=labeled_mask.astype(np.uint8))
                image = augmented['image']
                labeled_mask = augmented['mask'].astype(np.int32)

            #Tính probability map và rays
            prob_map, rays = self.compute_star_dist(labeled_mask)

            # --- Chuẩn hóa và chuyển sang tensor ---
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # (C,H,W)
            prob_map = torch.from_numpy(prob_map).unsqueeze(0)  # (1,H,W)
            rays = torch.from_numpy(rays)  # (n_rays,H,W)

            return image, prob_map, rays

        else:
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            return image, image_id

def visualize_augmentation(dataset, idx=0):
    image_id = dataset.image_ids[idx]
    path = os.path.join(dataset.data_folder, image_id)

    img_path = os.path.join(path, 'images', image_id + '.png')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dataset.img_size, interpolation=cv2.INTER_LINEAR)

    mask_dir = os.path.join(path, 'masks')
    mask_files = next(os.walk(mask_dir))[2]
    labeled_mask = np.zeros(dataset.img_size, dtype = np.int32)
    current_id = 1
    for mask_file in mask_files:
        mask_i = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        mask_i = cv2.resize(mask_i, dataset.img_size, interpolation=cv2.INTER_NEAREST)
        mask_bin = (mask_i > 0)
        labeled_mask[(labeled_mask == 0) & mask_bin] = current_id
        current_id += 1

    if dataset.augment:
        aug = dataset.transform(image = img, mask = labeled_mask.astype(np.uint16))
        img_aug = aug['image']
        mask_aug = aug['mask'].astype(np.int32)
    else:
        img_aug = img.copy()
        mask_aug = labeled_mask.copy()
    
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
    plt.show()

def visualize_dataset(dataset, idx = 0):

    image, prob_map, rays = dataset[idx]
    n_rays = rays.shape[0]  # Should be 32
    
    # Convert to numpy
    image_np = image.permute(1, 2, 0).numpy()
    prob_np = prob_map.squeeze().numpy()
    rays_np = rays.numpy()  # (n_rays, H, W)
    
    # Tạo composite visualization: vẽ tất cả 32 rays lên một ảnh
    H, W = rays_np.shape[1], rays_np.shape[2]
    rays_composite = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Lấy center points (pixels có prob > threshold)
    threshold = 0.5
    centers = np.where(prob_np > threshold)
    
    # Vẽ rays từ mỗi center pixel
    for y, x in zip(centers[0][::10], centers[1][::10]):  # Sample mỗi 10 pixels để không quá đông
        for k in range(n_rays):
            angle = 2 * np.pi * k / n_rays
            distance = rays_np[k, y, x]
            
            if distance > 0:
                # Tính endpoint của ray
                end_x = int(x + distance * np.sin(angle))
                end_y = int(y + distance * np.cos(angle))
                
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
    axs[1, 0].set_title(f'All {n_rays} Distance Rays\n(Color = ray direction)', 
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
    print(f"✓ Visualized {n_rays} rays from {len(centers[0][::10])} sample points")
    plt.show()

def main():
    dataset = DSB2018Datasets(
        root="data",
        img_size=(256, 256),
        split='train',
        n_rays=32,
        augment=True
    )

    # print(f"Dataset size: {len(dataset)}")
    # image, prob_map, rays = dataset[100]
    # print(f"Image shape: {image.shape}")
    # print(f"Prob map shape: {prob_map.shape}")
    # print(f"Rays shape: {rays.shape}")
    # print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    # print(f"Prob map range: [{prob_map.min():.3f}, {prob_map.max():.3f}]")
    # print(f"Rays range: [{rays.min():.3f}, {rays.max():.3f}]")
    visualize_augmentation(dataset, idx = 14)
    visualize_dataset(dataset, idx = 14)

if __name__ == '__main__':
    main()
