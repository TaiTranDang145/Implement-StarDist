import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A # data augmentation
import matplotlib.pyplot as plt
from stardist.geometry import star_dist


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
                A.Rotate(limit=180, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3)
            ])

    def compute_star_dist(self, labeled_mask):
        """Tính khoảng cách star-convex theo chuẩn StarDist"""
        prob_map = (labeled_mask > 0).astype(np.float32)

        if np.max(labeled_mask) == 0:
            rays = np.zeros((self.n_rays, *labeled_mask.shape), dtype=np.float32)
            return prob_map, rays

        # star_dist trả về (H, W, n_rays)
        rays = star_dist(labeled_mask, n_rays=self.n_rays)
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


def main():
    dataset = DSB2018Datasets(
        root="data",
        img_size=(256, 256),
        split='train',
        n_rays=32,
        augment=False
    )

    print(f"Dataset size: {len(dataset)}")

    image, prob_map, rays = dataset[100]
    print(f"Image shape: {image.shape}")
    print(f"Prob map shape: {prob_map.shape}")
    print(f"Rays shape: {rays.shape}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Prob map range: [{prob_map.min():.3f}, {prob_map.max():.3f}]")
    print(f"Rays range: [{rays.min():.3f}, {rays.max():.3f}]")



if __name__ == '__main__':
    main()
