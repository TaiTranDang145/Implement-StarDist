import argparse
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import StarDist
from my_datasets import DSB2018Datasets
from tqdm import tqdm
import numpy as np
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--n-rays', type=int, default=32)
    parser.add_argument('--base-filters', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint', type=str, default='checkpoints')
    parser.add_argument("--logging", type=str, default="tensorboard")
    parser.add_argument("--trained_models", type=str, default="trained_models")
    return parser.parse_args()

class StarDistLoss(nn.Module):
    def __init__(self, w_prob=1.0, w_dist=0.4):
        super().__init__()
        self.w_prob = w_prob
        self.w_dist = w_dist
        self.bce = nn.BCEWithLogitsLoss()
        self.mae = nn.L1Loss()

    def forward(self, pred_prob, pred_dist, true_prob, true_dist):
        # Probability loss (binary cross entropy)
        prob_loss = self.bce(pred_prob, true_prob)

        # Distance loss (mean absolute error)
        # Only calculate loss on object pixels (true_prob > 0)
        mask = (true_prob > 0).float()
        dist_loss = self.mae(pred_dist * mask, true_dist * mask)

        return self.w_prob * prob_loss + self.w_dist * dist_loss


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset & DataLoader
    train_dataset = DSB2018Datasets(
        root=args.data_dir,
        img_size=(args.img_size, args.img_size),
        split='train',
        n_rays=args.n_rays,
        augment=True
    )
    val_dataset = DSB2018Datasets(
        root=args.data_dir,
        img_size=(args.img_size, args.img_size),
        split='val',
        n_rays=args.n_rays
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)

    # Model, Loss, Optimizer
    model = StarDist(
        n_channels=3,
        n_rays=args.n_rays,
        base_filter=args.base_filters
    ).to(device)

    writer = SummaryWriter(log_dir=args.logging)
    criterion = StarDistLoss(w_prob=1.0, w_dist=0.4)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        start_epoch = 0
        best_val_loss = 0


    # Load checkpoint if resuming
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print(f"Loading checkpoint '{args.resume}'")
    #         checkpoint = torch.load(args.resume, map_location=device)
    #         start_epoch = checkpoint['epoch'] + 1
    #         model.load_state_dict(checkpoint['model_state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    #         print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
    #     else:
    #         print(f"No checkpoint found at '{args.resume}'")
    #
    # writer = SummaryWriter()

    num_iters = len(train_loader)
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progressbar = tqdm(train_loader, colour='green')
        # with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}') as pbar:
        #     for imgs, probs, dists in pbar:
        #         imgs = imgs.to(device)
        #         probs = probs.to(device)
        #         dists = dists.to(device)
        #
        #         pred_probs, pred_dists = model(imgs)
        #         loss = criterion(pred_probs, pred_dists, probs, dists)
        #
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #
        #         train_loss += loss.item()
        #         pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        for iter, (image, prob, rays) in enumerate(progressbar):
            image = image.to(device)
            prob = prob.to(device)
            rays = rays.to(device)
            # forward
            pred_probs, pred_dists = model(image)
            loss_value = criterion(pred_probs, pred_dists, probs, dists)
            progressbar.set_description(f'Epoch {epoch + 1}/{args.epochs} Iter {iter + 1}/{num_iters} Loss: {loss_value:.3f}')

            # backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()


        # Validation
        model.eval()
        for iter, (image, prob, rays) in enumerate(val_loader):
            image = image.to(device)
            prob = prob.to(device)
            rays = rays.to(device)

            with torch.no_grad():
                pred_probs, pred_dists = model(image)
                val_loss = criterion(pred_probs, pred_dists, prob, rays).item()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, f'{args.save_dir}/best_model.pt')
            print(f'Saved new best model with validation loss: {best_val_loss:.4f}')

        # Save checkpoint periodically
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = f'{args.save_dir}/checkpoint_epoch_{epoch + 1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')

        scheduler.step()

    writer.close()


if __name__ == '__main__':
    main()
