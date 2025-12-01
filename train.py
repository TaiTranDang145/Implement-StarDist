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
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--n-rays', type=int, default=32)
    parser.add_argument('--base-filters', type=int, default=64)
    parser.add_argument('--shared-channels', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument("--logging", type=str, default="tensorboard")
    parser.add_argument("--trained_models", type=str, default="trained_models")
    return parser.parse_args()

class StarDistLoss(nn.Module):
    def __init__(self, w_prob=1.0, w_dist=0.2):
        super().__init__()
        self.w_prob = w_prob
        self.w_dist = w_dist
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_prob, pred_dist, true_prob, true_dist):
        # Probability loss (binary cross entropy)
        prob_loss = self.bce(pred_prob, true_prob)

        with torch.no_grad():
            weights = true_prob

        diff = torch.abs(pred_dist - true_dist)
        weighted_diff = diff * weights
        denom = weights.sum() * pred_dist.size(1) + 1e-8
        dist_loss = weighted_diff.sum() / denom

        total_loss =  self.w_prob * prob_loss + self.w_dist * dist_loss
        return total_loss, prob_loss, dist_loss


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, num_iters):
    model.train()
    train_loss = 0.0
    train_prob_loss = 0.0
    train_dist_loss = 0.0

    progressbar = tqdm(dataloader, colour='green')
    for iter, (images, prob_gt, dist_gt) in enumerate(progressbar):
        images = images.to(device)
        prob_gt = prob_gt.to(device)
        dist_gt = dist_gt.to(device)

        # forward
        pred_probs, pred_dists = model(images)
        loss_value, prob_loss, dist_loss = criterion(pred_probs, pred_dists, prob_gt, dist_gt)

        train_loss += loss_value.item()
        train_prob_loss += prob_loss.item()
        train_dist_loss += dist_loss.item()

        progressbar.set_description(
            f'Epoch {epoch + 1} Iter {iter + 1}/{num_iters} Loss: {loss_value:.3f} '
        )
        step = epoch * num_iters + iter
        writer.add_scalar("Train/Loss", loss_value.item(), step)
        writer.add_scalar("Train/Prob_Loss", prob_loss.item(), step)
        writer.add_scalar("Train/Dist_Loss", dist_loss.item(), step)

        # backward
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

    avg_train_loss = train_loss / len(dataloader)
    avg_prob_loss = train_prob_loss / len(dataloader)
    avg_dist_loss = train_dist_loss / len(dataloader)

    writer.add_scalar("Epoch/Train_Loss", avg_train_loss, epoch)
    writer.add_scalar("Epoch/Train_Prob_Loss", avg_prob_loss, epoch)
    writer.add_scalar("Epoch/Train_Dist_Loss", avg_dist_loss, epoch)

    return avg_train_loss, avg_prob_loss, avg_dist_loss

def validate(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    val_loss = 0
    val_prob_loss = 0
    val_dist_loss = 0
   

    progressbar = tqdm(dataloader, colour='blue')
    with torch.no_grad():
        for iter, (images, prob_gt, dist_gt) in enumerate(progressbar):
            images = images.to(device)
            prob_gt = prob_gt.to(device)
            dist_gt = dist_gt.to(device)

            pred_probs, pred_dists = model(images)
            loss, prob_loss, dist_loss = criterion(pred_probs, pred_dists, prob_gt, dist_gt)

            
            val_loss += loss.item()
            val_prob_loss += prob_loss.item()
            val_dist_loss += dist_loss.item()

            progressbar.set_description(
                f'Validation Iter {iter + 1}/{len(dataloader)} Loss: {loss:.3f} '
            )


    avg_val_loss = val_loss / len(dataloader)
    avg_prob_loss = val_prob_loss / len(dataloader)
    avg_dist_loss = val_dist_loss / len(dataloader)
    

    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    writer.add_scalar('Loss/val_prob', avg_prob_loss, epoch)
    writer.add_scalar('Loss/val_dist', avg_dist_loss, epoch)

    

    return avg_val_loss, avg_prob_loss, avg_dist_loss


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset & DataLoader
    train_dataset = DSB2018Datasets(
        root=args.root,
        img_size=(args.img_size, args.img_size),
        split='train',
        n_rays=args.n_rays,
        augment=True
    )
    val_dataset = DSB2018Datasets(
        root=args.root,
        img_size=(args.img_size, args.img_size),
        split='val',
        n_rays=args.n_rays
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=False
    )

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)

    # Model, Loss, Optimizer
    model = StarDist(
        n_channels=3,
        n_rays=args.n_rays,
        base_filter=args.base_filters,
        shared_channels=args.shared_channels
    ).to(device)

    writer = SummaryWriter(log_dir=args.logging)
    criterion = StarDistLoss(w_prob=1.0, w_dist=0.2)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    start_epoch = 0
    best_val_loss = np.inf
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resumed from epoch {start_epoch}, Best val loss: {best_val_loss:.4f}")

    num_iters = len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss, train_prob_loss, train_dist_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, num_iters
        )


        val_loss, val_prob_loss, val_dist_loss = validate(
            model, val_loader, criterion, device, epoch, writer
        )


        # save best model
        last_check_point = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'val_loss': val_loss,
        }
        torch.save(last_check_point, os.path.join(args.trained_models, 'last_model.pt'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt = last_check_point.copy()
            torch.save(best_ckpt, os.path.join(args.trained_models, 'best_model.pt'))

        scheduler.step()

if __name__ == '__main__':
    main()
