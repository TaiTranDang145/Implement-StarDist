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
    parser.add_argument('--checkpoint', type=str, default=None)
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
        if mask.sum() > 0:
            dist_loss = (torch.abs(pred_dist - true_dist) * mask).sum() / mask.sum()
        else:
            dist_loss = torch.tensor(0.0, device=pred_dist.device)

        total_loss =  self.w_prob * prob_loss + self.w_dist * dist_loss
        return total_loss, prob_loss, dist_loss

def calculate_metrics(pred_prob, true_prob, threshold=0.5):
    pred_prob_sigmoid = torch.sigmoid(pred_prob)
    pred_binary = (pred_prob_sigmoid >= threshold).float()
    true_binary = (true_prob >= threshold).float()

    tp = (pred_binary * true_binary).sum().item()
    fp = (pred_binary * (1 - true_binary)).sum().item()
    fn = ((1 - pred_binary) * true_binary).sum().item()
    tn = ((1 - pred_binary) * (1 - true_binary)).sum().item()

    # IOU
    intersection = tp
    union = tp + fp + fn
    iou = intersection / (union + 1e-6)

    # Precision, Recall, F1-Score
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    # accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy
    }

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, num_iters):
    model.train()
    train_loss = 0
    train_prob_loss = 0
    train_dist_loss = 0
    train_metrics = {'iou': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'accuracy': 0}
    progressbar = tqdm(dataloader, colour='green')
    for iter, (images, prob, rays) in enumerate(progressbar):
        images = images.to(device)
        prob = prob.to(device)
        rays = rays.to(device)

        # forward
        pred_probs, pred_dists = model(images)
        loss_value, prob_loss, dist_loss = criterion(pred_probs, pred_dists, prob, rays)

        with torch.no_grad():
            metrics = calculate_metrics(pred_probs, prob)
            for key in train_metrics.keys():
                train_metrics[key] += metrics[key]
        train_loss += loss_value.item()
        train_prob_loss += prob_loss.item()
        train_dist_loss += dist_loss.item()

        progressbar.set_description(
            f'Epoch {epoch + 1} Iter {iter + 1}/{num_iters} Loss: {loss_value:.3f} '
            f'IOU: {metrics["iou"]:.3f} F1: {metrics["f1_score"]:.3f}'
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
    for key in train_metrics.keys():
        train_metrics[key] /= len(dataloader)


    return avg_train_loss, avg_prob_loss, avg_dist_loss, train_metrics

def validate(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    val_loss = 0
    val_prob_loss = 0
    val_dist_loss = 0
    val_metrics = {'iou': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'accuracy': 0}

    # visualization
    sample_images = []
    sample_preds = []
    sample_gts = []
    progressbar = tqdm(dataloader, colour='blue')
    with torch.no_grad():
        for iter, (images, prob, rays) in enumerate(progressbar):
            images = images.to(device)
            prob = prob.to(device)
            rays = rays.to(device)

            pred_probs, pred_dists = model(images)
            loss, prob_loss, dist_loss = criterion(pred_probs, pred_dists, prob, rays)

            metrics = calculate_metrics(pred_probs, prob)
            for key in val_metrics.keys():
                val_metrics[key] += metrics[key]
            val_loss += loss.item()
            val_prob_loss += prob_loss.item()
            val_dist_loss += dist_loss.item()

            progressbar.set_description(
                f'Validation Iter {iter + 1}/{len(dataloader)} Loss: {loss:.3f} '
                f'IOU: {metrics["iou"]:.3f} F1: {metrics["f1_score"]:.3f}'
            )

            if iter == 0:
                sample_images = images[:4].cpu()
                sample_preds = torch.sigmoid(pred_probs[:4]).cpu()
                sample_gts = prob[:4].cpu()

    avg_val_loss = val_loss / len(dataloader)
    avg_prob_loss = val_prob_loss / len(dataloader)
    avg_dist_loss = val_dist_loss / len(dataloader)
    for key in val_metrics.keys():
        val_metrics[key] /= len(dataloader)

    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    writer.add_scalar('Loss/val_prob', avg_prob_loss, epoch)
    writer.add_scalar('Loss/val_dist', avg_dist_loss, epoch)

    for key, value in val_metrics.items():
        writer.add_scalar(f'Metrics/val_{key}', value, epoch)
    # Log sample images
    if epoch % 5 == 0:
        writer.add_images('Images/Input', sample_images, epoch)
        writer.add_images('prediction/prob', sample_preds, epoch)
        writer.add_images('GroundTruth/prob', sample_gts, epoch)

    return avg_val_loss, avg_prob_loss, avg_dist_loss, val_metrics


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
        base_filter=args.base_filters
    ).to(device)

    writer = SummaryWriter(log_dir=args.logging)
    criterion = StarDistLoss(w_prob=1.0, w_dist=0.4)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    start_epoch = 0
    best_iou = -np.inf
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint.get('best_iou', best_iou)
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resumed from epoch {start_epoch}, Best IoU: {best_iou:.4f}")

    num_iters = len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss, train_prob_loss, train_dist_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, num_iters
        )
        # writer.add_scalar('Loss/train', train_loss, epoch)
        # writer.add_scalar('Loss/train_prob', train_prob_loss, epoch)
        # writer.add_scalar('Loss/train_dist', train_dist_loss, epoch)

        for key, value in train_metrics.items():
            writer.add_scalar(f'Metrics/train_{key}', value, epoch)


        val_loss, val_prob_loss, val_dist_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch, writer
        )


        # save best model
        current_iou = val_metrics['iou']
        if current_iou > best_iou:
            best_iou = current_iou
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, os.path.join(args.trained_models, 'best_model.pt'))

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_iou': best_iou,
            'val_loss': val_loss,
            'val_metrics': val_metrics
        }
        torch.save(checkpoint, os.path.join(args.trained_models, 'last_model.pt'))

        scheduler.step()

if __name__ == '__main__':
    main()
