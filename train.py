"""
CSE 144 Final Project - Transfer Learning Pipeline (v2)
UC Santa Cruz, Spring 2026

Usage:
    Training:   python train.py --mode train --data_dir data/ --epochs 60
    Inference:  python train.py --mode test  --data_dir data/ --weights outputs/best_model.pth
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models

# ─────────────────────────────────────────────
# 0. Reproducibility
# ─────────────────────────────────────────────
SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ─────────────────────────────────────────────
# 1. Datasets
# ─────────────────────────────────────────────
class TrainDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.samples = []
        self.transform = transform
        class_dirs = sorted(os.listdir(root), key=lambda x: int(x))
        for label, class_name in enumerate(class_dirs):
            class_path = os.path.join(root, class_name)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class TestDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.transform = transform
        files = [f for f in os.listdir(root) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.files = sorted(files, key=lambda f: int(os.path.splitext(f)[0]))
        self.root = root

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.root, fname)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        img_id = int(os.path.splitext(fname)[0])
        return img, img_id


# ─────────────────────────────────────────────
# 2. Transforms
# ─────────────────────────────────────────────
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomRotation(20),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# TTA transforms — 5 views per image averaged at inference
tta_transforms = [
    transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
]


# ─────────────────────────────────────────────
# 3. Mixup
# ─────────────────────────────────────────────
def mixup_data(x, y, alpha=0.3):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─────────────────────────────────────────────
# 4. Model
# ─────────────────────────────────────────────
def build_model(num_classes: int, backbone: str = 'convnext_tiny'):
    if backbone == 'convnext_tiny':
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes),
        )
    elif backbone == 'efficientnet_b4':
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        model = models.efficientnet_b4(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    elif backbone == 'efficientnet_b2':
        weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
        model = models.efficientnet_b2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    elif backbone == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model


# ─────────────────────────────────────────────
# 5. Training helpers
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_mixup:
            imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, alpha=0.3)
            outputs = model(imgs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# 6. Main: train
# ─────────────────────────────────────────────
def run_training(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    full_dataset = TrainDataset(os.path.join(args.data_dir, 'train'), transform=train_transform)
    num_classes = len(set(label for _, label in full_dataset.samples))
    print(f"Found {len(full_dataset)} training images across {num_classes} classes.")

    val_size = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    val_set.dataset = TrainDataset(os.path.join(args.data_dir, 'train'), transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = build_model(num_classes, backbone=args.backbone)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-6)

    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    weights_path = os.path.join(args.output_dir, 'best_model.pth')

    print(f"\nTraining {args.backbone} for {args.epochs} epochs ...\n")
    for epoch in range(1, args.epochs + 1):
        # Disable mixup in final 10 epochs for cleaner convergence
        use_mixup = epoch <= (args.epochs - 10)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, use_mixup)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), weights_path)
            print(f"  ✓ Saved best model (val acc = {best_val_acc:.3f})")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.3f}")
    print(f"Weights saved to: {weights_path}")


# ─────────────────────────────────────────────
# 7. Main: inference (with TTA)
# ─────────────────────────────────────────────
@torch.no_grad()
def run_inference(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_root = os.path.join(args.data_dir, 'train')
    num_classes = len([d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))])
    print(f"Detected {num_classes} classes.")

    model = build_model(num_classes, backbone=args.backbone)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    model.eval()

    test_root = os.path.join(args.data_dir, 'test')
    test_files = sorted(
        [f for f in os.listdir(test_root) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda f: int(os.path.splitext(f)[0])
    )

    print(f"Running TTA inference on {len(test_files)} test images ({len(tta_transforms)} views each)...")

    all_ids, all_preds = [], []

    for fname in test_files:
        img_path = os.path.join(test_root, fname)
        img = Image.open(img_path).convert('RGB')
        img_id = int(os.path.splitext(fname)[0])

        # Average logits across all TTA views
        logits_sum = None
        for tfm in tta_transforms:
            tensor = tfm(img).unsqueeze(0).to(device)
            logits = model(tensor)
            if logits_sum is None:
                logits_sum = logits
            else:
                logits_sum += logits

        pred = logits_sum.argmax(dim=1).item()
        all_ids.append(img_id)
        all_preds.append(pred)

    df = pd.DataFrame({'ID': all_ids, 'Label': all_preds})
    df['ID'] = df['ID'].astype(str) + '.jpg'
    df = df.sort_values('ID', key=lambda x: x.str.replace('.jpg','').astype(int)).reset_index(drop=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'submission.csv')
    df.to_csv(out_path, index=False)
    print(f"Submission saved to: {out_path}  ({len(df)} rows)")


# ─────────────────────────────────────────────
# 8. CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='CSE 144 Transfer Learning Pipeline v2')
    parser.add_argument('--mode',       type=str,   default='train', choices=['train', 'test'])
    parser.add_argument('--data_dir',   type=str,   default='data/')
    parser.add_argument('--output_dir', type=str,   default='outputs/')
    parser.add_argument('--weights',    type=str,   default='outputs/best_model.pth')
    parser.add_argument('--backbone',   type=str,   default='convnext_tiny',
                        choices=['efficientnet_b2', 'efficientnet_b4', 'convnext_tiny', 'resnet50'])
    parser.add_argument('--epochs',     type=int,   default=60)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--seed',       type=int,   default=SEED)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'train':
        run_training(args)
    else:
        run_inference(args)
