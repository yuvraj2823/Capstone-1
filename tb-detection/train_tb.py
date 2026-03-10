#!/usr/bin/env python3


import os
import sys
import argparse
import shutil
import random
import zipfile
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------ Argument parsing ------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Train TB detection model with PyTorch.')
    parser.add_argument('--data_root', type=str, default='tb_detection',
                        help='Root directory for data and models')
    parser.add_argument('--epochs_phase1', type=int, default=10)
    parser.add_argument('--epochs_phase2', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_phase1', type=float, default=1e-3)
    parser.add_argument('--lr_phase2', type=float, default=1e-5)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--test_split', type=float, default=0.15)
    return parser.parse_args()

# ------------------------------ Dataset download ------------------------------
def download_kaggle_dataset(dataset_name, target_dir):
    import kaggle
    print(f"Downloading {dataset_name} from Kaggle...")
    os.makedirs(target_dir, exist_ok=True)
    kaggle.api.dataset_download_files(dataset_name, path=target_dir, unzip=True)
    print("Download complete.")

def ensure_data_exists(data_root, raw_dir_name='raw'):
    raw_path = os.path.join(data_root, raw_dir_name)
    # Check if we already have the cleaned structure (Normal/TB folders)
    if os.path.isdir(os.path.join(raw_path, 'Normal')) and os.path.isdir(os.path.join(raw_path, 'TB')):
        print("Cleaned raw data already exists. Skipping download.")
        return

    # If raw folder exists but might be uncleaned, we still need to clean it.
    if not os.path.isdir(raw_path):
        print("Raw data not found. Downloading...")
        download_kaggle_dataset('tawsifurrahman/tuberculosis-tb-chest-xray-dataset', raw_path)

    # After download, normalize the folder structure
    normalize_raw_folder(data_root, raw_dir_name)

# ------------------------------ Normalize raw folder --------------------------
def normalize_raw_folder(data_root, raw_dir_name):
    """
    Scan the raw directory for class folders (case‑insensitive 'normal' and 'tb')
    and create a cleaned version with subfolders 'Normal' and 'TB'.
    """
    raw_path = os.path.join(data_root, raw_dir_name)
    dest_normal = os.path.join(raw_path, 'Normal')
    dest_tb = os.path.join(raw_path, 'TB')

    # If already cleaned, skip
    if os.path.isdir(dest_normal) and os.path.isdir(dest_tb):
        return

    print("Normalizing raw dataset structure...")
    # Find all subdirectories that might contain images
    potential_class_dirs = []
    for root, dirs, files in os.walk(raw_path):
        # If this directory contains image files, consider it a class folder
        image_extensions = ('.png', '.jpg', '.jpeg')
        if any(f.lower().endswith(image_extensions) for f in files):
            rel_path = os.path.relpath(root, raw_path)
            if rel_path != '.':  # not the root itself
                potential_class_dirs.append(root)

    if not potential_class_dirs:
        raise FileNotFoundError("No image-containing folders found in raw directory.")

    # Map each potential class dir to a normalized name
    normal_dirs = []
    tb_dirs = []
    for d in potential_class_dirs:
        folder_name = os.path.basename(d).lower()
        if 'normal' in folder_name:
            normal_dirs.append(d)
        elif 'tb' in folder_name or 'tuberculosis' in folder_name:
            tb_dirs.append(d)
        else:
            print(f"Warning: Ignoring folder {d} (does not contain 'normal' or 'tb')")

    if not normal_dirs or not tb_dirs:
        raise RuntimeError("Could not identify both 'normal' and 'tb' class folders. "
                           f"Found normal dirs: {normal_dirs}, tb dirs: {tb_dirs}")

    # Create destination folders
    os.makedirs(dest_normal, exist_ok=True)
    os.makedirs(dest_tb, exist_ok=True)

    # Copy all images from identified class folders to the corresponding destination
    for src_dir in normal_dirs:
        for fname in os.listdir(src_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                src = os.path.join(src_dir, fname)
                dst = os.path.join(dest_normal, fname)
                # Avoid overwriting if duplicate names exist (unlikely, but add suffix if needed)
                if os.path.exists(dst):
                    base, ext = os.path.splitext(fname)
                    dst = os.path.join(dest_normal, f"{base}_1{ext}")
                shutil.copy2(src, dst)

    for src_dir in tb_dirs:
        for fname in os.listdir(src_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                src = os.path.join(src_dir, fname)
                dst = os.path.join(dest_tb, fname)
                if os.path.exists(dst):
                    base, ext = os.path.splitext(fname)
                    dst = os.path.join(dest_tb, f"{base}_1{ext}")
                shutil.copy2(src, dst)

    print(f"Normalization complete. Copied images to {dest_normal} and {dest_tb}")

# ------------------------------ Data preparation ------------------------------
def prepare_splits(data_root, raw_dir_name, val_split, test_split, seed):
    """
    Create train/val/test folders using the cleaned raw data (Normal/TB folders).
    """
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    test_dir = os.path.join(data_root, 'test')

    # If splits already exist, skip
    if os.path.isdir(train_dir) and os.path.isdir(val_dir) and os.path.isdir(test_dir):
        if os.path.isdir(os.path.join(train_dir, 'Normal')) and os.path.isdir(os.path.join(train_dir, 'TB')):
            print("Train/val/test splits already exist. Skipping.")
            return

    raw_path = os.path.join(data_root, raw_dir_name)
    classes = ['Normal', 'TB']
    for cls in classes:
        cls_path = os.path.join(raw_path, cls)
        if not os.path.isdir(cls_path):
            raise FileNotFoundError(f"Expected class folder {cls_path} not found. "
                                    "Normalization may have failed.")

    # Get all image paths
    image_paths = {cls: [] for cls in classes}
    for cls in classes:
        cls_dir = os.path.join(raw_path, cls)
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_paths[cls] = files

    # Shuffle and split
    random.seed(seed)
    for cls in classes:
        random.shuffle(image_paths[cls])

    splits = {}
    for cls in classes:
        n_total = len(image_paths[cls])
        n_test = int(n_total * test_split)
        n_val = int(n_total * val_split)
        n_train = n_total - n_test - n_val

        train_files = image_paths[cls][:n_train]
        val_files = image_paths[cls][n_train:n_train+n_val]
        test_files = image_paths[cls][n_train+n_val:]

        splits[cls] = (train_files, val_files, test_files)

    # Create destination directories and copy files
    for split_name, split_dir in [('train', train_dir), ('val', val_dir), ('test', test_dir)]:
        for cls in classes:
            os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

    for cls in classes:
        train_f, val_f, test_f = splits[cls]
        for f in train_f:
            shutil.copy(f, os.path.join(train_dir, cls))
        for f in val_f:
            shutil.copy(f, os.path.join(val_dir, cls))
        for f in test_f:
            shutil.copy(f, os.path.join(test_dir, cls))

    print(f"Data split complete. Train: {sum(len(splits[c][0]) for c in classes)}, "
          f"Val: {sum(len(splits[c][1]) for c in classes)}, "
          f"Test: {sum(len(splits[c][2]) for c in classes)}")

# ------------------------------ Data loaders ----------------------------------
def get_data_loaders(data_root, img_size, batch_size, seed):
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(data_root, 'train'),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_root, 'val'),
        transform=val_test_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(data_root, 'test'),
        transform=val_test_transform
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_dataset.classes

# ------------------------------ Model creation --------------------------------
def create_model(num_classes=2):
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model

# ------------------------------ Training functions ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, epochs, criterion,
                optimizer, device, phase_name):
    best_val_acc = 0.0

    for epoch in range(1, epochs+1):
        print(f"\n{phase_name} - Epoch {epoch}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.data_root, 'models', 'best_model.pth'))
            print("Best model saved.")

# ------------------------------ Evaluation ------------------------------------
def evaluate(model, test_loader, class_names, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

# ------------------------------ Main ------------------------------------------
def main():
    global args
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(os.path.join(args.data_root, 'models'), exist_ok=True)

    # 1. Download dataset (if needed) and normalize structure
    ensure_data_exists(args.data_root, 'raw')

    # 2. Create train/val/test splits (using cleaned raw data)
    prepare_splits(args.data_root, 'raw', args.val_split, args.test_split, args.seed)

    # 3. Data loaders
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        args.data_root, args.img_size, args.batch_size, args.seed
    )
    print(f"Classes: {class_names}")

    # 4. Model, loss, device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = create_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()

    # 5. Phase 1: Train only classifier
    print("\n========== PHASE 1: Training classifier ==========")
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr_phase1)
    train_model(model, train_loader, val_loader, args.epochs_phase1,
                criterion, optimizer, device, "Phase 1")

    # 6. Phase 2: Fine-tune all layers
    print("\n========== PHASE 2: Fine-tuning all layers ==========")
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=args.lr_phase2)
    train_model(model, train_loader, val_loader, args.epochs_phase2,
                criterion, optimizer, device, "Phase 2")

    # 7. Evaluate on test set
    print("\n========== Final Evaluation on Test Set ==========")
    model.load_state_dict(torch.load(os.path.join(args.data_root, 'models', 'best_model.pth')))
    evaluate(model, test_loader, class_names, device)

    # 8. Save final model
    final_model_path = os.path.join(args.data_root, 'models', 'tb_model_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Export to TorchScript
    scripted_model = torch.jit.script(model)
    scripted_model.save(os.path.join(args.data_root, 'models', 'tb_model_scripted.pt'))
    print("TorchScript model saved.")

if __name__ == '__main__':
    main()