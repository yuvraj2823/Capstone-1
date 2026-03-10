#!/usr/bin/env python3

import os
import argparse
import shutil
import random
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
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
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

    if os.path.isdir(os.path.join(raw_path, 'Normal')) and os.path.isdir(os.path.join(raw_path, 'TB')):
        print("Cleaned raw data already exists. Skipping download.")
        return
    
    if not os.path.isdir(raw_path):
        print("Raw data not found. Downloading...")
        download_kaggle_dataset('tawsifurrahman/tuberculosis-tb-chest-xray-dataset', raw_path)

    normalize_raw_folder(data_root, raw_dir_name)

# ------------------------------ Normalize raw folder ------------------------------

def normalize_raw_folder(data_root, raw_dir_name):

    raw_path = os.path.join(data_root, raw_dir_name)
    dest_normal = os.path.join(raw_path, 'Normal')
    dest_tb = os.path.join(raw_path, 'TB')

    if os.path.isdir(dest_normal) and os.path.isdir(dest_tb):
        return

    print("Normalizing raw dataset structure...")

    potential_class_dirs = []

    for root, dirs, files in os.walk(raw_path):
        if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
            rel = os.path.relpath(root, raw_path)
            if rel != '.':
                potential_class_dirs.append(root)

    normal_dirs = []
    tb_dirs = []

    for d in potential_class_dirs:
        name = os.path.basename(d).lower()
        if 'normal' in name:
            normal_dirs.append(d)
        elif 'tb' in name or 'tuberculosis' in name:
            tb_dirs.append(d)

    os.makedirs(dest_normal, exist_ok=True)
    os.makedirs(dest_tb, exist_ok=True)

    for src in normal_dirs:
        for f in os.listdir(src):
            if f.lower().endswith(('.png','.jpg','.jpeg')):
                shutil.copy2(os.path.join(src,f), os.path.join(dest_normal,f))

    for src in tb_dirs:
        for f in os.listdir(src):
            if f.lower().endswith(('.png','.jpg','.jpeg')):
                shutil.copy2(os.path.join(src,f), os.path.join(dest_tb,f))

    print("Normalization complete.")

# ------------------------------ Data splitting ------------------------------

def prepare_splits(data_root, raw_dir_name, val_split, test_split, seed):

    train_dir = os.path.join(data_root,'train')
    val_dir = os.path.join(data_root,'val')
    test_dir = os.path.join(data_root,'test')

    if os.path.isdir(train_dir) and os.path.isdir(val_dir) and os.path.isdir(test_dir):
        print("Train/val/test splits already exist.")
        return

    raw = os.path.join(data_root,raw_dir_name)
    classes=['Normal','TB']

    image_paths={c:[] for c in classes}

    for c in classes:
        folder=os.path.join(raw,c)
        files=[os.path.join(folder,f) for f in os.listdir(folder)
               if f.lower().endswith(('.png','.jpg','.jpeg'))]
        image_paths[c]=files

    random.seed(seed)

    splits={}

    for c in classes:

        random.shuffle(image_paths[c])

        n=len(image_paths[c])
        n_test=int(n*test_split)
        n_val=int(n*val_split)
        n_train=n-n_test-n_val

        splits[c]=(
            image_paths[c][:n_train],
            image_paths[c][n_train:n_train+n_val],
            image_paths[c][n_train+n_val:]
        )

    for split,folder in [('train',train_dir),('val',val_dir),('test',test_dir)]:
        for c in classes:
            os.makedirs(os.path.join(folder,c),exist_ok=True)

    for c in classes:
        train,val,test=splits[c]

        for f in train:
            shutil.copy(f, os.path.join(train_dir,c))
        for f in val:
            shutil.copy(f, os.path.join(val_dir,c))
        for f in test:
            shutil.copy(f, os.path.join(test_dir,c))

    print("Dataset splitting complete.")

# ------------------------------ Data loaders ------------------------------

def get_data_loaders(data_root,img_size,batch_size):

    train_transform=transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    test_transform=transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_dataset=datasets.ImageFolder(os.path.join(data_root,'train'),transform=train_transform)
    val_dataset=datasets.ImageFolder(os.path.join(data_root,'val'),transform=test_transform)
    test_dataset=datasets.ImageFolder(os.path.join(data_root,'test'),transform=test_transform)

    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=batch_size)
    test_loader=DataLoader(test_dataset,batch_size=batch_size)

    return train_loader,val_loader,test_loader,train_dataset.classes

# ------------------------------ Model ------------------------------

def create_model(num_classes=2):

    weights=ResNet50_Weights.DEFAULT
    model=models.resnet50(weights=weights)

    for p in model.parameters():
        p.requires_grad=False

    num_features=model.fc.in_features

    model.fc=nn.Sequential(
        nn.Linear(num_features,256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256,num_classes)
    )

    return model

# ------------------------------ Training ------------------------------

def train_one_epoch(model,loader,criterion,optimizer,device):

    model.train()

    running_loss=0
    correct=0
    total=0

    for inputs,labels in tqdm(loader):

        inputs,labels=inputs.to(device),labels.to(device)

        optimizer.zero_grad()

        outputs=model(inputs)

        loss=criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        running_loss+=loss.item()*inputs.size(0)

        _,pred=torch.max(outputs,1)

        total+=labels.size(0)
        correct+=(pred==labels).sum().item()

    return running_loss/total,100*correct/total

# ------------------------------ Validation ------------------------------

def validate(model,loader,criterion,device):

    model.eval()

    loss_total=0
    correct=0
    total=0

    with torch.no_grad():

        for inputs,labels in loader:

            inputs,labels=inputs.to(device),labels.to(device)

            outputs=model(inputs)

            loss=criterion(outputs,labels)

            loss_total+=loss.item()*inputs.size(0)

            _,pred=torch.max(outputs,1)

            total+=labels.size(0)
            correct+=(pred==labels).sum().item()

    return loss_total/total,100*correct/total

# ------------------------------ Training loop ------------------------------

def train_model(model,train_loader,val_loader,epochs,criterion,optimizer,device):

    best_acc=0

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss,train_acc=train_one_epoch(model,train_loader,criterion,optimizer,device)

        val_loss,val_acc=validate(model,val_loader,criterion,device)

        print(f"Train Loss {train_loss:.4f} Acc {train_acc:.2f}%")
        print(f"Val Loss {val_loss:.4f} Acc {val_acc:.2f}%")

        if val_acc>best_acc:

            best_acc=val_acc

            torch.save(model.state_dict(),"best_model.pth")

            print("Best model saved")

# ------------------------------ Evaluation ------------------------------

def evaluate(model,test_loader,class_names,device):

    model.eval()

    preds=[]
    labels_all=[]

    with torch.no_grad():

        for inputs,labels in test_loader:

            inputs=inputs.to(device)

            outputs=model(inputs)

            _,p=torch.max(outputs,1)

            preds.extend(p.cpu().numpy())

            labels_all.extend(labels.numpy())

    print(classification_report(labels_all,preds,target_names=class_names))
    print(confusion_matrix(labels_all,preds))

# ------------------------------ Main ------------------------------

def main():

    args=parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.data_root,exist_ok=True)

    ensure_data_exists(args.data_root,'raw')

    prepare_splits(args.data_root,'raw',args.val_split,args.test_split,args.seed)

    train_loader,val_loader,test_loader,class_names=get_data_loaders(
        args.data_root,args.img_size,args.batch_size
    )

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model=create_model(len(class_names)).to(device)

    criterion=nn.CrossEntropyLoss()

    optimizer=optim.Adam(model.fc.parameters(),lr=args.lr)

    train_model(model,train_loader,val_loader,args.epochs,criterion,optimizer,device)

    model.load_state_dict(torch.load("best_model.pth"))

    evaluate(model,test_loader,class_names,device)

if __name__=="__main__":
    main()