import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models.tb_model import TBClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trainer")

def train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.getenv("DATASET_PATH", os.path.join(base_dir, "dataset", "TBX11K"))
    save_path = os.getenv("MODEL_PATH", os.path.join(base_dir, "models", "tb_model.pth"))
    
    batch_size = 16
    num_epochs = 30
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    class XRayNormalize(object):
        def __call__(self, img_tensor):
            return img_tensor * 2048.0 - 1024.0

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        XRayNormalize()
    ])

    class TBXDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.transform = transform
            self.samples = []
            
            health_dir = os.path.join(root_dir, "imgs", "health")
            sick_dir = os.path.join(root_dir, "imgs", "sick")
            tb_dir = os.path.join(root_dir, "imgs", "tb")
            
            for label, d in [(0.0, health_dir), (0.0, sick_dir), (1.0, tb_dir)]:
                if os.path.exists(d):
                    for f in os.listdir(d):
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((os.path.join(d, f), label))
            
            if not self.samples:
                logger.error(f"No images found in {root_dir}")
                return

            logger.info(f"Loaded {len(self.samples)} images.")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('L')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label)

    dataset = TBXDataset(root_dir=dataset_dir, transform=train_transform)
    if not dataset.samples: return
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TBClassifier().to(device)
    
    pos_count = sum(1 for _, l in dataset.samples if l == 1.0)
    neg_count = len(dataset) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count]).to(device) if pos_count > 0 else None
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    best_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 7

    logger.info("Starting training loop...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        all_labels = []
        all_preds = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()
            
            optimizer.zero_grad()
            probs, _ = model(images)
            loss = nn.BCELoss()(probs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend((probs > 0.5).cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"  > Loss:      {avg_loss:.4f}")
        logger.info(f"  > Accuracy:  {acc:.4f}")
        logger.info(f"  > Precision: {prec:.4f}")
        logger.info(f"  > Recall:    {rec:.4f}")
        logger.info(f"  > F1-Score:  {f1:.4f}")
        
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict()}, save_path)
            logger.info(f"Saved new best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info("Early stopping triggered.")
                break

    logger.info("Training complete.")

if __name__ == "__main__":
    train_model()
