import time
import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm

ROOT = Path("/mnt/d/merged_fruit_36_clean_v2")
DATA_DIR = ROOT / "cnn_dataset_combined"
OUT_DIR = ROOT / "cnn_runs_transfer"
OUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
PATIENCE = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_tf)

if train_ds.classes != val_ds.classes:
    raise RuntimeError("Train và Val không cùng danh sách class.")

classes = train_ds.classes
num_classes = len(classes)

print("Classes:", num_classes)
print(classes)
print("Train:", len(train_ds))
print("Val:", len(val_ds))

targets = [y for _, y in train_ds.samples]
counts = Counter(targets)
class_weights = {cls: 1.0 / count for cls, count in counts.items()}
sample_weights = [class_weights[y] for y in targets]

sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)

best_acc = 0
best_loss = 999
no_improve = 0
history = []

start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} train", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)
        pred = out.argmax(1)
        train_correct += (pred == y).sum().item()
        train_total += y.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} val", leave=False):
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)

            val_loss += loss.item() * x.size(0)
            pred = out.argmax(1)
            val_correct += (pred == y).sum().item()
            val_total += y.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    scheduler.step(val_acc)

    print(f"\nEpoch {epoch}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    })

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "classes": classes,
        "best_val_acc": best_acc,
        "best_val_loss": best_loss,
        "img_size": IMG_SIZE,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "trained_on": "cnn_dataset_combined",
    }

    if val_acc > best_acc:
        best_acc = val_acc
        ckpt["best_val_acc"] = best_acc
        torch.save(ckpt, OUT_DIR / "best_acc_mobilenetv3_36class.pt")
        print("Saved best ACC model.")
        no_improve = 0
    else:
        no_improve += 1

    if val_loss < best_loss:
        best_loss = val_loss
        ckpt["best_val_loss"] = best_loss
        torch.save(ckpt, OUT_DIR / "best_loss_mobilenetv3_36class.pt")

    if no_improve >= PATIENCE:
        print("Early stopping.")
        break

with open(OUT_DIR / "history_combined.json", "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

print("\n===== DONE =====")
print("Best Val Acc:", best_acc)
print("Best Val Loss:", best_loss)
print("Model saved:", OUT_DIR / "best_acc_mobilenetv3_36class.pt")
print("Elapsed minutes:", round((time.time() - start) / 60, 2))
