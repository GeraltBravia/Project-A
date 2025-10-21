# File: swin_train_test.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import time
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np


# =====================
# 1. Config
# =====================
DATA_DIR = r'D:\doan\dataset_split_past1'   # chứa train/, val/, test/
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
MODEL_NAME = "swin_tiny_patch4_window7_224"
MODEL_PATH = "swin_tiny_finetuned.pth"

# =====================
# 2. Device
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================
# 3. Transform
# =====================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# =====================
# 4. Dataset & Loader
# =====================
train_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/train", transform=train_transform)
val_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/val", transform=test_transform)
test_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/test", transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(train_dataset.classes)
class_names = train_dataset.classes
print("Classes:", class_names)

# =====================
# 5. Model
# =====================
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
model.to(device)

# =====================
# 6. Loss & Optimizer
# =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# =====================
# 7. Training
# =====================
for epoch in range(EPOCHS):
    start = time.time()
    # --- Train ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = 100. * correct / total

    # --- Validate ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = 100. * val_correct / val_total

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
          f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% "
          f"| Time: {time.time()-start:.1f}s")

# =====================
# 8. Save Model
# =====================
torch.save(model.state_dict(), MODEL_PATH)
print("✅ Training Done! Model saved as", MODEL_PATH)

# =====================
# 9. Test Model
# =====================
print("\n=== TESTING MODEL ===")
model = timm.create_model(
    MODEL_NAME, pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

all_labels, all_preds = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = outputs.max(1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Accuracy & Report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# =====================
# 10. Evaluate & Save Results
# =====================
print("\n=== TESTING MODEL ===")
model = timm.create_model(
    MODEL_NAME, pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

all_labels, all_preds = [], []
start_test = time.time()
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = outputs.max(1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
end_test = time.time()

# Classification metrics
report = classification_report(
    all_labels, all_preds, target_names=class_names, output_dict=True)
accuracy = report["accuracy"]
precision = np.mean([report[c]["precision"] for c in class_names])
recall = np.mean([report[c]["recall"] for c in class_names])
f1 = np.mean([report[c]["f1-score"] for c in class_names])

# Prediction time per sample
num_samples = len(test_dataset)
time_per_sample = (end_test - start_test) / num_samples

# Model size (MB)
model_size = os.path.getsize(MODEL_PATH) / (1024*1024)

# Tổng thời gian train (đã in ở trên), bạn có thể lưu lại biến đó
# ở đây mình demo đơn giản: giả sử training_time_s và training_time_m đã được tính
# Nếu chưa có thì bạn tính bằng start_time ở đầu training và end_time sau EPOCHS
training_time_s = None  # TODO: nếu cần thì bạn lưu biến trong training loop
training_time_m = None

# GPU info
gpu_name = torch.cuda.get_device_name(
    0) if torch.cuda.is_available() else "CPU"

results = {
    "Algorithm": MODEL_NAME,
    "GPU": gpu_name,
    "Training time (s)": training_time_s,
    "Training time (m)": training_time_m,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
    "Prediction time per sample (s)": time_per_sample,
    "Model size (MB)": round(model_size, 2)
}

print("\n=== FINAL RESULTS ===")
for k, v in results.items():
    print(f"{k}: {v}")

# Save to Excel
df = pd.DataFrame([results])
df.to_excel("swin_results.xlsx", index=False)

# Save to TXT
with open("swin_results.txt", "w") as f:
    for k, v in results.items():
        f.write(f"{k}: {v}\n")

print("✅ Results saved to swin_results.xlsx and swin_results.txt")
