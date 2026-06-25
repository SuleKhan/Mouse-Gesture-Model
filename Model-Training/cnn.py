import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# =====================================================
# CONFIG
# =====================================================

DATA_DIR = "Data-Collection/Gesture-Data"  # Path to the directory containing gesture images

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3

MODEL_PATH = "gesture_cnn.pt"
CLASSES_PATH = "gesture_classes.json"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {DEVICE}")


# =====================================================
# TRANSFORMS
# =====================================================

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


# =====================================================
# DATASET
# =====================================================
class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x), y
    
# -----------------------
# Full dataset (NO transform here)
# -----------------------
full_dataset = datasets.ImageFolder(DATA_DIR)

class_names = full_dataset.classes

print("Classes:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

with open(CLASSES_PATH, "w") as f:
    json.dump(class_names, f)

# -----------------------
# Split indices
# -----------------------
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_subset, val_subset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=generator
)

train_dataset = TransformDataset(train_subset, train_transform)
val_dataset = TransformDataset(val_subset, val_transform)

# -----------------------
# DataLoaders
# -----------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")


# =====================================================
# MODEL
# =====================================================

class GestureCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = GestureCNN(
    len(class_names)
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =====================================================
# HELPERS
# =====================================================

def compute_accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)
    return (preds == labels).float().mean().item()


history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}


# =====================================================
# TRAINING
# =====================================================

best_model_state = copy.deepcopy(
    model.state_dict()
)

best_val_acc = 0.0

for epoch in range(EPOCHS):

    # -----------------------
    # TRAIN
    # -----------------------

    model.train()

    train_loss = 0.0
    train_acc = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        train_acc += compute_accuracy(
            outputs,
            labels
        )

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    # -----------------------
    # VALIDATION
    # -----------------------

    model.eval()

    val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

            val_acc += compute_accuracy(
                outputs,
                labels
            )

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)

    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch+1:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            best_model_state,
            MODEL_PATH
        )

        print(
            f"Saved best model "
            f"(val_acc={val_acc:.4f})"
        )


# =====================================================
# FINAL RESULTS
# =====================================================

print()
print("=" * 50)
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print("=" * 50)

model.load_state_dict(best_model_state)


# =====================================================
# PLOTS
# =====================================================

epochs = range(
    1,
    len(history["train_loss"]) + 1
)

plt.figure(figsize=(8, 5))
plt.plot(
    epochs,
    history["train_loss"],
    label="Train Loss"
)
plt.plot(
    epochs,
    history["val_loss"],
    label="Validation Loss"
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(
    epochs,
    history["train_acc"],
    label="Train Accuracy"
)
plt.plot(
    epochs,
    history["val_acc"],
    label="Validation Accuracy"
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()
plt.grid(True)
plt.show()