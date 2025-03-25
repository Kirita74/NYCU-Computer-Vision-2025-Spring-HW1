import gc
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import CustomResnextModel


TRAIN_DATA_PATH = "data/train"
VAL_DATA_PATH = "data/val"
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
LOG_PATH = "logs/log12"
WEIGHT_PATH = ""
CLASS_MAPPING_FILE = "class_mapping.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms():
    """
    Get the data augments
    """

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.225, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.225, 0.225]
        )
    ])

    return train_transform, val_transform


def load_data():
    """
    Load data
    """

    train_transform, val_transform = get_transforms()

    train_dataset = torchvision.datasets.ImageFolder(
        root=TRAIN_DATA_PATH,
        transform=train_transform
    )

    val_dataset = torchvision.datasets.ImageFolder(
        root=VAL_DATA_PATH,
        transform=val_transform
    )

    with open(CLASS_MAPPING_FILE, "w+") as f:
        json.dump(train_dataset.class_to_idx, f)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_dataset, val_dataset, train_dataloader, val_dataloader


def train():
    # Docstring
    writer = SummaryWriter(log_dir="logs/log12")

    train_dataset, val_dataset, train_dataloader, val_dataloader = load_data()
    num_classes = len(train_dataset.classes)

    model = CustomResnextModel(num_classes=num_classes, pretrained=False)
    model.to(device=DEVICE)
    model.load_weight(path=WEIGHT_PATH)
    # change dimension of model's final output

    # model.load_state_dict(torch.load("resnext101_model_v2.pth", map_location = device))

    # Setting loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-2,
        betas=(0.9, 0.999)
    )

    # training

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='max',
        factor=0.3,
        patience=3,
        threshold=0.005,
        cooldown=2,
        min_lr=1e-7,
        verbose=True
    )
    
    num_epochs = 100
    best_valid_acc = 0.0

    for epoch in range(num_epochs):
        gc.collect()
        if epoch % 10 == 0:
            torch.cuda.empty_cache()

        model.train()
        running_loss = 0.0
        losses = []

        training_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            total=len(train_dataloader)
        )

        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            losses.append(loss.item())
            training_bar.update(1)
            training_bar.set_postfix(
                loss=f"{running_loss / len(train_dataloader):.4f}")

            writer.add_scalar(
                "Training Loss (Batch)",
                loss.item(),
                epoch * len(train_dataloader) + batch_idx
            )

        training_loss = np.mean(losses)
        writer.add_scalar("Training Loss (Epoch)", training_loss, epoch)

        model.eval()
        valid_correct = 0
        valid_total = 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()

        valid_acc = 100 * valid_correct / valid_total

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - "
            f"train Loss: {training_loss:.4f} - "
            f"validation accuracy: {valid_acc:.2f}%"
        )

        writer.add_scalar("Validation Accuracy", valid_acc, epoch)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            model.save_weight(path="resnext101_model_v4.pth")

        scheduler.step(valid_acc)
    writer.close()


if __name__ == '__main__':
    train()