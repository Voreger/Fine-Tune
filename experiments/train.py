import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm

from configs import TrainConfig

# Конфигурация
cfg = TrainConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("outputs", exist_ok=True)

# DataLoader
def get_dataloaders(cfg):
    train_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    full_dataset = datasets.ImageFolder(cfg.data_dir, transform=train_tf)
    val_size = int(len(full_dataset) * cfg.val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    class_names = full_dataset.classes
    return train_loader, val_loader, class_names

# Сборка модели
def build_model(cfg):
    if cfg.model_name == "resnet18":
        model = timm.create_model("resnet18", pretrained=True, num_classes=cfg.num_classes)
    elif cfg.model_name == "efficientnet_b0":
        model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=cfg.num_classes)
    else:
        raise ValueError(f"Unknown model: {cfg.model_name}")
    return model

def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
    # Разморозим классификатор
    for param in model.get_classifier().parameters():
        param.requires_grad = True


# Обучение одной модели
def train_one_model(cfg):
    train_loader, val_loader, class_names = get_dataloaders(cfg)
    model = build_model(cfg)
    freeze_backbone(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)

    best_acc = 0
    # Сохраняем истории обучения
    history = {"train_acc": [], "val_acc": []}

    for epoch in range(cfg.epochs):
        model.train()
        correct, total = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total
        history["train_acc"].append(train_acc)

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1).cpu()
                all_preds.extend(preds)
                all_labels.extend(labels)
        val_acc = sum([p==l for p,l in zip(all_preds, all_labels)]) / len(all_labels)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{cfg.epochs}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

        # Сохраняем лучшую модель
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"outputs/best_model_{cfg.model_name}.pt")
            print(f"Saved best model for {cfg.model_name} with val_acc={best_acc:.3f}")

    # Возвращаем историю
    return history

# Основной блок
if __name__ == "__main__":
    models_to_train = ["resnet18", "efficientnet_b0"]
    all_history = {}

    for m_name in models_to_train:
        print(f"\n=== Training {m_name} ===")
        cfg.model_name = m_name
        history = train_one_model(cfg)
        all_history[m_name] = history

    # Сохраняем истории в outputs
    torch.save(all_history, "outputs/training_history.pt")
    print("\nTraining complete. Histories saved to outputs/training_history.pt")
