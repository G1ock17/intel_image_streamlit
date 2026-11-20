import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
from torchvision.models import resnet50, ResNet50_Weights

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

def find_split(root: Path, split: str):
    candidates = [root / split, root / split / split]
    for path in candidates:
        if path.is_dir() and all((path / c).is_dir() for c in CLASS_NAMES):
            return path
    raise FileNotFoundError(
        f"Не найдена папка {split} с подпапками классов. "
        f"Ожидается {root}/{split}/<class> или {root}/{split}/{split}/<class>."
    )

def main(args):
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data_dir)
    train_dir = find_split(root, "seg_train")
    test_dir = find_split(root, "seg_test")

    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    if train_dataset.classes != test_dataset.classes:
        raise ValueError("Наборы классов train и test не совпадают.")
    class_names = train_dataset.classes
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=(device.type == "cuda"))

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    history = {"train_losses": [], "train_accs": [], "valid_losses": [], "valid_accs": []}
    for epoch in range(args.epochs):
        # Замороженная backbone остаётся в eval, чтобы BatchNorm не обновлялся.
        model.eval()
        model.fc.train()
        train_loss = train_correct = train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        valid_loss = valid_correct = valid_total = 0
        with torch.inference_mode():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                valid_loss += loss.item() * labels.size(0)
                valid_correct += (logits.argmax(1) == labels).sum().item()
                valid_total += labels.size(0)

        history["train_losses"].append(train_loss / train_total)
        history["train_accs"].append(train_correct / train_total)
        history["valid_losses"].append(valid_loss / valid_total)
        history["valid_accs"].append(valid_correct / valid_total)
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {history['train_losses'][-1]:.4f}, "
              f"Train Acc: {history['train_accs'][-1]:.4f} | "
              f"Test Loss: {history['valid_losses'][-1]:.4f}, "
              f"Test Acc: {history['valid_accs'][-1]:.4f}")

    y_true, y_pred = [], []
    model.eval()
    with torch.inference_mode():
        for images, labels in test_loader:
            logits = model(images.to(device))
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            y_true.extend(labels.numpy().tolist())

    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    elapsed = time.perf_counter() - started

    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    torch.save(model.cpu().state_dict(), out / "resnet50_intel.pth")
    metadata = {
        "classes": class_names,
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "class_counts_train": {c: int(sum(1 for _, y in train_dataset.samples if y == i))
                               for i, c in enumerate(class_names)},
        "class_counts_test": {c: int(sum(1 for _, y in test_dataset.samples if y == i))
                              for i, c in enumerate(class_names)},
        "history": history,
        "f1_macro": float(f1),
        "confusion_matrix": cm.tolist(),
        "training_time_seconds": elapsed,
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        ),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device": str(device),
    }
    (out / "metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nMacro F1: {f1:.4f}")
    print(f"Время обучения: {elapsed:.1f} сек.")
    print("Артефакты сохранены в папке artifacts/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    main(parser.parse_args())
