# train_stage1.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import logging
import json
from datetime import datetime
import signal
import sys

import config
from stanford_cars_dataset import StanfordCarsDataset, get_transforms

PRETRAIN_EPOCHS = getattr(config, "PRETRAIN_EPOCHS", 1000)
PRETRAIN_MODEL_SAVE_PATH = os.path.join(config.MODEL_SAVE_DIR, f"{config.BACKBONE}_pretrain.pth")
TRAINING_STATE_PATH = os.path.join(config.MODEL_SAVE_DIR, "training_state.json")


def setup_logger():
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/pretrain_{timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def save_checkpoint(model, optimizer, epoch, best_val_acc, training_state):
    os.makedirs(os.path.dirname(PRETRAIN_MODEL_SAVE_PATH), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'training_state': training_state
    }, PRETRAIN_MODEL_SAVE_PATH)
    with open(TRAINING_STATE_PATH, 'w') as f:
        json.dump(training_state, f, indent=4)


def signal_handler(signum, frame):
    logging.info("中断信号，退出训练...")
    sys.exit(0)


def adjust_model(model, num_classes):
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_phase1(model, device, train_loader, criterion, optimizer, logger):
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            logger.info(f"[Pretrain] Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")


def validate(model, device, val_loader, criterion, logger):
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100. * correct / total
    logger.info(f"验证集损失: {val_loss / len(val_loader):.4f}, 准确率: {acc:.2f}%")
    return acc


def main():
    logger = setup_logger()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    device = config.DEVICE
    logger.info(f"使用设备: {device}")

    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    train_dataset = StanfordCarsDataset(data_dir="standford_cars", split='train', transform=train_transform)
    val_dataset = StanfordCarsDataset(data_dir="standford_cars", split='val', transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    model = models.resnet50(pretrained=True)
    model = adjust_model(model, config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_acc = 0.0
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        logger.info(f"=== Epoch {epoch}/{PRETRAIN_EPOCHS} ===")
        train_phase1(model, device, train_loader, criterion, optimizer, logger)
        if epoch % 2 == 0:
            acc = validate(model, device, val_loader, criterion, logger)
            if acc > best_val_acc:
                best_val_acc = acc
                save_checkpoint(model, optimizer, epoch, best_val_acc, {'phase': 2, 'epoch': epoch})
                logger.info(f"[最佳] Epoch {epoch} 验证准确率: {acc:.2f}%")

if __name__ == '__main__':
    main()