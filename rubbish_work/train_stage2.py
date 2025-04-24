import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import logging
import json
from datetime import datetime

import config
from stanford_cars_dataset import StanfordCarsDataset, get_transforms
from grad_cam import GradCAM
from augmentation_module.keypart_mix import keypart_guided_mix
from train_stage1 import adjust_model, validate, save_checkpoint, load_checkpoint, setup_logger

# 日志初始化
logger = setup_logger("finetune")
logger.info("===== 阶段2: 微调训练开始 =====")

# 设备
device = config.DEVICE

# 数据
train_transform = get_transforms(train=True)
val_transform = get_transforms(train=False)
train_dataset = StanfordCarsDataset(data_dir="standford_cars", split='train', transform=train_transform)
val_dataset = StanfordCarsDataset(data_dir="standford_cars", split='val', transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
logger.info(f"训练样本数: {len(train_dataset)}，验证样本数: {len(val_dataset)}")

# 模型
model = models.resnet50(pretrained=True)
model = adjust_model(model, config.NUM_CLASSES)
model = model.to(device)

# 优化器
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE / 10)
criterion = nn.CrossEntropyLoss()

# 加载预训练阶段的模型
start_epoch, best_val_acc, training_state = load_checkpoint(model, optimizer, os.path.join(config.MODEL_SAVE_DIR, f"{config.BACKBONE}_pretrain.pth"), logger)
assert training_state['phase'] == 2, "当前模型未完成阶段1训练。"

# Grad-CAM 实例
grad_cam = GradCAM(model, target_layer_name="layer4")

# 训练微调
for epoch in range(start_epoch, config.FINETUNE_EPOCHS + 1):
    logger.info(f"微调 Epoch {epoch}/{config.FINETUNE_EPOCHS}")
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        if torch.rand(1).item() < config.AUGMENTATION_PROB:
            images, labels = keypart_guided_mix(images, labels, grad_cam, threshold_factor=config.THRESHOLD_FACTOR)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        running_loss += loss.item()

    acc = 100. * correct / total
    logger.info(f"Epoch {epoch} Train Acc: {acc:.2f}%  Loss: {running_loss / len(train_loader):.4f}")

    # 验证
    val_acc, val_loss = validate(model, device, val_loader, criterion, logger)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        training_state['epoch'] = epoch
        save_path = os.path.join(config.MODEL_SAVE_DIR, f"{config.BACKBONE}_finetune_best.pth")
        save_checkpoint(model, optimizer, epoch, best_val_acc, training_state, save_path)

# 保存最终模型
torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
grad_cam.remove_hooks()
logger.info("===== 阶段2: 微调训练完成 =====")