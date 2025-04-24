#!/usr/bin/env python3
"""
File: train.py
Description: 训练脚本，分为两个阶段：
    阶段1（基础预训练阶段）：在原始数据集上训练模型，使模型具备基本分类能力与稳定的注意力机制。
    阶段2（混合增强微调阶段）：在预训练模型基础上，利用关键部件引导的Mix增强策略进行微调，
         强制模型在关键部件替换后仍能正确分类，从而提升细粒度识别性能。
Author: [Your Name]
Date: [Date]
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import logging
from datetime import datetime
import json
import signal
import sys

import config
from stanford_cars_dataset import StanfordCarsDataset, get_transforms
from grad_cam import GradCAM
from augmentation_module.keypart_mix import keypart_guided_mix

# 设置日志
def setup_logger():
    # 创建logs目录
    os.makedirs('logs', exist_ok=True)
    
    # 生成日志文件名，包含时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training_{timestamp}.log'
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# 为两个阶段分别设定训练轮数
# getattr的作用是获取config中的属性值，如果没有则使用默认值
PRETRAIN_EPOCHS = getattr(config, "PRETRAIN_EPOCHS", 0)
FINETUNE_EPOCHS = getattr(config, "FINETUNE_EPOCHS", 1000)
GUASSAIN_EPOCHS = getattr(config, "GUASSAIN_EPOCHS", 0)
PRETRAIN_MODEL_SAVE_PATH = os.path.join(config.MODEL_SAVE_DIR, f"{config.BACKBONE}_pretrain.pth")
Guassain_MODEL_SAVE_PATH = os.path.join(config.MODEL_SAVE_DIR, f"{config.BACKBONE}_gaussian.pth")
TRAINING_STATE_PATH = os.path.join(config.MODEL_SAVE_DIR, "training_state.json")
FiNETUNE_MODEL_SAVE_PATH = os.path.join(config.MODEL_SAVE_DIR, f"{config.BACKBONE}_finetune.pth")

def save_checkpoint(model, optimizer, epoch, best_val_acc, training_state, save_path):
    """保存训练状态和模型"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存模型状态
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'training_state': training_state
        }, save_path)
        
        # 保存训练状态到JSON文件
        with open(TRAINING_STATE_PATH, 'w') as f:
            json.dump(training_state, f, indent=4)
            
        return True
    except Exception as e:
        logging.error(f"保存checkpoint失败: {str(e)}")
        return False

def load_checkpoint(model, optimizer, save_path, logger):
    """加载训练状态和模型"""
    try:
        if os.path.exists(save_path):
            checkpoint = torch.load(save_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['best_val_acc']
            training_state = checkpoint['training_state']
            logger.info(f"从epoch {start_epoch} 继续训练")
            logger.info(f"最佳验证准确率: {best_val_acc:.2f}%")
            return start_epoch, best_val_acc, training_state
        return 1, 0.0, {'phase': 1, 'epoch': 0}
    except Exception as e:
        logger.error(f"加载checkpoint失败: {str(e)}")
        return 1, 0.0, {'phase': 1, 'epoch': 0}

def signal_handler(signum, frame):
    """处理训练中断信号"""
    logging.info("收到中断信号，正在保存训练状态...")
    sys.exit(0)

def adjust_model(model, num_classes):
    """
    调整预训练模型的最后一层，使其适应Stanford Cars的类别数（196类）。
    这里以ResNet50为例。
    """
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def train_phase1(model, device, train_loader, criterion, optimizer, logger):
    """
    阶段1：基础预训练阶段，不采用关键部件Mix增强，
    仅在原始数据上训练，目标是让模型学会基本的分类和注意力定位能力。
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        try:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            running_loss += loss.item()
            if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                avg_loss = running_loss / config.LOG_INTERVAL
                accuracy = 100. * correct / total
                logger.info(f"[Pretrain] Batch {batch_idx+1}/{len(train_loader)}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
                running_loss = 0.0
                correct = 0
                total = 0
        except Exception as e:
            logger.error(f"训练批次 {batch_idx} 发生错误: {str(e)}")
            continue
    return

def train_phase2(model, device, train_loader, criterion, optimizer, grad_cam, augmentation_prob, threshold_factor, logger):
    """
    阶段2：混合增强微调阶段。
    在这一阶段，利用预训练好的模型生成的注意力热图，
    对每个batch以一定概率应用关键部件引导的Mix增强，
    强化模型对局部细节变化的鲁棒性。
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        try:
            images, labels = images.to(device), labels.to(device)
            
            # 根据概率决定是否使用Mix增强
            if random.random() < augmentation_prob:
                # 注意：此时模型已经预训练完成，GradCAM生成的注意力热图较为稳定
                images, labels = keypart_guided_mix(images, labels, grad_cam, threshold_factor=threshold_factor)
                logger.debug(f"Applied Mix augmentation to batch {batch_idx}")
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            running_loss += loss.item()
            if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                avg_loss = running_loss / config.LOG_INTERVAL
                accuracy = 100. * correct / total
                logger.info(f"[Finetune] Batch {batch_idx+1}/{len(train_loader)}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
                running_loss = 0.0
                correct = 0
                total = 0
        except Exception as e:
            logger.error(f"训练批次 {batch_idx} 发生错误: {str(e)}")
            continue
    return

def train_on_gaussian_noise(model, device, train_loader, criterion, optimizer, logger):
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        try:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            running_loss += loss.item()
            if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                avg_loss = running_loss / config.LOG_INTERVAL
                accuracy = 100. * correct / total
                logger.info(f"[Gaussian Noise] Batch {batch_idx+1}/{len(train_loader)}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
                running_loss = 0.0
                correct = 0
                total = 0
        except Exception as e:
            logger.error(f"训练批次 {batch_idx} 发生错误: {str(e)}")
            continue


def validate(model, device, val_loader, criterion, logger):
    """
    在验证集上评估模型性能
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            try:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                #对于错误的预测，打印其logit和错判logit
                logger.debug(f"Logits: {outputs}, Predicted: {predicted}, Actual: {labels}")

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            except Exception as e:
                logger.error(f"验证批次发生错误: {str(e)}")
                continue
    
    accuracy = 100. * correct / total
    avg_loss = val_loss / len(val_loader)
    logger.info(f'验证集损失: {avg_loss:.4f}, 准确率: {accuracy:.2f}%')
    return accuracy, avg_loss

def main():
    # 设置日志
    logger = setup_logger()
    logger.info("开始训练...")
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建保存模型的文件夹
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    device = config.DEVICE
    logger.info(f"使用设备: {device}")
    
    # 数据加载：使用训练数据增强（仅基础数据增强用于预训练）
    train_transform_pretrain = get_transforms(train=True)
    val_transform = get_transforms(train=False)  # 验证集不需要数据增强
    
    # 使用新的验证集划分方式
    train_dataset = StanfordCarsDataset(data_dir="standford_cars", split='train', transform=train_transform_pretrain)
    val_dataset = StanfordCarsDataset(data_dir="standford_cars", split='test', transform=val_transform)
    guassian_train_dataset = StanfordCarsDataset(data_dir="standford_cars", split='gaussian_noise', transform=train_transform_pretrain)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    guassian_train_loader = DataLoader(guassian_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    
    logger.info(f"训练集加载完成，总样本数: {len(train_dataset)}")
    logger.info(f"验证集加载完成，总样本数: {len(val_dataset)}")
    logger.info(f"高斯噪声训练集加载完成，总样本数: {len(guassian_train_dataset)}")
    


    # 加载预训练ResNet50模型，并调整最后一层
    model = models.resnet50(pretrained=True)
    model = adjust_model(model, config.NUM_CLASSES)
    model = model.to(device)
    logger.info("模型加载完成")
    
    model = models.resnet152 


    # 定义损失函数和优化器（阶段1和阶段2均使用）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    logger.info(f"使用优化器: Adam, 学习率: {config.LEARNING_RATE}")
    
    # 加载checkpoint（如果存在）
    #start_epoch, best_val_acc, training_state = load_checkpoint(model, optimizer, PRETRAIN_MODEL_SAVE_PATH, logger)
    start_epoch = 1
    # ===== 阶段1：基础预训练 =====
    if training_state['phase'] == 1:
        logger.info("========== Stage 1: Pretraining on Original Data ==========")
        for epoch in range(start_epoch, PRETRAIN_EPOCHS + 1):
            try:
                logger.info(f"预训练 Epoch {epoch}/{PRETRAIN_EPOCHS}")
                train_phase1(model, device, train_loader, criterion, optimizer, logger)
                
                # 每2个epoch进行一次验证
                if epoch % 5 == 0:
                    logger.info(f"\n=== 验证 Epoch {epoch} ===")
                    val_acc, val_loss = validate(model, device, val_loader, criterion, logger)
                    
                    # 保存最佳模型
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        training_state['epoch'] = epoch
                        save_checkpoint(model, optimizer, epoch, best_val_acc, training_state, PRETRAIN_MODEL_SAVE_PATH)
                        logger.info(f"保存最佳模型，验证准确率: {best_val_acc:.2f}%")
                
                    
            except Exception as e:
                logger.error(f"Epoch {epoch} 训练过程中发生错误: {str(e)}")
                # 保存当前状态
                training_state['epoch'] = epoch
                save_checkpoint(model, optimizer, epoch, best_val_acc, training_state, PRETRAIN_MODEL_SAVE_PATH)
                raise
        
        logger.info(f"预训练完成，最佳验证准确率: {best_val_acc:.2f}%")
        training_state['phase'] = 2
        save_checkpoint(model, optimizer, PRETRAIN_EPOCHS, best_val_acc, training_state, PRETRAIN_MODEL_SAVE_PATH)

    # ===== 阶段2：混合增强微调 =====
    if training_state['phase'] == 2:
        logger.info("========== Stage 2: Finetuning with Key-Part Guided Mix Augmentation ==========")
        # 创建GradCAM实例，目标层选择 "layer4"
        grad_cam = GradCAM(model, target_layer_name="layer4")
        # 可以选择重新初始化优化器或调整学习率
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE / 10)  # 通常微调时用较小的学习率
        logger.info(f"微调阶段使用较小的学习率: {config.LEARNING_RATE / 10}")
        
        for epoch in range(start_epoch, FINETUNE_EPOCHS + 1):
            try:
                logger.info(f"Finetune Epoch {epoch}/{FINETUNE_EPOCHS}")
                train_phase2(model, device, train_loader, criterion, optimizer, grad_cam, 
                            augmentation_prob=config.AUGMENTATION_PROB, 
                            threshold_factor=config.THRESHOLD_FACTOR,
                            logger=logger)
                
                if epoch % 5 == 0:
                    logger.info(f"\n=== 验证 Epoch {epoch} ===")
                    val_acc, val_loss = validate(model, device, val_loader, criterion, logger)
                    # 保存最佳模型
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        training_state['epoch'] = epoch
                        save_checkpoint(model, optimizer, epoch, best_val_acc, training_state, FiNETUNE_MODEL_SAVE_PATH)
                        logger.info(f"保存最佳模型，验证准确率: {best_val_acc:.2f}%")
                    
                
            except Exception as e:
                logger.error(f"Epoch {epoch} 训练过程中发生错误: {str(e)}")
                # 保存当前状态
                training_state['epoch'] = epoch
                save_checkpoint(model, optimizer, epoch, best_val_acc, training_state, FineTUNE_MODEL_SAVE_PATH)
                raise
        
        # 保存最终模型
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        logger.info(f"Final model saved at: {config.MODEL_SAVE_PATH}")
        
        # 清理GradCAM hook
        grad_cam.remove_hooks()
        logger.info("Training complete.")


if __name__ == "__main__":
    main()
