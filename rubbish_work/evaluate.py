#!/usr/bin/env python3
"""
File: evaluate.py
Description: 评估训练好的模型在测试集上的性能，计算常用的评价指标。
Author: [Your Name]
Date: [Date]
"""

import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import logging
from stanford_cars_dataset import StanfordCarsDataset, get_transforms
import config
from train import adjust_model
from datetime import datetime
from torchvision import models

# 设置日志
def setup_logger():
    # 创建logs目录
    os.makedirs('logs', exist_ok=True)
    
    # 生成日志文件名，包含时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/evaluation_{timestamp}.log'
    
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

def load_model(model, model_path):
    """加载预训练的模型"""
    try:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logging.info(f"成功加载模型: {model_path}")
    except Exception as e:
        logging.error(f"加载模型失败: {str(e)}")
        raise

def evaluate(model, device, test_loader, logger):
    """
    在测试集上评估模型性能
    """
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            try:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
            except Exception as e:
                logger.error(f"测试批次发生错误: {str(e)}")
                continue
    
    # 计算常用指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    logger.info(f'测试集准确率: {accuracy:.4f}')
    logger.info(f'精确率: {precision:.4f}')
    logger.info(f'召回率: {recall:.4f}')
    logger.info(f'F1得分: {f1:.4f}')
    
    return accuracy, precision, recall, f1

def main():
    # 设置日志
    logger = setup_logger()
    logger.info("开始评估模型...")
    
    device = config.DEVICE
    logger.info(f"使用设备: {device}")
    
    # 加载测试集
    test_transform = get_transforms(train=False)  # 测试集不需要数据增强
    test_dataset = StanfordCarsDataset(data_dir="standford_cars", split='test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    logger.info(f"测试集加载完成，总样本数: {len(test_dataset)}")
    
    # 加载预训练模型
    model = models.resnet50(pretrained=False)
    model = adjust_model(model, config.NUM_CLASSES)  # 根据Stanford Cars数据集调整输出层
    model = model.to(device)
    
    # 加载模型权重
    load_model(model, config.MODEL_SAVE_DIR+"/resnet50_gaussian.pth")
    
    # 评估模型性能
    evaluate(model, device, test_loader, logger)

if __name__ == "__main__":
    main()
