#!/usr/bin/env python3
"""
File: stanford_cars_dataset.py
Description: PyTorch Dataset and DataLoader implementation for Stanford Cars dataset.
Author: [Your Name]
Date: [Date]

Usage:
    python stanford_cars_dataset.py --batch_size 32
"""

import os
# 在导入其他模块之前禁用accimage
os.environ['TORCHVISION_USE_ACCIMAGE'] = '0'
os.environ['ACCIMAGE_DISABLE'] = '1'

# 强制使用PIL后端
import torchvision
torchvision.set_image_backend('PIL')

import logging
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import numpy as np
from pathlib import Path
from datasets import load_from_disk
import random
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class StanfordCarsDataset(Dataset):
    """
    Stanford Cars Dataset.
    
    Args:
        data_dir (str): 数据集根目录路径
        split (str): 数据集划分 ('train', 'val' 或 'test')
        transform (callable, optional): 数据转换函数
        val_ratio (float): 从训练集中划分验证集的比例，默认为0.2
    """
    def __init__(self, data_dir, split='train', transform=None):
        # 加载数据集
        dataset = load_from_disk(data_dir)
        if split == 'train':
            self.dataset = dataset['train']
        elif split == 'val':
            self.dataset = dataset['val']
        else:
            self.dataset = dataset[split]
        self.transform = transform

        # 提取所有标签并保存为 targets 属性
        self.targets = [sample['label'] for sample in self.dataset]


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            sample = self.dataset[idx]
            image = sample['image']
            label = sample['label']
            
            # 确保图像数据是正确的格式
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                raise ValueError(f"不支持的图像类型: {type(image)}")
            
            # 确保图像是RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            if self.transform:
                try:
                    image = self.transform(image)
                except Exception as e:
                    logging.error(f"图像转换失败 (索引 {idx}): {str(e)}")
                    raise
            return image, label
        
        except Exception as e:
            logging.error(f"加载索引 {idx} 的图像时发生错误: {str(e)}")
            # 返回一个默认图像和标签
            return self._get_default_item()

    def _get_default_item(self):
        """返回一个默认的图像和标签"""
        # 创建一个空的RGB图像
        image = Image.new('RGB', (224, 224), color='black')
        if self.transform:
            image = self.transform(image)
        return image, 0

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    

def main(args):
    try:
        # 创建数据转换
        transform = get_transforms(train=args.train)
        
        # 创建数据集实例
        dataset = StanfordCarsDataset(
            data_dir=args.data_dir,
            split='train' if args.train else 'test',
            transform=transform
        )
        
        # 创建 DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # 测试迭代
        for batch_idx, (images, labels) in enumerate(dataloader):
            logging.info(f"Batch {batch_idx}: images.shape={images.shape}, labels.shape={labels.shape}")
            if batch_idx >= 2:
                break
                
    except Exception as e:
        logging.error(f"运行过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stanford Cars Dataset Loader")
    parser.add_argument("--data_dir", type=str, default="./standford_cars", help="数据集根目录路径")
    parser.add_argument("--batch_size", type=int, default=32, help="DataLoader的批次大小")
    parser.add_argument("--shuffle", action="store_true", help="是否打乱数据集")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader的工作进程数")
    parser.add_argument("--train", action="store_true", help="是否使用训练集（否则使用测试集）")
    args = parser.parse_args()
    
    # 检查数据集目录是否存在
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"数据集目录 {args.data_dir} 不存在，请确保路径正确")
    
    main(args)
