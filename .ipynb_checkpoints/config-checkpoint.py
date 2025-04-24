# config.py
import os
import torch

# 数据集路径配置
TRAIN_CSV = "data/StanfordCars/train_labels.csv"
TRAIN_IMG_DIR = "data/StanfordCars/train"
VAL_CSV = "data/StanfordCars/val_labels.csv"  # 如果没有独立验证集，可以自行从训练集中划分
VAL_IMG_DIR = "data/StanfordCars/val"

# 训练超参数
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
AUGMENTATION_PROB = 0.5  # 每个batch中应用关键部件Mix增强的概率
THRESHOLD_FACTOR = 1.0  # 用于关键部件裁剪时热图的阈值因子

# 模型参数
NUM_CLASSES = 196  # Stanford Cars有196个类别
BACKBONE = "resnet50"

# 模型保存路径
MODEL_SAVE_DIR = "checkpoints"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f"{BACKBONE}_keypart_mix.pth")

# 训练日志设置
LOG_INTERVAL = 10

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
