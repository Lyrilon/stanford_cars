import torch
import torch.nn as nn
import clip
from torchvision import datasets, transforms
import config

# 1. 加载 CLIP 模型（只用视觉部分）
clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
image_encoder = clip_model.visual  # 视觉 backbone

# 2. 分类模型定义：CLIP视觉模块 + 分类头
class CLIPClassifier(nn.Module):
    def __init__(self, image_encoder, num_classes):
        super().__init__()
        self.encoder = image_encoder
        self.classifier = nn.Sequential(
            nn.Linear(image_encoder.proj.shape[1], 512),  # 512是CLIP的输出维度
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():  # 可选：不训练 encoder
            x = self.encoder(x)
        return self.classifier(x)

model = CLIPClassifier(image_encoder, num_classes=config.NUM_CLASSES).to("cuda")

#查看你encoder输入维度
print(model.encoder.input_resolution)  # 输入分辨率

a_forward = model(torch.randn(3, 224, 224).to("cuda"))
print(a_forward.shape)  # 输出形状


