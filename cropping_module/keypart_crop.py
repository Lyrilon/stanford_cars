#!/usr/bin/env python3
"""
File: keypart_mix.py
Description: 实现关键部件引导的Mix增强策略。
             从一个batch中随机选取不同类别的图像作为Mix源图像，
             利用Grad-CAM得到目标图像的关键部件区域（bounding box），
             并将源图像对应区域裁剪后替换到目标图像中，
             最终输出增强后的图像和原始标签（标签保持原图）。
Author: [Your Name]
Date: [Date]
"""

import torch
import random
import numpy as np

# 假设我们已经实现了get_keypart_bbox函数在 cropping_module/keypart_crop.py 中
try:
    from cropping_module.keypart_crop import get_keypart_bbox
except ImportError:
    # 如果找不到，则提供一个简单的实现（与之前代码一致）
    def get_keypart_bbox(heatmap, threshold_factor=1.0):
        """
        根据热图（numpy数组，形状 [H, W]，值范围[0,1]）采用阈值法得到关键区域的边界框。
        阈值 = mean + threshold_factor * std
        返回 (x_min, y_min, x_max, y_max)，若无满足条件区域则返回 None。
        """
        mean_val = np.mean(heatmap)
        std_val = np.std(heatmap)
        threshold = mean_val + threshold_factor * std_val
        mask = heatmap > threshold
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return (int(x_min), int(y_min), int(x_max), int(y_max))

def keypart_guided_mix(img_batch, labels, grad_cam, threshold_factor=1.0):
    """
    对一批图像进行关键部件引导的Mix增强。
    
    参数：
        img_batch (torch.Tensor): 输入图像Tensor, shape [B, C, H, W]
        labels (torch.Tensor or list): 对应的标签，形状 [B]
        grad_cam: 已构造的GradCAM实例，用于计算注意力热图
        threshold_factor (float): 阈值因子，默认1.0，即阈值 = mean + std
        
    返回：
        mixed_images (torch.Tensor): 增强后的图像Tensor, shape [B, C, H, W]
        mixed_labels (torch.Tensor): 增强后的标签，与原始标签一致
    """
    # 复制输入，准备输出
    mixed_images = img_batch.clone()
    # 保证labels为Tensor
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)
    mixed_labels = labels.clone()
    
    B, C, H, W = img_batch.size()
    
    # 创建一个随机排列，保证每个样本有可能选择到不同类别的源样本
    perm_indices = torch.randperm(B)
    
    # 遍历batch中的每个图像
    for i in range(B):
        target_label = labels[i].item()
        source_idx = perm_indices[i].item()
        # 确保选取的源样本与目标样本不同类别
        if labels[source_idx].item() == target_label:
            # 如果随机选到同类别，则尝试在剩余样本中随机选择一个不同类别的源图像
            candidates = [j for j in range(B) if labels[j].item() != target_label]
            if len(candidates) == 0:
                continue  # 若无不同类别样本，则跳过增强
            source_idx = random.choice(candidates)
        
        target_img = img_batch[i].unsqueeze(0)  # [1, C, H, W]
        source_img = img_batch[source_idx]        # [C, H, W]
        
        # 计算目标图像的注意力热图
        # 这里调用grad_cam，注意返回的热图形状为 [1, 1, H, W]
        heatmap, _ = grad_cam(target_img)
        # 将热图转换为numpy数组，并取出二维热图 [H, W]
        heatmap_np = heatmap.cpu().numpy()[0, 0]
        
        # 获取关键部件区域边界框
        bbox = get_keypart_bbox(heatmap_np, threshold_factor=threshold_factor)
        if bbox is None:
            # 若未找到有效区域，则跳过该图像的增强
            continue
        
        x_min, y_min, x_max, y_max = bbox
        # 检查边界框尺寸是否合理（至少有1像素）
        if x_max <= x_min or y_max <= y_min:
            continue
        
        # 执行区域替换：
        # 将目标图像中[b, :, y_min:y_max, x_min:x_max]替换为源图像同一位置的区域
        mixed_images[i, :, y_min:y_max, x_min:x_max] = source_img[:, y_min:y_max, x_min:x_max]
        # 更新标签为源图像的标签
        mixed_labels[i] = labels[source_idx]
        
    return mixed_images, mixed_labels

# 示例测试代码（可独立运行）
if __name__ == "__main__":
    import argparse
    from torchvision import models, transforms
    from PIL import Image
    import matplotlib.pyplot as plt
    # 假设我们使用预训练ResNet50，并使用其layer4作为目标层
    from grad_cam import GradCAM  # 确保grad_cam.py在路径中

    parser = argparse.ArgumentParser(description="Key-Part Guided Mix Augmentation Module")
    parser.add_argument("--image_path1", type=str, required=True, help="目标图像路径")
    parser.add_argument("--image_path2", type=str, required=True, help="源图像路径（不同类别）")
    parser.add_argument("--threshold_factor", type=float, default=1.0, help="阈值因子")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载预训练模型ResNet50
    model = models.resnet50(pretrained=True).to(device)
    # 创建GradCAM实例，选择layer4作为目标层
    grad_cam = GradCAM(model, target_layer_name="layer4")
    
    # 定义图像预处理
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 加载两张测试图像，并转换为Tensor
    img1 = Image.open(args.image_path1).convert("RGB")
    img2 = Image.open(args.image_path2).convert("RGB")
    tensor1 = preprocess(img1).to(device)
    tensor2 = preprocess(img2).to(device)
    
    # 创建一个batch: 假设batch size为2，图像1和图像2分别对应不同类别
    batch = torch.stack([tensor1, tensor2], dim=0)  # [2, C, H, W]
    labels = torch.tensor([0, 1])  # 假设图像1类别为0, 图像2类别为1

    # 调用关键部件引导的Mix增强函数
    mixed_batch, mixed_labels = keypart_guided_mix(batch, labels, grad_cam, threshold_factor=args.threshold_factor)
    
    # 显示结果（将Tensor转回PIL图像，去归一化处理）
    def tensor_to_pil(tensor):
        # tensor shape: [C, H, W]
        # 反归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(tensor.device)
        tensor = tensor * std + mean
        tensor = tensor.clamp(0,1)
        np_img = tensor.cpu().numpy().transpose(1,2,0) * 255
        return Image.fromarray(np.uint8(np_img))
    
    mixed_img1 = tensor_to_pil(mixed_batch[0])
    mixed_img2 = tensor_to_pil(mixed_batch[1])
    
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(mixed_img1)
    plt.title(f"Augmented Image 1, Label {mixed_labels[0].item()}")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(mixed_img2)
    plt.title(f"Augmented Image 2, Label {mixed_labels[1].item()}")
    plt.axis("off")
    plt.show()
    
    # 移除grad_cam hook
    grad_cam.remove_hooks()
