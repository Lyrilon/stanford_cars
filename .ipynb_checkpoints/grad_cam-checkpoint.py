#!/usr/bin/env python3
"""
File: grad_cam.py
Description: 实现基于Grad-CAM的注意力机制模块，
             用于定位输入图像中对分类最关键的区域。
Author: [Your Name]
Date: [Date]

Usage:
    # 示例：加载预训练模型（例如ResNet50），并对一张输入图像生成Grad-CAM热图
    python grad_cam.py --target_layer layer4 --image_path path/to/your/image.jpg
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import argparse
from datasets import load_from_disk
import os

class GradCAM:
    """
    Grad-CAM实现：基于反向传播梯度和卷积层特征，生成目标类别的注意力热图。
    
    参数:
        model: 预训练模型
        target_layer_name: 指定的卷积层名称，用于捕获激活和梯度
    """
    def __init__(self, model, target_layer_name):
        self.model = model.eval()
        self.target_layer_name = target_layer_name
        self.gradients = None  # 存储目标层梯度
        self.activations = None  # 存储目标层激活值
        self.hook_handles = []
        self._register_hooks()

    def _find_target_layer(self, module, target_layer_name):
        """
        递归查找目标层。
        """
        for name, mod in module.named_children():
            if name == target_layer_name:
                return mod
            else:
                found = self._find_target_layer(mod, target_layer_name)
                if found is not None:
                    return found
        return None

    def _register_hooks(self):
        """
        注册前向和后向hook，分别用于捕获激活值和梯度。
        """
        target_layer = self._find_target_layer(self.model, self.target_layer_name)
        if target_layer is None:
            raise ValueError(f"未找到目标层：{self.target_layer_name}")
        
        # 前向hook：捕获激活
        def forward_hook(module, input, output):
            self.activations = output.detach()
        handle_forward = target_layer.register_forward_hook(forward_hook)
        self.hook_handles.append(handle_forward)
        
        # 后向hook：捕获梯度
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        handle_backward = target_layer.register_backward_hook(backward_hook)
        self.hook_handles.append(handle_backward)

    def remove_hooks(self):
        """
        移除所有hook，释放资源。
        """
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, input_tensor, target_class=None):
        """
        对输入图像进行前向和反向传播，计算Grad-CAM热图。
        
        参数:
            input_tensor: 输入图像Tensor, shape [B, C, H, W]
            target_class: 指定目标类别，如果为None则取预测类别
        返回:
            grad_cam_map: Grad-CAM热图，归一化到[0,1]，shape [B, 1, H, W]
            output: 模型的输出结果
        """
        # 清空之前的梯度和激活
        self.gradients = None
        self.activations = None
        
        # 前向传播
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # 创建one-hot向量作为反向传播的目标
        one_hot = torch.zeros_like(output)
        for i in range(output.size(0)):
            one_hot[i, target_class[i]] = 1
        
        # 反向传播
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 计算权重：对梯度进行全局平均池化
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # 计算加权激活和
        grad_cam_map = torch.sum(weights * self.activations, dim=1)
        # 应用ReLU仅保留正值
        grad_cam_map = F.relu(grad_cam_map)
        # 上采样至与输入图像相同尺寸
        grad_cam_map = F.interpolate(grad_cam_map.unsqueeze(1), size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        # 归一化热图至[0,1]
        B, _, H, W = grad_cam_map.shape
        grad_cam_map = grad_cam_map.view(B, -1)
        grad_cam_map_min = grad_cam_map.min(dim=1, keepdim=True)[0]
        grad_cam_map_max = grad_cam_map.max(dim=1, keepdim=True)[0]
        grad_cam_map = (grad_cam_map - grad_cam_map_min) / (grad_cam_map_max - grad_cam_map_min + 1e-8)
        grad_cam_map = grad_cam_map.view(B, 1, H, W)
        
        return grad_cam_map, output

def preprocess_image(image_path, device):
    """
    对输入图像进行预处理，返回Tensor。
    """
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 使用PIL打开图像
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"无法打开图像 {image_path}: {e}")
        raise
    
    # 转换为张量
    image_tensor = preprocess(image).unsqueeze(0)  # shape: [1, C, H, W]
    return image_tensor.to(device)

def overlay_heatmap(image_path, heatmap, output_path):
    """
    将热图覆盖到原图上，并保存输出图像。
    """
    # 读取原图
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 调整大小
    img = cv2.resize(img, (heatmap.shape[2], heatmap.shape[3]))
    
    # 处理热图
    heatmap_np = heatmap.cpu().numpy()[0, 0]
    heatmap_np = cv2.applyColorMap(np.uint8(255 * heatmap_np), cv2.COLORMAP_JET)
    
    # 叠加
    overlayed = cv2.addWeighted(img, 0.5, heatmap_np, 0.5, 0)
    
    # 保存
    cv2.imwrite(output_path, overlayed)
    print(f"热图已保存到: {output_path}")

def extract_sample_image(dataset_path, output_path, split='train', index=0):
    """
    从数据集中提取一张样本图片用于测试。
    
    Args:
        dataset_path: 数据集路径
        output_path: 输出图片路径
        split: 数据集划分
        index: 要提取的图片索引
    """
    # 加载数据集
    dataset = load_from_disk(dataset_path)[split]
    
    # 获取样本
    sample = dataset[index]
    image = sample['image']
    
    # 确保图像是RGB格式
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:  # 如果是灰度图
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:  # 如果是RGBA
            image = image[..., :3]
        image = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
    
    # 保存图片
    image.save(output_path)
    print(f"样本图片已保存到: {output_path}")
    return output_path

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载预训练模型，这里以ResNet50为例
    print("正在加载预训练模型...")
    model = models.resnet50(pretrained=True).to(device)
    
    # 根据用户指定的目标层名称创建GradCAM实例
    print(f"创建GradCAM实例，目标层: {args.target_layer}")
    grad_cam = GradCAM(model, target_layer_name=args.target_layer)
    
    # 预处理图像
    print(f"正在处理图像: {args.image_path}")
    image_tensor = preprocess_image(args.image_path, device)
    
    # 计算Grad-CAM热图，target_class可选
    print("计算Grad-CAM热图...")
    grad_cam_map, output = grad_cam(image_tensor, target_class=None)
    
    # 获取预测类别
    pred_class = output.argmax(dim=1)[0].item()
    print(f"预测类别: {pred_class}")
    
    # 可选：保存热图覆盖图
    if args.output_path:
        overlay_heatmap(args.image_path, grad_cam_map, args.output_path)
    
    # 可视化热图（使用matplotlib显示）
    try:
        import matplotlib.pyplot as plt
        heatmap_np = grad_cam_map.cpu().numpy()[0, 0]
        plt.figure(figsize=(10, 5))
        
        # 显示原图
        plt.subplot(1, 2, 1)
        img = cv2.imread(args.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title("原始图像")
        plt.axis('off')
        
        # 显示热图
        plt.subplot(1, 2, 2)
        plt.imshow(heatmap_np, cmap='jet')
        plt.title("Grad-CAM 热图")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("matplotlib 未安装，无法显示热图。")
    
    # 移除hook，释放资源
    grad_cam.remove_hooks()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM Attention Module Implementation")
    parser.add_argument("--target_layer", type=str, default="layer4", help="目标卷积层的名称，例如 'layer4'")
    parser.add_argument("--image_path", type=str, default=None, help="输入图像路径")
    parser.add_argument("--output_path", type=str, default="grad_cam_output.jpg", help="输出覆盖热图的保存路径")
    parser.add_argument("--dataset_path", type=str, default="./standford_cars", help="数据集路径")
    args = parser.parse_args()
    
    # 如果没有提供图片路径，从数据集中提取一张
    if args.image_path is None:
        print("未提供图片路径，从数据集中提取样本图片...")
        args.image_path = extract_sample_image(args.dataset_path, "sample_image.jpg")
    
    main(args)
