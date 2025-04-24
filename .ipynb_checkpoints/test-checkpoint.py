import torch
import torch.nn as nn
import clip
from torchvision import datasets, transforms
import config
from stanford_cars_dataset import get_transforms,StanfordCarsDataset
import os
import torch
from torch.utils.data import DataLoader
from PIL import Image
import shutil

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
        x = x.to(torch.float32)
        return self.classifier(x)


# 创建保存错误图片的目录
error_dir = "errors"
if not os.path.exists(error_dir):
    os.makedirs(error_dir)

# 运行测试
def test_model(model, test_loader, device):
    model.eval()  # 设置模型为评估模式
    all_correct = 0
    all_total = 0
    errors = []  # 用于存储错误的图片和标签

    with torch.no_grad():  # 禁用梯度计算，提高效率
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            # 推理
            outputs = model(images)

            # 获取预测标签
            _, predicted = torch.max(outputs, 1)

            # 统计正确率
            all_total += labels.size(0)
            all_correct += (predicted == labels).sum().item()

            # 判断哪些是错误的
            incorrect_indices = (predicted != labels).nonzero(as_tuple=True)[0]

            for idx in incorrect_indices:
                # 获取错误样本对应的图片和标签
                img = images[idx].cpu()
                label = labels[idx].cpu()
                predicted_label = predicted[idx].cpu()

                # 保存图片到错误文件夹
                img_path = os.path.join(error_dir, f"error_{batch_idx}_{idx.item()}.png")
                img_pil = transforms.ToPILImage()(img)
                img_pil.save(img_path)

                # 记录下错误的图片路径和标签
                errors.append({
                    "image_path": img_path,
                    "true_label": label.item(),
                    "predicted_label": predicted_label.item()
                })

    accuracy = 100 * all_correct / all_total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return errors



# Run the test
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. 加载 CLIP 模型（只用视觉部分）
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    image_encoder = clip_model.visual  # 视觉 backbone 
    model = CLIPClassifier(image_encoder, num_classes=config.NUM_CLASSES).to(device)

    # 加载保存的权重
    checkpoint = torch.load("accu0.7827384653650044 top5 0.9669195373709738Pretain-epoch_265.pth",map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    test_transform_pretrain = get_transforms(train = False)
    test_dataset = StanfordCarsDataset(data_dir="standford_cars", split='test', transform=test_transform_pretrain)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    # 测试模型并保存错误样本
    errors = test_model(model, test_loader, device)

    # 打印一些错误的例子
    print("Some incorrectly classified samples:")
    for error in errors[:5]:  # 仅展示前5个错误样本
        print(f"True: {error['true_label']}, Predicted: {error['predicted_label']}, Image Path: {error['image_path']}")