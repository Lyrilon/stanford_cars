import torch
from train import adjust_model
from torchvision import models
from config import config
import logging
import os

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
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            except Exception as e:
                logger.error(f"验证批次发生错误: {str(e)}")
                continue
    
    accuracy = 100. * correct / total
    avg_loss = val_loss / len(val_loader)
    logger.info(f'验证集损失: {avg_loss:.4f}, 准确率: {accuracy:.2f}%')
    return accuracy, avg_loss

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("开始加载模型")
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    model = adjust_model(model, config.NUM_CLASSES)
    model = model.to(device)
    logger.info("模型加载完成")
    #加载权重
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("模型权重加载完成")
    else:
        logger.error(f"未找到检查点文件: {checkpoint_path}")
        exit(1)
    # 加载验证集
    

    
