import clip
from PIL import Image
import torch
from stanford_cars_dataset import StanfordCarsDataset, get_transforms
import config
import torch.nn.functional as F
import torch.optim as optim

# 获取图像和文本特征
def get_features_from_batch(images, texts, model):
    images = images.to(device)
    texts = clip.tokenize(texts).to(device)

    # 获取图像特征
    image_features = model.encode_image(images)

    # 获取文本特征
    text_features = model.encode_text(texts)

    return image_features, text_features

# 对比损失函数（InfoNCE）
def clip_contrastive_loss(image_features, text_features, temperature=0.07):
    # 对图像和文本特征进行归一化
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    # 计算相似度矩阵（batch_size × batch_size）
    logits = image_features @ text_features.T / temperature
    
    # 对角线是正样本（img[i] 对应 text[i]）
    labels = torch.arange(logits.shape[0]).to(logits.device)
    
    # 计算交叉熵损失（双向对比）
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    
    return (loss_i2t + loss_t2i) / 2


# 训练过程
def train(model, train_loader, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            # 假设 labels 是一个包含文本标签的字典，如果不是，需要调整这部分

            # 获取图像和文本特征
            image_features, text_features = get_features_from_batch(images, labels, model)
            
            # 计算对比损失
            loss = clip_contrastive_loss(image_features, text_features)

            # 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算损失
            running_loss += loss.item()

            if batch_idx % 100 == 0:  # 每100个批次打印一次
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {running_loss / (batch_idx+1):.4f}")

        # 打印每个epoch的平均损失
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # 每隔一些epoch保存模型
        torch.save(model.state_dict(), f'loss{loss}clip_finetuned_epoch_{epoch+1}.pth')

# 主程序
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)

    # 检查CUDA是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载 CLIP 模型（视觉和文本部分）
    model, preprocess = clip.load("ViT-B/32", device=device)


    # 优化器，针对 CLIP 模型所有参数进行优化
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    # 数据集和数据加载器
    train_transform_pretrain = get_transforms(train=True)
    train_dataset = StanfordCarsDataset(data_dir="standford_cars", split='train', transform=train_transform_pretrain)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=4)

    # 训练模型
    train(model, train_loader, optimizer, num_epochs=1000)
