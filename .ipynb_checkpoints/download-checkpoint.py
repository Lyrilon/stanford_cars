from datasets import load_dataset

# 加载数据集
ds = load_dataset("tanganke/stanford_cars")

# 保存到本地
ds.save_to_disk("/Users/zhaoyu/Desktop/stanford_cars")
