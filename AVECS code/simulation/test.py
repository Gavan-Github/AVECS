import os
os.environ["WANDB_INSECURE"] = "true"
import wandb
import time
import random

"""
sudo apt-get update
sudo apt-get install --reinstall ca-certificates

"""
# 初始化 wandb 项目
wandb.init(project="wandb_test_project")

# 配置超参数
wandb.config.update({
    "learning_rate": 0.01,
    "batch_size": 32,
    "epochs": 10
})

# 模拟训练过程并记录指标
for epoch in range(wandb.config.epochs):
    loss = random.random() * 0.5  # 随机生成一个损失值
    accuracy = 1 - loss  # 假设准确率与损失成反比

    # 记录损失和准确率
    wandb.log({
        "epoch": epoch,
        "loss": loss,
        "accuracy": accuracy
    })

    # 模拟每个 epoch 的运行时间
    time.sleep(1)  # 等待 1 秒钟（模拟计算过程）

# 结束 wandb 记录
wandb.finish()
