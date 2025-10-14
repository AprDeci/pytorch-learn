import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ----------------------------
# 1. 配置参数（集中管理！）
# ----------------------------
class Config:
    input_dim = 1
    hidden_dim = 64
    output_dim = 1
    num_epochs = 1000
    lr = 0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


config = Config()


# ----------------------------
# 2. 数据生成函数（可复用）
# ----------------------------
def generate_data(func='sin', n_samples=200, noise=0.05):
    x = torch.linspace(-2 * 3.1416, 2 * 3.1416, n_samples).reshape(-1, 1)
    if func == 'sin':
        y_clean = torch.sin(x)
    elif func == 'xsinx':
        y_clean = x * torch.sin(x)
    elif func == 'abs':
        y_clean = torch.abs(x)
    else:
        raise ValueError("Unsupported function")

    y_noisy = y_clean + noise * torch.randn_like(y_clean)
    return x.to(config.device), y_noisy.to(config.device), y_clean.to(config.device)


# ----------------------------
# 3. 模型定义
# ----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# 4. 训练函数（核心！）
# ----------------------------
def train_model(model, x_train, y_train, config):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    losses = []
    for epoch in range(config.num_epochs):
        # 前向
        model.train()  # 设置为训练模式（虽本例无影响，但好习惯）
        pred = model(x_train)
        loss = criterion(pred, y_train)

        # 反向 + 更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch + 1}/{config.num_epochs}], Loss: {loss.item():.6f}")

    return losses


# ----------------------------
# 5. 评估与可视化
# ----------------------------
def evaluate_and_plot(model, x_train, y_train, y_clean, losses, config):
    model.eval()
    with torch.no_grad():
        x_test = torch.linspace(-2 * 3.1416, 2 * 3.1416, 300).reshape(-1, 1).to(config.device)
        y_pred = model(x_test)
        y_true = torch.sin(x_test)  # 或根据 func 调整

    # 创建子图
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：拟合效果
    axs[0].scatter(x_train.cpu(), y_train.cpu(), s=10, alpha=0.6, label='Noisy data')
    axs[0].plot(x_test.cpu(), y_true.cpu(), 'r--', label='True function', linewidth=2)
    axs[0].plot(x_test.cpu(), y_pred.cpu(), 'b', label='Neural Net', linewidth=2)
    axs[0].legend()
    axs[0].set_title('Function Fitting')

    # 右图：loss 曲线
    axs[1].plot(losses)
    axs[1].set_title('Training Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')

    plt.tight_layout()
    plt.show()


# ----------------------------
# 6. 主流程
# ----------------------------
if __name__ == "__main__":
    # 生成数据
    x_train, y_train, y_clean = generate_data(func='sin')

    # 创建模型
    model = MLP(config.input_dim, config.hidden_dim, config.output_dim).to(config.device)

    # 训练
    losses = train_model(model, x_train, y_train, config)

    # 评估
    evaluate_and_plot(model, x_train, y_train, y_clean, losses, config)

    # 保存模型（重要！）
    torch.save(model.state_dict(), 'sin_model.pth')
    print("✅ 模型已保存为 'sin_model.pth'")