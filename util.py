import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset


# 生成随机NGSIM数据
def generate_dummy_ngsim_data(file_path, num_samples=1000):
    np.random.seed(42)
    speed = np.random.uniform(0, 30, num_samples)  # 速度（0-30 m/s）
    acceleration = np.random.uniform(-3, 3, num_samples)  # 加速度（-3~3 m/s²）
    steering_angle = np.random.uniform(-30, 30, num_samples)  # 方向盘角度（-30°~30°）

    df = pd.DataFrame({
        'speed': speed,
        'acceleration': acceleration,
        'steering_angle': steering_angle
    })
    df.to_csv(file_path, index=False)


# 1. 读取 NGSIM 数据
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['speed', 'acceleration', 'steering_angle']]
    return df


# 2. 数据预处理
def preprocess_data(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled, scaler


# 3. 驾驶风格分类（聚类）
def classify_driving_styles(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels, kmeans


# 4. 生成对抗网络（GAN）定义
def generate_synthetic_data(real_data, epochs=1000, batch_size=64):
    class Generator(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(Generator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)
            )

        def forward(self, z):
            return self.model(z)

    class Discriminator(nn.Module):
        def __init__(self, input_dim):
            super(Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.model(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(input_dim=3, output_dim=3).to(device)
    discriminator = Discriminator(input_dim=3).to(device)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)

    real_data = torch.tensor(real_data, dtype=torch.float32).to(device)
    dataset = TensorDataset(real_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for real_batch in dataloader:
            real_samples = real_batch[0].to(device)
            batch_size = real_samples.size(0)

            # 训练判别器
            optimizer_d.zero_grad()
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            real_output = discriminator(real_samples)
            real_loss = criterion(real_output, real_labels)

            z = torch.randn(batch_size, 3).to(device)
            fake_samples = generator(z)
            fake_output = discriminator(fake_samples.detach())
            fake_loss = criterion(fake_output, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # 训练生成器
            optimizer_g.zero_grad()
            fake_output = discriminator(fake_samples)
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            optimizer_g.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

    return generator


# 5. 生成典型路谱
def synthesize_routes(generator, num_samples=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.randn(num_samples, 3).to(device)
    synthetic_data = generator(z).cpu().detach().numpy()
    return synthetic_data


# 6. 可视化
def plot_comparison(real_data, synthetic_data):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.5, label='Real')
    plt.xlabel('Speed')
    plt.ylabel('Acceleration')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(synthetic_data[:, 0], synthetic_data[:, 1], alpha=0.5, color='r', label='Synthetic')
    plt.xlabel('Speed')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.show()


# 主流程
def main():
    file_path = "ngsim_data.csv"
    generate_dummy_ngsim_data(file_path)  # 生成随机数据
    data = load_data(file_path)
    data_scaled, scaler = preprocess_data(data)

    labels, model = classify_driving_styles(data_scaled)
    generator = generate_synthetic_data(data_scaled)
    synthetic_data = synthesize_routes(generator)

    plot_comparison(data_scaled, synthetic_data)


if __name__ == "__main__":
    main()
