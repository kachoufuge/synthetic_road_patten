import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import interp1d
# 生成随机NGSIM数据
# def generate_dummy_ngsim_data(file_path, num_samples=1000):
#     np.random.seed(42)
#     speed = np.random.uniform(0, 30, num_samples)  # 速度（0-30 m/s）
#     acceleration = np.random.uniform(-3, 3, num_samples)  # 加速度（-3~3 m/s²）
#     steering_angle = np.random.uniform(-30, 30, num_samples)  # 方向盘角度（-30°~30°）
#
#     df = pd.DataFrame({
#         'speed': speed,
#         'acceleration': acceleration,
#         'steering_angle': steering_angle
#     })
#     df.to_csv(file_path, index=False)

# def generate_realistic_ngsim_data(file_path, num_samples=1000):
#     np.random.seed(42)
#
#     # 初始化数组
#     speed = np.zeros(num_samples)
#     acceleration = np.zeros(num_samples)
#     steering_angle = np.zeros(num_samples)
#
#     # 驾驶状态参数
#     current_speed = 0.0
#     current_accel = 0.0
#     current_steer = 0.0
#
#     # 定义驾驶阶段状态机
#     states = ['launch', 'cruise', 'turn']
#     current_state = 'launch'
#     state_duration = 0
#     max_speed = np.random.uniform(20, 30)  # 最大巡航速度
#
#     for t in range(num_samples):
#         # 状态转换逻辑
#         if state_duration <= 0:
#             # 概率转移到下一个状态
#             if current_state == 'launch':
#                 current_state = 'cruise'
#                 state_duration = np.random.randint(50, 200)  # 巡航持续时间
#             elif current_state == 'cruise':
#                 # 30%概率进入转弯
#                 if np.random.rand() < 0.3:
#                     current_state = 'turn'
#                     state_duration = np.random.randint(20, 50)
#                     turn_direction = np.sign(np.random.randn())  # 随机转向方向
#                     target_steer = turn_direction * np.random.uniform(15, 30)
#                 else:
#                     state_duration = np.random.randint(50, 150)
#             else:  # turn
#                 current_state = 'cruise'
#                 state_duration = np.random.randint(100, 300)
#
#         # 状态行为模拟
#         if current_state == 'launch':
#             # 启动加速阶段
#             target_accel = np.clip(np.random.normal(1.5, 0.5), 0.8, 2.5)
#             current_accel = smooth_transition(current_accel, target_accel, 0.5)
#             current_speed += current_accel * 0.1  # 时间步长假设0.1秒
#
#         elif current_state == 'cruise':
#             # 平稳巡航阶段
#             target_speed = max_speed + np.random.normal(0, 1)
#             speed_error = target_speed - current_speed
#             current_accel = np.clip(speed_error * 0.1, -0.5, 0.5)
#             current_speed += current_accel * 0.1
#             current_steer = smooth_transition(current_steer, 0, 0.2)
#
#         elif current_state == 'turn':
#             # 转弯阶段
#             current_accel = np.clip(current_accel - 0.1, -2.0, 0)
#             current_speed += current_accel * 0.1
#             current_steer = smooth_transition(current_steer, target_steer, 0.8)
#
#         # 物理限制约束
#         current_speed = np.clip(current_speed, 0, 35)
#         current_accel = np.clip(current_accel, -3, 3)
#         current_steer = np.clip(current_steer, -35, 35)
#
#         # 添加噪声
#         speed[t] = current_speed + np.random.normal(0, 0.2)
#         acceleration[t] = current_accel + np.random.normal(0, 0.1)
#         steering_angle[t] = current_steer + np.random.normal(0, 0.5)
#
#         state_duration -= 1
#
#     # 后处理平滑
#     speed = smooth_trajectory(speed, window=5)
#     steering_angle = smooth_trajectory(steering_angle, window=3)
#
#     df = pd.DataFrame({
#         'speed': speed,
#         'acceleration': acceleration,
#         'steering_angle': steering_angle
#     })
#
#     df.to_csv(file_path, index=False)
#
#
# def smooth_transition(current, target, rate):
#     """平滑参数过渡"""
#     return current + (target - current) * rate
#
#
# def smooth_trajectory(data, window=3):
#     """应用移动平均平滑"""
#     return np.convolve(data, np.ones(window) / window, mode='same')

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
def generate_synthetic_data(real_data, epochs=1500, batch_size=32):
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
def synthesize_routes(generator, num_samples=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.randn(num_samples, 3).to(device)
    synthetic_data = generator(z).cpu().detach().numpy()
    return synthetic_data


# 6. 可视化
# def plot_comparison(real_data, synthetic_data):
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.5, label='Real')
#     plt.xlabel('Speed')
#     plt.ylabel('Acceleration')
#     plt.legend()
#
#     plt.subplot(1, 2, 2)
#     plt.scatter(synthetic_data[:, 0], synthetic_data[:, 1], alpha=0.5, color='r', label='Synthetic')
#     plt.xlabel('Speed')
#     plt.ylabel('Acceleration')
#     plt.legend()
#     plt.show()
def plot_comparison(real_data, synthetic_data):
    fig = plt.figure(figsize=(12, 6))
    # 第一个子图：真实数据的 3D 散点图
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(real_data[:, 0], real_data[:, 1], real_data[:, 2], alpha=0.5, label='Real')
    ax1.set_xlabel('Speed')
    ax1.set_ylabel('Acceleration')
    ax1.set_zlabel('Steering Angle')
    ax1.legend()

    # 第二个子图：合成数据的 3D 散点图
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(synthetic_data[:, 0], synthetic_data[:, 1], synthetic_data[:, 2], alpha=0.5, color='r',
                label='Synthetic')
    ax2.set_xlabel('Speed')
    ax2.set_ylabel('Acceleration')
    ax2.set_zlabel('Steering Angle')
    ax2.legend()

    plt.show()


# 主流程
def main():
    file_path = "ngsim_data_CITY.csv"
    # generate_realistic_ngsim_data(file_path)  # 生成随机数据
    data = load_data(file_path)
    data_scaled, scaler = preprocess_data(data)

    labels, model = classify_driving_styles(data_scaled)
    generator = generate_synthetic_data(data_scaled)
    synthetic_data = synthesize_routes(generator)

    plot_comparison(data_scaled, synthetic_data)


if __name__ == "__main__":
    main()
