import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


###################################################
# 1. 数据加载与合并（可选）
###################################################
# def load_and_merge_data(folder_path):
#     """
#     假设 folder_path 下有多个 CSV 文件，每个文件可能对应不同驾驶员/工况的数据
#     CSV 列示例: [driver_id, scenario_type, speed, acceleration, steering_angle, ...]
#     """
#     all_files = glob.glob(os.path.join(folder_path, "*.csv"))
#     df_list = []
#     for file in all_files:
#         df = pd.read_csv(file)
#         df_list.append(df)
#     merged_df = pd.concat(df_list, ignore_index=True)
#     return merged_df
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['speed', 'acceleration', 'steering_angle']]
    return df

###################################################
# 2. 数据预处理 & 特征分析
###################################################
def preprocess_data(df):
    # 只取我们关心的列
    features = ['speed', 'acceleration', 'steering_angle']
    data = df[features].values

    # 标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler


# def analyze_features(df):
#     """
#     简单示例：绘制速度/加速度分布直方图，或者不同 scenario_type 的统计对比
#     """
#     scenario_types = df['scenario_type'].unique()
#     plt.figure(figsize=(10, 4))
#     for i, stype in enumerate(scenario_types):
#         subset = df[df['scenario_type'] == stype]
#         plt.subplot(1, len(scenario_types), i + 1)
#         plt.hist(subset['speed'], bins=20, alpha=0.7, label='Speed')
#         plt.title(f"Scenario: {stype}")
#         plt.xlabel("Speed (m/s)")
#         plt.ylabel("Count")
#     plt.tight_layout()
#     plt.show()


###################################################
# 3. 驾驶风格/工况分类（可选）
###################################################
# def classify_driving_styles(data_scaled, n_clusters=3):
#     """
#     基于 K-Means 对驾驶风格进行聚类
#     也可换成 GMM/DBSCAN 等
#     """
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     labels = kmeans.fit_predict(data_scaled)
#     return labels, kmeans
#

###################################################
# 4. 数据合成算法（GAN 为例）
###################################################
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


def synthesize_routes(generator, num_samples=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.randn(num_samples, 3).to(device)
    synthetic_data = generator(z).cpu().detach().numpy()
    return synthetic_data


###################################################
# 5. 可视化对比
###################################################
def plot_comparison(real_data, synthetic_data):
    # 转换输入为DataFrame（如果传入的是NumPy数组）
    if not isinstance(real_data, pd.DataFrame):
        real_data = pd.DataFrame(real_data,
                                 columns=['speed', 'acceleration', 'steering_angle'])

    if not isinstance(synthetic_data, pd.DataFrame):
        synthetic_data = pd.DataFrame(synthetic_data,
                                      columns=['speed', 'acceleration', 'steering_angle'])

    # 3D可视化
    fig = plt.figure(figsize=(15, 10))

    # 真实数据散点图
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(real_data['speed'],
                real_data['acceleration'],
                real_data['steering_angle'],
                alpha=0.5,
                label='Real')

    # 合成数据散点图
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(synthetic_data['speed'],
                synthetic_data['acceleration'],
                synthetic_data['steering_angle'],
                color='r',
                alpha=0.5,
                label='Synthetic')

    # 统一坐标轴范围
    max_speed = max(real_data['speed'].max(), synthetic_data['speed'].max())
    max_accel = max(real_data['acceleration'].abs().max(), synthetic_data['acceleration'].abs().max())

    for ax in [ax1, ax2]:
        ax.set_xlim(0, max_speed)
        ax.set_ylim(-max_accel, max_accel)
        ax.set_zlim(-35, 35)
        ax.set_xlabel('Speed (m/s)')
        ax.set_ylabel('Acceleration (m/s²)')
        ax.set_zlabel('Steering Angle (°)')

    plt.tight_layout()
    plt.savefig('comparison_3d.png')
    plt.show()
###################################################
# 主函数示例
###################################################
def main():
    # 1) 加载/合并数据
    # folder_path = "./sim_data"  # 存放多驾驶员多工况 CSV 的文件夹  可选
    #df = load_and_merge_data(folder_path)
    file_path = "ngsim_data_CITY.csv"
    df = load_data(file_path)
    # 2) 特征分析 (可选)
    #analyze_features(df)

    # 3) 数据预处理 (标准化) + 聚类识别不同驾驶风格 (可选)
    data_scaled, scaler = preprocess_data(df)
    #labels, kmeans_model = classify_driving_styles(data_scaled, n_clusters=3)
    #df['cluster_label'] = labels

    # 4) 训练 GAN 进行数据合成
    generator = generate_synthetic_data(data_scaled)

    # 5) 合成新路谱
    synthetic_data = synthesize_routes(generator)
    # 6) 对合成数据进行反归一化处理
    synthetic_data_inv = scaler.inverse_transform(synthetic_data)

    # 7) 可视化对比
    plot_comparison(df, synthetic_data_inv)

    # 8) 存入csv表格
    df_synthetic_CITY = pd.DataFrame(synthetic_data_inv, columns=['speed', 'acceleration', 'steering_angle'])
    df_synthetic_CITY.to_csv("synthetic_output_inv.csv", index=False)

if __name__ == '__main__':
    main()
