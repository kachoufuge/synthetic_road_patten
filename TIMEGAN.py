import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset


# 假设你的数据已经保存为 CSV，包含 'speed', 'acceleration', 'steering_angle'
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['speed', 'acceleration', 'steering_angle']]
    return df


def preprocess_data(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled, scaler


def classify_driving_styles(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels, kmeans


# 为简化起见，我们假设 TimeGAN 只由 LSTM 构成的生成器和判别器
# 注意：真正的 TimeGAN 包含更多模块（如嵌入器、恢复器、监督器），这里只实现最基本的时序 GAN 部分

def generate_synthetic_data_timegan(real_data, epochs=1500, batch_size=32, seq_len=10):
    """
    假设 real_data 是形状为 (num_samples, seq_len, 3) 的时间序列数据，
    其中 3 表示 [speed, acceleration, steering_angle]。
    """

    # 生成器：输入噪声序列 -> 输出合成的时序数据
    class Generator(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, seq_len, output_dim):
            super(Generator, self).__init__()
            self.seq_len = seq_len
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_dim, output_dim)

        def forward(self, z):
            # z: (batch_size, seq_len, input_dim)
            lstm_out, _ = self.lstm(z)  # lstm_out: (batch_size, seq_len, hidden_dim)
            out = self.linear(lstm_out)  # out: (batch_size, seq_len, output_dim)
            return out

    # 判别器：判断一个时间序列是真实还是合成的
    class Discriminator(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, seq_len):
            super(Discriminator, self).__init__()
            self.seq_len = seq_len
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_dim, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # x: (batch_size, seq_len, input_dim)
            lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim)
            # 取序列最后一个时刻的隐藏状态进行判别
            last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
            out = self.linear(last_hidden)  # (batch_size, 1)
            out = self.sigmoid(out)
            return out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 这里输入和输出的特征数均为 3（速度、加速度、转向角），噪声也设为 3
    hidden_dim = 64
    num_layers = 1
    generator = Generator(input_dim=3, hidden_dim=hidden_dim, num_layers=num_layers,
                          seq_len=seq_len, output_dim=3).to(device)
    discriminator = Discriminator(input_dim=3, hidden_dim=hidden_dim, num_layers=num_layers,
                                  seq_len=seq_len).to(device)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)

    # real_data 应该是时间序列数据，形状为 (N, seq_len, 3)
    real_data = torch.tensor(real_data, dtype=torch.float32).to(device)
    dataset = TensorDataset(real_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for real_batch in dataloader:
            real_samples = real_batch[0].to(device)  # (batch_size, seq_len, 3)
            current_batch_size = real_samples.size(0)

            # 训练判别器
            optimizer_d.zero_grad()
            real_labels = torch.ones(current_batch_size, 1).to(device)
            fake_labels = torch.zeros(current_batch_size, 1).to(device)

            real_output = discriminator(real_samples)
            real_loss = criterion(real_output, real_labels)

            # 生成器输入噪声：形状 (batch_size, seq_len, 3)
            z = torch.randn(current_batch_size, seq_len, 3).to(device)
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


def synthesize_routes_timegan(generator, num_samples=500, seq_len=10):
    """
    生成合成数据，输出形状为 (num_samples, seq_len, 3)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.randn(num_samples, seq_len, 3).to(device)
    synthetic_data = generator(z).cpu().detach().numpy()
    return synthetic_data


# 3D 可视化（对比真实和合成数据，这里只可视化序列中第一个样本的全部时刻）
from mpl_toolkits.mplot3d import Axes3D


def plot_comparison_timegan(real_data, synthetic_data):
    # 为简单起见，我们只选取第一个样本的时间序列进行 3D 绘图
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # real_data[0]: shape (seq_len, 3)
    ax1.plot(real_data[0, :, 0], real_data[0, :, 1], real_data[0, :, 2], 'o-', label='Real')
    ax1.set_xlabel('Speed')
    ax1.set_ylabel('Acceleration')
    ax1.set_zlabel('Steering Angle')
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot(synthetic_data[0, :, 0], synthetic_data[0, :, 1], synthetic_data[0, :, 2], 'o-', color='r',
             label='Synthetic')
    ax2.set_xlabel('Speed')
    ax2.set_ylabel('Acceleration')
    ax2.set_zlabel('Steering Angle')
    ax2.legend()

    plt.show()


# 主流程
def main():
    file_path = "ngsim_data_CITY.csv"
    data = load_data(file_path)
    data_scaled, scaler = preprocess_data(data)

    # 注意：原始数据为 (N, 3) 需构造成时间序列数据。
    # 这里我们简单地将每连续 seq_len 个样本看作一个时间序列
    seq_len = 10
    num_records = data_scaled.shape[0]
    num_sequences = num_records // seq_len
    # 截断数据以整除 seq_len
    data_seq = data_scaled[:num_sequences * seq_len]
    # 重塑为 (num_sequences, seq_len, 3)
    data_seq = data_seq.reshape((num_sequences, seq_len, 3))

    labels, model = classify_driving_styles(data_scaled)

    # 训练 TimeGAN
    generator = generate_synthetic_data_timegan(data_seq, epochs=1500, batch_size=32, seq_len=seq_len)

    # 生成合成数据（归一化空间）
    synthetic_data = synthesize_routes_timegan(generator, num_samples=500, seq_len=seq_len)

    # 对合成数据进行反归一化处理
    # 注意：由于 scaler 是在原始 (N, 3) 数据上训练的，
    # 这里需要先将 (num_samples, seq_len, 3) 数据转换为二维，再反归一化，再恢复为三维
    synthetic_data_2d = synthetic_data.reshape(-1, 3)
    synthetic_data_inv = scaler.inverse_transform(synthetic_data_2d)
    synthetic_data_inv = synthetic_data_inv.reshape(synthetic_data.shape)  # (num_samples, seq_len, 3)

    # 保存反归一化后的合成数据到 CSV 文件
    # 这里将每个序列展平后保存，每行表示一个序列
    df_synthetic = pd.DataFrame(synthetic_data_inv.reshape(synthetic_data_inv.shape[0], -1))
    df_synthetic.to_csv("synthetic_output_timegan.csv", index=False)
    print("反归一化后的合成数据已保存到 synthetic_output_timegan.csv")

    # 可视化对比（这里对比第一条时间序列）
    # 真实数据同样选取第一条序列
    plot_comparison_timegan(data_seq, synthetic_data)


if __name__ == "__main__":
    main()
