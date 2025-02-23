import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ydata_synthetic.synthesizers import TimeGAN
from ydata_synthetic.synthesizers.gan import TimeSeriesGAN


def generate_time_series_data(real_data, seq_length=24, epochs=1000, batch_size=32):
    """
    使用TimeGAN生成时间序列数据
    :param real_data: 原始数据，形状为(N, features)
    :param seq_length: 时间序列长度（时间窗口）
    :param epochs: 训练轮次
    :param batch_size: 批大小
    :return: 训练好的TimeGAN模型
    """

    # 将数据转换为时间序列格式
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            sequences.append(seq)
        return np.array(sequences)

    # 数据预处理
    real_sequences = create_sequences(real_data, seq_length)
    print(f"生成的时间序列数据形状: {real_sequences.shape}")  # (num_samples, seq_length, features)

    # 配置TimeGAN参数
    gan_args = {
        "batch_size": batch_size,
        "lr": 0.001,
        "noise_dim": 3,  # 噪声维度
        "layer_dim": 128,  # 隐藏层维度
        "gamma": 1,  # 监督损失系数
        "n_seq": real_sequences.shape[2],  # 特征数量
        "hidden_dim": 24  # 隐藏状态维度
    }

    # 初始化并训练TimeGAN
    synth = TimeSeriesGAN(
        model_parameters=gan_args,
        n_seq=gan_args["n_seq"],
        hidden_dim=gan_args["hidden_dim"]
    )

    synth.train(
        real_sequences,
        train_steps=epochs,
        seq_len=seq_length,
        n_seq=gan_args["n_seq"]
    )

    return synth


def synthesize_time_series(model, num_samples=500, seq_length=24):
    """
    生成时间序列数据
    :param model: 训练好的TimeGAN模型
    :param num_samples: 要生成的样本数
    :param seq_length: 时间序列长度
    :return: 合成数据，形状为(num_samples, seq_length, features)
    """
    synthetic_sequences = model.sample(num_samples)
    return synthetic_sequences


# 使用示例
if __name__ == "__main__":
    # 1. 准备真实数据（示例）
    real_data = np.random.randn(10000, 3)  # 假设有10000个时间步，3个特征

    # 2. 训练TimeGAN模型
    seq_length = 24  # 2.4秒时间窗口（假设每个时间步0.1秒）
    gan_model = generate_time_series_data(real_data, seq_length=seq_length)

    # 3. 生成合成数据
    synthetic_sequences = synthesize_time_series(gan_model, num_samples=500)

    # 4. 后处理（将序列转换回时间步）
    synthetic_data = synthetic_sequences.reshape(-1, 3)  # 形状变为(500*24, 3)

    # 5. 保存数据
    np.save("synthetic_time_series.npy", synthetic_data)