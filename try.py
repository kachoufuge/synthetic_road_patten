import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- 定义数据预处理函数 ---
def preprocess_data(df):
    # 只取我们关心的列：速度、加速度、转向角
    features = ['speed', 'acceleration', 'steering_angle']
    data = df[features].values

    # 使用 StandardScaler 对数据进行标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

# --- 定义特征分析函数 ---
def analyze_features(df):
    """
    绘制不同 scenario_type 下的速度直方图，用于展示各场景的速度分布。
    """
    # 获取所有不同的场景类型
    scenario_types = df['scenario_type'].unique()
    plt.figure(figsize=(10, 4))
    # 针对每个场景绘制一个子图
    for i, stype in enumerate(scenario_types):
        subset = df[df['scenario_type'] == stype]  # 筛选出当前场景的数据
        plt.subplot(1, len(scenario_types), i+1)   # 1行多列，第 i+1 个子图
        plt.hist(subset['speed'], bins=20, alpha=0.7, label='Speed')
        plt.title(f"Scenario: {stype}")
        plt.xlabel("Speed (m/s)")
        plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# --- 构造示例输入数据 ---
# 这里我们模拟一个 merge_df 数据框，包含多驾驶员、不同工况下的数据
data = {
    "driver_id": [1, 1, 2, 2, 3, 3, 4, 4],
    "scenario_type": ["urban", "urban", "highway", "highway", "urban", "urban", "highway", "highway"],
    "speed": [10, 12, 30, 32, 15, 14, 28, 29],
    "acceleration": [0.5, 0.6, 0.2, 0.3, 0.8, 0.7, 0.1, 0.2],
    "steering_angle": [5, 6, 2, 1, 4, 5, 3, 2]
}
merge_df = pd.DataFrame(data)

# --- 打印输入数据 ---
print("原始数据 (merge_df):")
print(merge_df)

# --- 调用预处理函数，获得标准化后的数据 ---
data_scaled, scaler = preprocess_data(merge_df)
print("\n标准化后的数据（前5行）：")
print(data_scaled[:5])

# --- 调用特征分析函数，显示直方图 ---
analyze_features(merge_df)
