import pandas as pd
import numpy as np

# 读取文件
file_path = "synthetic_output_timegan.csv"  # 替换为你的文件路径
data = pd.read_csv(file_path, header=None,encoding="utf-8")  # 读取数据，无表头

# 确保数据的形状为 (-1, 3)
num_columns = 3
data = data.values.reshape(-1, num_columns)

# 创建 DataFrame 并添加表头
df = pd.DataFrame(data, columns=["speed", "acceleration", "steering_angel"])

# 保存到新的 CSV 文件
output_path = "formatted_synthetic_output.csv"
df.to_csv(output_path, index=False,encoding="utf-8")

print(f"格式化数据已保存到: {output_path}")