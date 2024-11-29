import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 示例数据，假设输入是一个 DataFrame
# data = pd.DataFrame({
#     "Time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
#     "Var1": [0.0, 0.219, 0.410, 0.770, 1.044, 1.228, 1.410, 1.775, 2.171, 2.499, 2.620, 2.944],
#     "Var2": [0.0, -0.007, 0.001, 0.010, 0.010, 0.009, 0.010, 0.013, 0.008, 0.007, 0.004, 0.006],
#     "Var3": [-0.698, -0.698, -0.699, -0.674, -0.666, -0.650, -0.624, -0.569, -0.534, -0.532, -0.526, -0.485],
#     "Var4": [0.0194, 0.0194, 0.0199, 0.0198, 0.0194, 0.0193, 0.0197, 0.0191, 0.0187, 0.0186, 0.0192, 0.0197],
#     "Var5": [-2.48, -2.48, -2.49, -2.48, -2.48, -2.47, -2.46, -2.44, -2.42, -2.42, -2.42, -2.40],
#     "Var6": [0.044, 0.044, 0.035, 0.026, 0.025, 0.025, 0.019, 0.021, 0.042, 0.046, 0.049, 0.040],
#     "Var7": [1.818, 1.818, 1.848, 1.862, 1.867, 1.876, 1.888, 1.909, 1.923, 1.925, 1.927, 1.917],
#     "Var8": [0.734, 0.734, 0.738, 0.749, 0.749, 0.750, 0.757, 0.763, 0.758, 0.756, 0.755, 0.789]
# })

file_path = r"D:\Astudy\RobotQA\data\kitchen_40demo\raw_data\1.0.0\cartesian_joint.csv"  # 替换为您的文件路径

data = pd.read_csv(file_path)

# 1. 完整性检查
def check_completeness(data):
    missing = data.isnull().sum().sum()
    return 100 if missing == 0 else max(0, 100 - missing * 10)

# 2. 时间间隔一致性
def check_time_intervals(data, time_col="timestamp"):
    time_diffs = np.diff(data[time_col])
    std_dev = np.std(time_diffs)
    return max(0, 100 - std_dev * 50)

# 3. 平稳性评估
def check_stationarity(data):
    stationarity_scores = []
    for col in data.columns[2:]:  # 忽略前两列（frame_idx 和 timestamp）
        diff = np.diff(data[col])
        stationarity_scores.append(100 - min(100, np.std(diff) * 100))
    return np.mean(stationarity_scores)

# 4. 噪声评估
def check_noise(data):
    noise_scores = []
    for col in data.columns[2:]:
        high_freq_energy = np.sum(np.abs(np.fft.fft(data[col])[5:]))
        total_energy = np.sum(np.abs(np.fft.fft(data[col])))
        ratio = high_freq_energy / total_energy
        noise_scores.append(100 - min(100, ratio * 1000))
    return np.mean(noise_scores)

# 5. 数据一致性（变量间相关性）
def check_correlation(data):
    correlation_scores = []
    columns = data.columns[2:]  # 忽略前两列
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            corr, _ = pearsonr(data[columns[i]], data[columns[j]])
            correlation_scores.append(abs(corr) * 100)
    return np.mean(correlation_scores)

# 综合评分
def evaluate_data_quality(data):
    completeness_score = check_completeness(data)
    interval_score = check_time_intervals(data)
    stationarity_score = check_stationarity(data)
    noise_score = check_noise(data)
    correlation_score = check_correlation(data)

    # 综合加权平均得分
    total_score = (
        0.2 * completeness_score +
        0.2 * interval_score +
        0.2 * stationarity_score +
        0.2 * noise_score +
        0.2 * correlation_score
    )
    return {
        "Completeness Score": completeness_score,
        "Interval Score": interval_score,
        "Stationarity Score": stationarity_score,
        "Noise Score": noise_score,
        "Correlation Score": correlation_score,
        "Total Quality Score": total_score
    }

# 执行评估
results = evaluate_data_quality(data)
for metric, score in results.items():
    print(f"{metric}: {score:.2f}")