import numpy as np
import pandas as pd

file_path = r"D:\Astudy\RobotQA\data\kitchen_40demo\raw_data\1.0.0\cartesian_joint.csv"  # 替换为您的文件路径

data = pd.read_csv(file_path)


# 1. 时间间隔一致性
def check_time_intervals(data, time_col="timestamp"):
    time_diffs = np.diff(data[time_col])
    std_dev = np.std(time_diffs)  # 时间间隔的标准差
    mean_interval = np.mean(time_diffs)

    # 分数：标准差越小，得分越高
    interval_score = max(0, 100 - (std_dev / mean_interval) * 1000)
    return interval_score, time_diffs


# 2. 变量随时间的平滑性
def check_smoothness(data):
    smoothness_scores = []
    for col in data.columns[2:]:  # 忽略前两列
        diff = np.diff(data[col])
        std_diff = np.std(diff)  # 差分的标准差
        smoothness_scores.append(max(0, 100 - std_diff * 100))
    return np.mean(smoothness_scores)


# 3. 动态一致性（差分相关性）
def check_dynamic_consistency(data):
    columns = data.columns[2:]  # 忽略前两列
    diff_data = data[columns].diff().iloc[1:]  # 计算差分，去掉第一个NaN值
    correlation_matrix = diff_data.corr()  # 差分之间的相关性
    # 提取上三角相关系数
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    avg_correlation = upper_triangle.mean().mean()  # 平均相关性
    return avg_correlation * 100


# 4. 异常点检测
def check_outliers(data):
    outlier_scores = []
    for col in data.columns[2:]:
        diff = np.diff(data[col])
        z_scores = (diff - np.mean(diff)) / np.std(diff)  # Z-Score 标准化
        outliers = np.sum(np.abs(z_scores) > 3)  # 离群点计数
        outlier_scores.append(max(0, 100 - outliers * 10))
    return np.mean(outlier_scores)


# 综合评估
def evaluate_time_series_consistency(data):
    interval_score, time_diffs = check_time_intervals(data)
    smoothness_score = check_smoothness(data)
    dynamic_consistency_score = check_dynamic_consistency(data)
    outlier_score = check_outliers(data)

    # 综合得分加权平均
    total_score = (
            0.4 * interval_score +  # 时间间隔一致性占比 40%
            0.3 * smoothness_score +  # 平滑性占比 30%
            0.2 * dynamic_consistency_score +  # 动态一致性占比 20%
            0.1 * outlier_score  # 异常点检测占比 10%
    )
    return {
        "Interval Score": interval_score,
        "Smoothness Score": smoothness_score,
        "Dynamic Consistency Score": dynamic_consistency_score,
        "Outlier Score": outlier_score,
        "Total Consistency Score": total_score
    }


# 执行评估
results = evaluate_time_series_consistency(data)
for metric, score in results.items():
    print(f"{metric}: {score:.2f}")
