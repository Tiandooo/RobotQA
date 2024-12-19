import os
import pandas as pd
from threading import Event

def run_data_cleaning(params, progress_callback, stop_event: Event):
    input_path = params.get("input_path")  # 父路径
    output_path = params.get("output_path")  # 输出的父路径
    cleaning_params = params.get("cleaning_params")

    # 初始化清洗记录
    deleted_values = []
    filled_values = []

    # 遍历所有子目录和文件
    csv_files = []
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    total_files = len(csv_files)
    if total_files == 0:
        raise ValueError("No CSV files found in the input directory.")

    for idx, file_path in enumerate(csv_files):

        if stop_event.is_set():
            print(f"Task stopped by user. Stopping at file {idx + 1}/{total_files}.")

            return {
                "status": "paused",
                "message": "Task stopped.",
            }


        # 生成输出路径，保持与输入结构一致
        relative_path = os.path.relpath(file_path, input_path)  # 子目录结构
        clean_vis_dir = os.path.join(output_path, "clean_vis", os.path.dirname(relative_path))
        clean_res_dir = os.path.join(output_path, "cleaning_res", os.path.dirname(relative_path))
        os.makedirs(clean_vis_dir, exist_ok=True)
        os.makedirs(clean_res_dir, exist_ok=True)

        # 读取 CSV 文件
        df = pd.read_csv(file_path)

        # 标记
        delete_flag = False
        fill_flag = False

        # 处理缺失值
        if cleaning_params.get("missing_value") == "delete":
            deleted_df = df[df.isnull().any(axis=1)]  # 缺失值的行
            if not deleted_df.empty:
                delete_flag = True
            deleted_values.extend(deleted_df.to_dict(orient="records"))  # 保存删除的记录
            df = df.dropna()  # 删除含缺失值的行
        elif cleaning_params.get("missing_value") == "fill":
            fill_method = cleaning_params.get("fill_method", "linear")  # 默认线性插值
            filled_df = df[df.isnull().any(axis=1)].copy()  # 保存填充前的记录
            if not filled_df.empty:
                fill_flag = True
            # 按列类型插值
            for column in df.columns:
                if df[column].isnull().any():
                    if df[column].dtype.kind in 'biufc':  # 数值型数据
                        df[column] = df[column].interpolate(method=fill_method, limit_direction="both")
                    else:  # 非数值型数据（如分类）
                        df[column] = df[column].fillna(method="ffill").fillna(method="bfill")

            filled_values.extend(filled_df.to_dict(orient="records"))  # 保存填充内容

        # 处理重复值
        if cleaning_params.get("remove_duplicates"):
            df = df.drop_duplicates()

        # 保存清洗后的文件
        output_clean_file = os.path.join(clean_res_dir, os.path.basename(file_path))
        df.to_csv(output_clean_file, index=False)

        # 保存删除或填充的记录
        if cleaning_params.get("missing_value") == "delete" and delete_flag:
            vis_file = os.path.join(clean_vis_dir, f"deleted_{os.path.basename(file_path)}")
            deleted_df.to_csv(vis_file, index=False)
        elif cleaning_params.get("missing_value") == "fill" and fill_flag:
            vis_file = os.path.join(clean_vis_dir, f"filled_{os.path.basename(file_path)}")
            filled_df.to_csv(vis_file, index=False)

        # 更新进度
        progress_callback(int((idx + 1) / total_files * 100))
        print("当前进度： ", int((idx + 1) / total_files * 100))

    return {
        "deleted_values": deleted_values,
        "filled_values": filled_values,
        "output_path": os.path.join(output_path, "cleaning_res"),
        "status": "success"
    }


