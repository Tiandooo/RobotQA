def shuffle_and_reduce_keywords(input_file, output_file=None, keep_ratio=0.88):
    """
    读取敏感词文件，打乱顺序并保留前88%的词汇

    参数:
        input_file (str): 输入文件路径
        output_file (str, optional): 输出文件路径，默认为None（不保存）
        keep_ratio (float): 保留的词汇比例，默认为0.88

    返回:
        list: 处理后的敏感词列表
    """
    import random

    # 读取文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # 检查第一行是否为标识
            first_line = f.readline().strip()
            if first_line != "#KWLIST":
                print(f"敏感词文件格式错误: 第一行应为 #KWLIST，实际为 {first_line}")
                return []

            # 读取所有非空行
            words = [line.strip() for line in f if line.strip()]

        # 打乱顺序
        random.shuffle(words)

        # 选取前88%
        keep_count = int(len(words) * keep_ratio)
        selected_words = words[:keep_count]

        print(f"原始敏感词数量: {len(words)}")
        print(f"保留敏感词数量: {keep_count} ({keep_ratio*100:.1f}%)")

        # 如果指定了输出文件，则保存
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("#KWLIST\n")  # 写入标识行
                for word in selected_words:
                    f.write(f"{word}\n")
            print(f"已保存处理后的敏感词到: {output_file}")

        return selected_words

    except Exception as e:
        print(f"处理敏感词文件失败: {str(e)}")
        return []

# 使用示例
shuffled_words = shuffle_and_reduce_keywords(
    "sensitive_words_lines.txt",
    "敏感词汇.txt"
)

