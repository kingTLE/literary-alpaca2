import csv
import json

# 读取原始 JSON 文件
with open('../data/math23k_train.json', 'r', encoding='utf-8') as json_file:
    # 初始化一个列表，用于存储最终的 CSV 行
    csv_rows = ["text"]

    # 用于存储当前 JSON 对象的字符串
    current_json = ""

    for line in json_file:
        # 判断当前行是否为 JSON 对象的开头或结尾
        is_start = line.strip().startswith("{")
        is_end = line.strip().endswith("}")

        # 如果是 JSON 对象的开头，则将当前 JSON 字符串置空
        if is_start:
            current_json = ""

        # 将当前行添加到当前 JSON 对象字符串
        current_json += line

        # 如果是 JSON 对象的结尾，则尝试解析并添加到 CSV 行中
        if is_end:
            try:
                item = json.loads(current_json)

                original_text = item["original_text"]
                equation = item["equation"]
                ans = item["ans"]

                human_text = f"<s>Human: {original_text}</s>"
                assistant_text = f"<s>Assistant: 根据方程式{equation}解得:\n{ans}</s>"
                example = f"{human_text}{assistant_text}"
                csv_rows.append(example)

            except json.decoder.JSONDecodeError as e:
                print(f"解析 JSON 对象时发生错误：{e}")

# 创建一个 CSV 文件，并将最终的 CSV 行写入其中
filename = "math23k.csv"
with open(filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows([row] for row in csv_rows)

print(f"转换完成，生成的 CSV 文件名为: {filename}")
