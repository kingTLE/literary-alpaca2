import csv
import json

# 读取原始 JSON 文件
with open('../data/math23k_train.json', 'r', encoding='utf-8') as json_file:
    # 初始化一个列表，用于存储最终的数据
    data_list = []

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

        # 如果是 JSON 对象的结尾，则尝试解析并添加到数据列表中
        if is_end:
            try:
                item = json.loads(current_json)

                original_text = item["original_text"]
                equation = item["equation"]
                ans = item["ans"]

                # 创建数据字典
                data_dict = {
                    "instruction": original_text,
                    "input": equation,
                    "output": ans
                }

                # 将数据字典添加到列表中
                data_list.append(data_dict)

            except json.decoder.JSONDecodeError as e:
                print(f"解析 JSON 对象时发生错误：{e}")

# 创建一个 JSON 文件，并将最终的数据写入其中
filename = "math23k.json"
with open(filename, mode='w', encoding='utf-8') as file:
    json.dump(data_list, file, ensure_ascii=False, indent=2)

print(f"转换完成，生成的 JSON 文件名为: {filename}")
