import csv
import json

# 从JSON文件加载数据
with open('../data/LCCD.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# 创建CSV文件
filename = "LCCD.csv"
with open(filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # 写入表头
    writer.writerow(["text"])

    # 遍历每个对话
    for conversation in data:
        # 初始化对话文本
        conversation_text = ""

        # 遍历每一轮对话
        for i, utterance in enumerate(conversation):
            # 根据奇偶性确定是Human还是Assistant的发言
            speaker = "Human" if i % 2 == 0 else "Assistant"
            conversation_text += f"<s>{speaker}: {utterance}\n</s>"

        # 写入CSV文件
        writer.writerow([conversation_text])