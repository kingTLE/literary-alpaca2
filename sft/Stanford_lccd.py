import csv
import json

# 从JSON文件加载数据
with open('../data/LCCD.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# 创建JSON数据列表
json_data_list = []

# 遍历每个对话
for conversation in data:
    # 初始化对话文本
    conversation_text = ""

    # 遍历每一轮对话
    for i, utterance in enumerate(conversation):
        # 根据奇偶性确定是Human还是Assistant的发言
        speaker = "Human" if i % 2 == 0 else "Assistant"
        conversation_text += f"{speaker}: {utterance}\n"

    # 将对话拆分为instruction、input和output
    instruction = "下面是人类之间的对话与交流"
    input_text = conversation[0]
    output_text = conversation[1:]

    # 创建数据字典
    data_dict = {
        "instruction": instruction,
        "input": input_text,
        "output": output_text
    }

    # 将数据字典添加到JSON数据列表中
    json_data_list.append(data_dict)

# 创建一个 JSON 文件，并将最终的数据写入其中
filename = "LCCD.json"
with open(filename, mode='w', encoding='utf-8') as file:
    json.dump(json_data_list, file, ensure_ascii=False, indent=2)

print(f"转换完成，生成的 JSON 文件名为: {filename}")
