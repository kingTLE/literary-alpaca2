import os
import re


def rename_files1(path, old_chars, new_char):
    for root, dirs, files in os.walk(path):
        for file in files:
            new_file = re.sub(old_chars, new_char, file)
            if new_file != file:
                os.rename(os.path.join(root, file), os.path.join(root, new_file))
        for dir in dirs:
            new_dir = re.sub(old_chars, new_char, dir)
            if new_dir != dir:
                os.rename(os.path.join(root, dir), os.path.join(root, new_dir))


def rename_files(path):
    for root, dirs, files in os.walk(path):
        files.sort()
        count = 1
        for file in files:
            new_file = str(count) + os.path.splitext(file)[1]
            os.rename(os.path.join(root, file), os.path.join(root, new_file))
            count += 1


# 指定目录路径、要替换的字符和替换后的字符
path = "/your/directory/path"
# old_chars = r"[\&\[\&\]]+"
# new_char = ""

# 调用函数进行批量修改
# rename_files1(path, old_chars, new_char)
rename_files(path)
