import re
import os
import chardet

with open("ill_ocr_regex.txt", "r", encoding="utf-8") as file:
    regex_patterns = file.read().splitlines()

directory_path = "./books"
output_directory_path = "./corrected"

for root, directories, filenames in os.walk(directory_path):
    for filename in filenames:
        if filename.endswith(".txt"):
            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, directory_path)
            output_file_path = os.path.join(output_directory_path, relative_path)

            with open(file_path, "rb") as file:
                raw_data = file.read()
                encoding_result = chardet.detect(raw_data)
                file_encoding = encoding_result["encoding"]

            with open(file_path, "r", encoding=file_encoding, errors="ignore") as file:
                book_text = file.read()

            for pattern in regex_patterns:
                book_text = re.sub(pattern, "", book_text)
            book_text = re.sub(r'；', '\n', book_text)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            with open(output_file_path, "w", encoding="utf-8") as file:
                file.write(book_text)

            os.remove(file_path)
            print(f"{file_path}的错误已更正并保存到 {output_file_path}.")
