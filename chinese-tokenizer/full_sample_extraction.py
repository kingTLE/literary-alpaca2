import os
from tqdm import tqdm


def main():
    corrected_folder = "./corrected"
    output_folder = "./books_dataset"

    os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(output_folder, "booksdataset.txt"), "w", encoding="utf-8") as data_file:
        for root, _, files in os.walk(corrected_folder, topdown=False):
            print(f"正在加载目录:{root}...")
            for filename in tqdm(files):
                if not filename.endswith(".txt"):
                    continue

                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, encoding="utf-8") as file:
                        j = file.read()
                        data_file.write(j)
                        data_file.write("\n")
                except:
                    print(f"出现错误，跳过:{file_path}")


if __name__ == "__main__":
    main()
