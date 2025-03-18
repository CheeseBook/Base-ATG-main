import json


def remove_id_from_json(file_path, output_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ["train", "dev", "test"]:
        for item in data.get(key, []):
            item.pop("id", None)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# 示例用法
remove_id_from_json("dataset/CQI/cqi.json", "dataset/CQI/cqi.json")
