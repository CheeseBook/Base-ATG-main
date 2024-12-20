import json
import pickle

# 输入 JSON 文件路径
input_file = "dataset/CQI/cqi.json"
# 输出 .pkl 文件路径
output_file = "dataset/cqi.pkl"


def validate_and_convert(data):
    """
    验证并转换单条数据的格式
    """
    tokens = data["tokens"]
    max_idx = len(tokens) - 1

    # 验证 ner
    valid_ner = []
    for entity in data["ner"]:
        if 0 <= entity[0] <= max_idx and 0 <= entity[1] <= max_idx:
            valid_ner.append(tuple(entity))
        else:
            print(f"Invalid NER entity in 'tokens': {entity} (tokens length: {len(tokens)})")

    # 验证 rel
    valid_rel = []
    for relation in data["rel"]:
        if relation[0] < len(valid_ner) and relation[1] < len(valid_ner):
            valid_rel.append(tuple(relation))
        else:
            print(f"Invalid relation in 'ner': {relation} (ner length: {len(valid_ner)})")

    # 更新数据
    data["ner"] = valid_ner
    data["rel"] = valid_rel
    return data


def process_dataset(dataset):
    """
    处理 train、dev、test 数据集
    """
    return [validate_and_convert(entry) for entry in dataset]


def json_to_pkl(input_file, output_file):
    """
    将 JSON 文件转成 .pkl 文件
    """
    # 读取 JSON 数据
    with open(input_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # 转换 train、dev、test 数据
    json_data["train"] = process_dataset(json_data["train"])
    json_data["dev"] = process_dataset(json_data["dev"])
    json_data["test"] = process_dataset(json_data["test"])

    # 保存为 .pkl 文件
    with open(output_file, "wb") as f:
        pickle.dump(json_data, f)
    print(f"Data successfully converted and saved to {output_file}")


# 执行转换
json_to_pkl(input_file, output_file)

"""============================================================================================================="""
# import json
# import pickle
#
# # 输入 JSON 文件路径
# json_file = 'dataset/CQI/cqi.json'
# # 输出 PKL 文件路径
# pkl_file = 'dataset/cqi.pkl'
#
# try:
#     # 读取 JSON 文件
#     with open(json_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#
#     # 写入 PKL 文件
#     with open(pkl_file, 'wb') as f:
#         pickle.dump(data, f)
#
#     print(f"JSON 文件已成功转换为 PKL 文件，保存到：{pkl_file}")
#
# except FileNotFoundError:
#     print(f"文件未找到：{json_file}")
# except json.JSONDecodeError:
#     print(f"无法解析 JSON 文件，请检查文件格式：{json_file}")
# except Exception as e:
#     print(f"发生错误：{e}")


