# import json
#
#
# def convert_format(data):
#     # 用于转换每条元素的函数
#     converted_data = []
#
#     for item in data:
#         # 转换 tokens
#         tokens = item['tokens']
#
#         # 转换 entities 格式
#         ner = [[entity['start'], entity['end'], entity['type']] for entity in item['entities']]
#
#         # 转换 relations 格式
#         rel = [[relation['head'], relation['tail'], relation['type']] for relation in item['relations']]
#
#         # 格式化成新的数据结构
#         converted_item = {
#             "tokens": tokens,
#             "ner": ner,
#             "rel": rel,
#             "seq": []  # seq 空数组
#         }
#
#         converted_data.append(converted_item)
#
#     return converted_data
#
#
# def main():
#     # 假设数据存储在json文件中，首先加载文件
#     input_file = 'dataset/SQI/sqi.json'  # 原始数据文件路径
#     output_file = 'dataset/SQI/processed_sqi.json'  # 输出文件路径
#
#     with open(input_file, 'r', encoding='utf-8') as infile:
#         data = json.load(infile)
#
#     # 转换 train, dev, test 数据
#     converted_train = convert_format(data['train'])
#     converted_dev = convert_format(data['dev'])
#     converted_test = convert_format(data['test'])
#
#     # 保存转换后的数据
#     converted_json = {
#         "train": converted_train,
#         "dev": converted_dev,
#         "test": converted_test
#     }
#
#     with open(output_file, 'w', encoding='utf-8') as outfile:
#         json.dump(converted_json, outfile, ensure_ascii=False, indent=4)
#
#     print(f"数据已成功转换并保存到 {output_file}")
#
#
# if __name__ == '__main__':
#     main()


import json
import pickle

def convert_ner_and_rel(data):
    """
    Convert ner and rel elements to tuple format.
    """
    for split in ['train', 'dev', 'test']:
        for item in data[split]:
            # Convert ner from list to tuple
            item['ner'] = [tuple(ner) for ner in item['ner']]
            # Convert rel from list to tuple
            item['rel'] = [tuple(rel) for rel in item['rel']]
    return data

def save_to_pkl(data, output_file):
    """ Save the processed data to a .pkl file """
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

def load_json(input_file):
    """ Load the JSON file """
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # Specify the input JSON and output .pkl file paths
    input_file = 'dataset/SQI/processed_sqi.json'  # Input JSON file
    output_file = 'dataset/sqi.pkl'  # Output .pkl file

    # Load the JSON data
    data = load_json(input_file)

    # Convert 'ner' and 'rel' to tuple format
    data = convert_ner_and_rel(data)

    # Save the data as .pkl
    save_to_pkl(data, output_file)
    print(f"Data has been successfully saved to {output_file}")

if __name__ == "__main__":
    main()



