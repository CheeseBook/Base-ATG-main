from transformers import AutoTokenizer, AutoModelForMaskedLM

# 下载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-chinese")

# 保存模型和分词器到本地
save_directory = "/model/bert-base-chinese"  # 可以修改为你希望保存的目录
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"模型和分词器已保存至 {save_directory}")
