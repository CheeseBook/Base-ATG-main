from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
#
# MODELS = {
#     "spanbert": f"/gpfswork/rech/pds/upa43yu/models/spanbert-base-cased",
#     "bert": f"/gpfswork/rech/pds/upa43yu/models/bert-base-cased",
#     "roberta": f"/gpfswork/rech/pds/upa43yu/models/roberta-base",
#     "scibert": f"/gpfswork/rech/pds/upa43yu/models/scibert-base",
#     "arabert": f"/gpfswork/rech/pds/upa43yu/models/bert-base-arabert",
#     "bertlarge": f"/gpfsdswork/dataset/HuggingFace_Models/bert-large-cased",
#     "scibert_cased": f"/gpfswork/rech/pds/upa43yu/models/scibert_cased",
#     "albert": f"/gpfswork/rech/pds/upa43yu/models/albert-xxlarge-v2",
#     "spanbertlarge": f"/gpfswork/rech/pds/upa43yu/models/spanbert-large-cased",
#     "t5-s": "/gpfsdswork/dataset/HuggingFace_Models/t5-small",
#     "t5-m": "/gpfsdswork/dataset/HuggingFace_Models/t5-base",
#     "t5-l": "/gpfsdswork/dataset/HuggingFace_Models/t5-large",
#     "deberta": "/gpfswork/rech/pds/upa43yu/models/deberta-v3-large",
# }
#
# model_name = "meta-llama/Llama-3.2-1B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# llama_decoder = AutoModelForCausalLM.from_pretrained(model_name)
# llama_decoder.save_pretrained("/models")
# tokenizer.save_pretrained("/models")

import torch
import torch.nn as nn

# Step 1: 加载 Llama 模型和 tokenizer
llama_model_name = "meta-llama/Meta-Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_decoder = AutoModelForCausalLM.from_pretrained(llama_model_name, device_map="auto")

# Step 2: 定义自己的模型
class CustomTransformer(nn.Module):
    def __init__(self, llama_decoder):
        super(CustomTransformer, self).__init__()
        # 假设已经有自己的 Encoder 部分
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8), num_layers=6
        )
        # 替换 Decoder 为 Llama 的解码器部分
        self.decoder = llama_decoder

    def forward(self, input_ids, encoder_hidden_states, attention_mask=None):
        # Encoder 前向传播
        encoder_outputs = self.encoder(encoder_hidden_states)

        # 调用 Llama 的解码器前向传播
        outputs = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_outputs,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return outputs

# Step 3: 初始化模型
custom_model = CustomTransformer(llama_decoder)

# Step 4: 测试输入
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom_model.to(device)

# 示例输入
input_text = "The quick brown fox"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
encoder_hidden_states = torch.rand((1, 10, 768)).to(device)  # 模拟 Encoder 输出

# Forward 传播
outputs = custom_model(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states)
decoded_text = tokenizer.decode(outputs.logits.argmax(dim=-1).squeeze().tolist())

print(f"Decoded text: {decoded_text}")
print(f"Decoded text: {decoded_text}")
print(f"Decoded text: {decoded_text}")
print(f"Decoded text: {decoded_text}")
