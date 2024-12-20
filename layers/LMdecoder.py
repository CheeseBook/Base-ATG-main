import torch
from transformers import LlamaForCausalLM, LlamaConfig

class CustomDecoder(torch.nn.Module):
    def __init__(self, llama_model_name):
        super().__init__()
        config = LlamaConfig()
        # 加载 LLaMA 模型
        self.llama = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype="auto")

    def forward(self, x, memory, causal_mask=None, memory_pad_mask=None):
        """
        x: 解码器输入，(batch_size, tgt_len, embed_dim)
        memory: Encoder 输出，(batch_size, src_len, embed_dim)
        causal_mask: 解码器的自回归掩码
        memory_pad_mask: Memory 的 padding 掩码
        """
        x = x.to(torch.bfloat16)
        # 设置 cross-attention 使用的 memory
        llama_inputs = {
            "inputs_embeds": x,  # 解码器输入
            "encoder_hidden_states": memory,  # Encoder 输出
            "attention_mask": memory_pad_mask,  # Memory 的 Padding 掩码
        }

        # LLaMA 前向传播
        outputs = self.llama(**llama_inputs)

        # 返回最后的 hidden states
        return outputs.last_hidden_state
