!pip install -q -U fastapi uvicorn pyngrok transformers peft bitsandbytes accelerate
!pip install -q -U transformers datasets peft trl bitsandbytes accelerate
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pyngrok import ngrok
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

# ==========================================
# 1. Custom Adapter Definitions (Required for Loading)
# ==========================================
from google.colab import drive
drive.mount('/content/drive')
class Safe_CVPR_Series_Adapter(nn.Module):
    def __init__(self, original_mlp: Qwen2MLP, config):
        super().__init__()
        self.original_mlp = original_mlp
        self.hidden_size = config.hidden_size

        self.cvpr_down_proj = nn.Linear(self.hidden_size, self.hidden_size // 4, bias=False)
        self.cvpr_dwconv = nn.Conv1d(
            in_channels=self.hidden_size // 4, out_channels=self.hidden_size // 4,
            kernel_size=3, padding=1, groups=self.hidden_size // 4
        )
        self.cvpr_up_proj = nn.Linear(self.hidden_size // 4, self.hidden_size, bias=False)

    def forward(self, hidden_states):
        x = self.original_mlp(hidden_states)
        if getattr(self.cvpr_down_proj, "disable_adapters", False):
            return x

        adapter_dtype = getattr(self.cvpr_down_proj, "weight", self.cvpr_down_proj).dtype
        cvpr_in = x.to(adapter_dtype)
        cvpr_out = self.cvpr_down_proj(cvpr_in)
        cvpr_out = cvpr_out.transpose(1, 2).contiguous()
        cvpr_out = self.cvpr_dwconv(cvpr_out)
        cvpr_out = F.gelu(cvpr_out)
        cvpr_out = cvpr_out.transpose(1, 2).contiguous()
        cvpr_out = self.cvpr_up_proj(cvpr_out)

        return x + cvpr_out.to(x.dtype)

class Safe_CVPR_Attention_Adapter(nn.Module):
    def __init__(self, original_attn, config):
        super().__init__()
        self.original_attn = original_attn
        self.hidden_size = config.hidden_size

        self.cvpr_down_proj = nn.Linear(self.hidden_size, self.hidden_size // 4, bias=False)
        self.cvpr_dwconv = nn.Conv1d(
            in_channels=self.hidden_size // 4, out_channels=self.hidden_size // 4,
            kernel_size=3, padding=1, groups=self.hidden_size // 4
        )
        self.cvpr_up_proj = nn.Linear(self.hidden_size // 4, self.hidden_size, bias=False)

    def forward(self, hidden_states, *args, **kwargs):
        attn_outputs = self.original_attn(hidden_states, *args, **kwargs)
        attn_hidden = attn_outputs[0]

        if getattr(self.cvpr_down_proj, "disable_adapters", False):
            return attn_outputs

        adapter_dtype = getattr(self.cvpr_down_proj, "weight", self.cvpr_down_proj).dtype
        cvpr_in = attn_hidden.to(adapter_dtype)
        cvpr_out = self.cvpr_down_proj(cvpr_in)
        cvpr_out = cvpr_out.transpose(1, 2).contiguous()
        cvpr_out = self.cvpr_dwconv(cvpr_out)
        cvpr_out = F.gelu(cvpr_out)
        cvpr_out = cvpr_out.transpose(1, 2).contiguous()
        cvpr_out = self.cvpr_up_proj(cvpr_out)

        new_attn_hidden = attn_hidden + cvpr_out.to(attn_hidden.dtype)
        return (new_attn_hidden,) + attn_outputs[1:]


# ==========================================
# 2. Model Loading & Injection
# ==========================================
print("Loading model for inference...")

# thre remaining code is hidden

# Start Uvicorn
config = uvicorn.Config(app, host="0.0.0.0", port=8000)
server = uvicorn.Server(config)
await server.serve()  # <--- This cooperates with Colab's background loop!
