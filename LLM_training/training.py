"""
This file is used for loading the training data and training the LLM Model on it
"""
# installing necessary libraries
!pip install -q -U transformers datasets peft trl bitsandbytes accelerate
!pip install datasets
# importing libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from google.colab import drive
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
torch.cuda.empty_cache()
gc.collect()
print("Mounting Google Drive...")
drive.mount('/content/drive')
model_path = "/content/drive/MyDrive/git_pilot/qwen7b_full"
output_dir = "/content/drive/MyDrive/git_pilot/qwen-cvpr-series-dpo-final"
data_path = "/content/drive/MyDrive/dpo_training_data.json"
os.makedirs(output_dir, exist_ok=True)
# 1. CVPR SERIES Adapters (MLP and Attention)
class Safe_CVPR_Series_Adapter(nn.Module):
    def _init_(self, original_mlp: Qwen2MLP, config):
        super()._init_()
        self.original_mlp = original_mlp
        self.hidden_size = config.hidden_size

        self.cvpr_down_proj = nn.Linear(self.hidden_size, self.hidden_size // 4, bias=False)
        self.cvpr_dwconv = nn.Conv1d(
            in_channels=self.hidden_size // 4, out_channels=self.hidden_size // 4,
            kernel_size=3, padding=1, groups=self.hidden_size // 4
        )
        self.cvpr_up_proj = nn.Linear(self.hidden_size // 4, self.hidden_size, bias=False)
        nn.init.zeros_(self.cvpr_up_proj.weight)

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
    def _init_(self, original_attn, config):
        super()._init_()
        self.original_attn = original_attn
        self.hidden_size = config.hidden_size

        self.cvpr_down_proj = nn.Linear(self.hidden_size, self.hidden_size // 4, bias=False)
        self.cvpr_dwconv = nn.Conv1d(
            in_channels=self.hidden_size // 4, out_channels=self.hidden_size // 4,
            kernel_size=3, padding=1, groups=self.hidden_size // 4
        )
        self.cvpr_up_proj = nn.Linear(self.hidden_size // 4, self.hidden_size, bias=False)
        nn.init.zeros_(self.cvpr_up_proj.weight)

    def forward(self, hidden_states, *args, **kwargs):
        # Pass through frozen attention (Returns tuple: (attn_output, attn_weights, ...))
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
# 2. Model Loading & Quantization
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)
# 3. Inject ONE CVPR Attention and ONE MLP
print("Injecting CVPR adapters into the middle layer only...")
model_layers = model.model.layers
target_layer_idx = len(model_layers) // 2
target_layer = model_layers[target_layer_idx]
print(f"Targeting Layer {target_layer_idx}")

# 4. Inject MLP Adapter
orig_mlp = target_layer.mlp
mlp_device = next(orig_mlp.parameters()).device
target_layer.mlp = Safe_CVPR_Series_Adapter(orig_mlp, model.config).to(mlp_device)

# 5. Inject Attention Adapter
orig_attn = target_layer.self_attn
attn_device = next(orig_attn.parameters()).device
target_layer.self_attn = Safe_CVPR_Attention_Adapter(orig_attn, model.config).to(attn_device)

# 6. Prepare for k-bit training
model = prepare_model_for_kbit_training(model)
# 7. LoRA Setup
lora_config = LoraConfig(
    r=8,        
    lora_alpha=16,    
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=["cvpr_down_proj", "cvpr_dwconv", "cvpr_up_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.print_trainable_parameters()
# 8. Dataset Loading & Preprocessing
print("Loading dataset...")
raw_dataset = load_dataset("json", data_files=data_path)

def chat_format_map(examples):
    p = examples.get("prompt") or examples.get("instruction") or ""
    c = examples.get("chosen") or examples.get("output_good") or ""
    r = examples.get("rejected") or examples.get("output_bad") or ""
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        formatted_prompt = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"

    return {
        "prompt": formatted_prompt,
        "chosen": c + tokenizer.eos_token,
        "rejected": r + tokenizer.eos_token
    }
formatted_dataset = raw_dataset["train"].map(chat_format_map)
formatted_dataset = formatted_dataset.filter(lambda x: len(x["prompt"]) > 0 and len(x["chosen"]) > 0)

split_dataset = formatted_dataset.train_test_split(test_size=0.1, seed=42)
train_data = split_dataset["train"]
eval_data = split_dataset["test"]
# 9. DPO Configuration (No Mid-Run Eval)
training_args = DPOConfig(
    output_dir=output_dir,
    beta=0.1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=5e-5,
    num_train_epochs=1,        
    max_length=512,

    eval_strategy="no",        

    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=True,
    remove_unused_columns=False,
    optim="paged_adamw_8bit"
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    processing_class=tokenizer,
)
# 10. Execution
print("Starting CVPR-Series DPO training...")
dpo_trainer.train()

print("Saving results...")
# 11 Saves the LoRA adapters and the CVPR adapter weights 
dpo_trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Training complete!")