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
base_model_path = "/content/drive/MyDrive/git_pilot/qwen7b_full"
peft_model_path = "/content/drive/MyDrive/git_pilot/qwen-cvpr-series-dpo-final"

tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# Re-inject empty adapters into the middle layer
model_layers = base_model.model.layers
target_layer_idx = len(model_layers) // 2
target_layer = model_layers[target_layer_idx]

orig_mlp = target_layer.mlp
target_layer.mlp = Safe_CVPR_Series_Adapter(orig_mlp, base_model.config).to(next(orig_mlp.parameters()).device)

orig_attn = target_layer.self_attn
target_layer.self_attn = Safe_CVPR_Attention_Adapter(orig_attn, base_model.config).to(next(orig_attn.parameters()).device)

# Load PEFT weights (This maps saved LoRA AND CVPR adapter weights into the injected model)
model = PeftModel.from_pretrained(base_model, peft_model_path)
model.eval()
print("Model loaded successfully!")

# ==========================================
# 3. Inference Function
# ==========================================
def generate_response(prompt_text: str) -> str:
    # 1. ADD A SYSTEM PROMPT to force JSON behavior
    messages = [
        {
            "role": "system",
            "content": "You are a strict API bot. You must translate the user's request into a valid JSON array of action objects. Output ONLY the raw JSON. Do not include greetings, explanations, or markdown code blocks (no ```json). "
        },
        {
            "role": "user",
            "content": prompt_text
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,         # 2. LOWER TEMPERATURE to make it deterministic
            top_p=0.95,
            repetition_penalty=1.0,   # Keep at 1.0 for JSON (higher values break JSON syntax)
            do_sample=True,           # Must be True to use temperature, but 0.01 makes it effectively greedy
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

# @title
github_prompts = [
    # 1. create repo
    "Create a new public repository named 'ecommerce-frontend'.",
    "Set up a new repository called 'data-pipeline' with a brief description 'Data processing scripts'.",

    # 2. update_repo_visibility
    "Change the visibility of my 'ecommerce-frontend' repository from public to private.",
    "Make the 'internal-tools' repo public so everyone can see it.",

    # 3. add_file
    "Add a new file named `index.js` to the 'ecommerce-frontend' repository.",
    "Create a `README.md` file in the 'data-pipeline' repo with the text 'Welcome to the project'.",

    # 4. create_branch
    "Create a new branch called 'feature/shopping-cart' from the 'main' branch in the 'ecommerce-frontend' repo.",
    "Branch off of 'develop' to create a new branch named 'bugfix/login-error'.",

    # 5. update_file
    "Update the `package.json` file in the 'ecommerce-frontend' repository to change the version number to 2.0.",
    "Modify the `config.yml` file in the 'data-pipeline' repo to update the database connection string.",

    # 6. add_collaborator
    "Invite the GitHub user 'johndoe99' as a collaborator to the 'ecommerce-frontend' repository.",
    "Add 'sarah-smith' with write access to my 'data-pipeline' repo.",

    # 7. remove_collaborator
    "Remove the user 'johndoe99' from the collaborators list on the 'ecommerce-frontend' repository.",
    "Revoke access for 'alex-dev' from the 'data-pipeline' repo.",

    # 8. delete_repo
    "Delete the repository named 'old-legacy-project'.",
    "Permanently remove the 'test-repo-123' repository from my account.",

    # 9. fork_repo
    "Fork the 'facebook/react' repository into my account.",
    "Create a fork of the 'torvalds/linux' repository.",

    # 10. star_repo
    "Star the 'vercel/next.js' repository.",
    "Add a star to 'hwchase17/langchain'.",

    # 11. delete_branch
    "Delete the branch named 'bugfix/login-error' from the 'ecommerce-frontend' repository.",
    "Remove the 'stale-feature' branch from the 'data-pipeline' repo.",

    # 12. create_issue
    "Create a new issue in the 'ecommerce-frontend' repo titled 'App crashes on mobile Safari' with the description 'The checkout button is unresponsive on iOS devices'.",
    "Open an issue in 'data-pipeline' named 'Missing API keys' and assign it to me.",

    # 13. close_issue
    "Close issue #42 in the 'ecommerce-frontend' repository.",
    "Mark issue #15 in the 'data-pipeline' repo as closed.",

    # 14. create_pr
    "Create a pull request in the 'ecommerce-frontend' repo to merge the 'feature/shopping-cart' branch into 'main' with the title 'Add new cart UI'.",
    "Open a PR merging 'bugfix/typo' into the 'develop' branch of the 'data-pipeline' repository.",

    # Bonus: Combined/Multi-Action Prompts
    "Create a new repository named 'my-blog', make it private, and add 'alice-dev' as a collaborator.",
    "Create a new branch called 'docs-update', add a file named `CONTRIBUTING.md` to it, and then create a PR to merge it into main."
]

for p in github_prompts:
  print("-prompt:",p)
  print("--response:",generate_response(p))



# @title
import json

# ==========================================
# 6. Direct Function Test & JSON Evaluation
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🧪 TESTING & EVALUATING STRUCTURED JSON OUTPUTS")
    print("="*50 + "\n")

    # Load the 10 samples (You can load this from a .json file instead if preferred)
    test_data = [
      {
        "input": "add a secret called AWS_ACCESS_KEY to the user-service repo and then trigger the deploy-prod workflow.",
        "output": [
          { "step": 1, "action": "add_repo_secret", "parameters": { "repo_name": "user-service", "secret_name": "AWS_ACCESS_KEY" } },
          { "step": 2, "action": "trigger_workflow", "parameters": { "repo_name": "user-service", "workflow_name": "deploy-prod workflow" } }
        ]
      },
      {
        "input": "enable branch protection on the main branch of payment-api and request a review from the security-team for pull request 104.",
        "output": [
          { "step": 1, "action": "enable_branch_protection", "parameters": { "repo_name": "payment-api", "branch": "main" } },
          { "step": 2, "action": "request_pr_review", "parameters": { "repo_name": "payment-api", "pr_number": "104", "reviewer": "security-team" } }
        ]
      },
      {
        "input": "draft a new release v2.0.0 in the order-backend repo using the main branch and publish it.",
        "output": [
          { "step": 1, "action": "create_release_draft", "parameters": { "repo_name": "order-backend", "tag_name": "v2.0.0", "target_branch": "main" } },
          { "step": 2, "action": "publish_release", "parameters": { "repo_name": "order-backend", "tag_name": "v2.0.0" } }
        ]
      },
      {
        "input": "trigger the rollback workflow in the inventory-db repo and pass the target_version parameter as v1.5.2.",
        "output": [
          { "step": 1, "action": "trigger_workflow_dispatch", "parameters": { "repo_name": "inventory-db", "workflow_name": "rollback", "input_target_version": "v1.5.2" } }
        ]
      },
      {
        "input": "create an issue about the database connection pool timeout in the core-engine repo, add the bug label, and assign it to the Q2-backend milestone.",
        "output": [
          { "step": 1, "action": "create_issue", "parameters": { "repo_name": "core-engine", "title": "database connection pool timeout" } },
          { "step": 2, "action": "add_issue_label", "parameters": { "repo_name": "core-engine", "label": "bug" } },
          { "step": 3, "action": "assign_milestone", "parameters": { "repo_name": "core-engine", "milestone": "Q2-backend" } }
        ]
      },
      {
        "input": "merge pull request 45 in the auth-gateway repo and delete the feature/oauth branch afterward.",
        "output": [
          { "step": 1, "action": "merge_pr", "parameters": { "repo_name": "auth-gateway", "pr_number": "45" } },
          { "step": 2, "action": "delete_branch", "parameters": { "repo_name": "auth-gateway", "branch": "feature/oauth" } }
        ]
      },
      {
        "input": "create a deployment environment called staging in the notification-service repo and add a required reviewer ops-lead.",
        "output": [
          { "step": 1, "action": "create_environment", "parameters": { "repo_name": "notification-service", "environment_name": "staging" } },
          { "step": 2, "action": "add_environment_reviewer", "parameters": { "repo_name": "notification-service", "environment_name": "staging", "reviewer": "ops-lead" } }
        ]
      },
      {
        "input": "add a webhook to the billing-system repo pointing to https://api.monitor.com/hook and configure it to listen for push events.",
        "output": [
          { "step": 1, "action": "create_webhook", "parameters": { "repo_name": "billing-system", "payload_url": "https://api.monitor.com/hook", "events": "push" } }
        ]
      },
      {
        "input": "set a repository variable called LOG_LEVEL to DEBUG in the search-api repo and restart the active workflow runs.",
        "output": [
          { "step": 1, "action": "set_repo_variable", "parameters": { "repo_name": "search-api", "variable_name": "LOG_LEVEL", "variable_value": "DEBUG" } },
          { "step": 2, "action": "restart_workflow_runs", "parameters": { "repo_name": "search-api" } }
        ]
      },
      {
        "input": "delete all github actions caches in the media-storage repo and cancel workflow run 8901.",
        "output": [
          { "step": 1, "action": "delete_actions_cache", "parameters": { "repo_name": "media-storage" } },
          { "step": 2, "action": "cancel_workflow_run", "parameters": { "repo_name": "media-storage", "run_id": "8901" } }
        ]
      }
    ]

    total_actions = 0
    correct_actions = 0
    perfect_sequences = 0

    for i, data in enumerate(test_data):
        print(f"\n--- Test {i+1}/10 ---")
        print(f"🗣️ PROMPT: {data['input']}")

        # 1. Get raw string response from your LLM
        reply_str = generate_response(data['input'])

        # 2. Parse the LLM's string into a Python list/dict
        try:
            # Try to strip out any markdown code blocks the model might wrap the JSON in
            clean_reply = reply_str.replace("```json", "").replace("```", "").strip()
            generated_json = json.loads(clean_reply)
        except json.JSONDecodeError:
            print("❌ MODEL FAILED TO OUTPUT VALID JSON")
            print(f"RAW OUTPUT:\n{reply_str}")
            continue

        reference_json = data['output']

        # Print for visual inspection
        print(f"🤖 PARSED MODEL OUTPUT:\n{json.dumps(generated_json, indent=2)}")

        # 3. Calculate Action Step Accuracy
        is_perfect_sequence = True

        # Check if length matches
        if len(generated_json) != len(reference_json):
            is_perfect_sequence = False

        for step_idx, ref_step in enumerate(reference_json):
            total_actions += 1

            # If the model didn't generate enough steps
            if step_idx >= len(generated_json):
                is_perfect_sequence = False
                continue

            gen_step = generated_json[step_idx]

            # Check if Action string matches
            action_matches = gen_step.get("action") == ref_step.get("action")

            # Check if Parameters match (subset matching)
            ref_params = ref_step.get("parameters", {})
            gen_params = gen_step.get("parameters", {})

            params_match = True
            for key, val in ref_params.items():
                if key not in gen_params or gen_params[key] != val:
                    params_match = False

            if action_matches and params_match:
                correct_actions += 1
            else:
                is_perfect_sequence = False

        if is_perfect_sequence:
            perfect_sequences += 1
            print("✅ Evaluation: Perfect Sequence Match")
        else:
            print("⚠️ Evaluation: Partial or Failed Match")

    # Final Metrics Output
    print("\n" + "="*50)
    print("📈 EVALUATION RESULTS")
    print("="*50)

    action_accuracy = (correct_actions / total_actions) * 100 if total_actions > 0 else 0
    sequence_accuracy = (perfect_sequences / len(test_data)) * 100

    print(f"🎯 Action Extraction Accuracy: {action_accuracy:.2f}% ({correct_actions}/{total_actions} steps fully matched)")
    print(f"🏆 Perfect Sequence Accuracy:  {sequence_accuracy:.2f}% ({perfect_sequences}/{len(test_data)} prompts flawlessly executed)")



!pip install -q pycloudflared

# ==========================================
# 4. FastAPI & Cloudflare Setup (Ngrok Alternative)
# ==========================================
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import json
import uvicorn
from pycloudflared import try_cloudflare # <--- Swapped Ngrok for Cloudflare

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Prompt(BaseModel):
    text: str

class Feedback(BaseModel):
    prompt: str
    response: str
    is_good: bool

@app.post("/generate")
def generate_text(prompt: Prompt):
    try:
        reply = generate_response(prompt.text)
        return {"response": reply}
    except Exception as e:
        return {"error": str(e)}

@app.post("/feedback")
def collect_feedback(feedback: Feedback):
    try:
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": feedback.prompt,
            "response": feedback.response,
            "is_good": feedback.is_good
        }
        with open("continuous_dpo_data.jsonl", "a") as f:
            f.write(json.dumps(feedback_entry) + "\n")
        return {"status": "success", "message": "Feedback recorded for next DPO run!"}
    except Exception as e:
        return {"error": str(e)}

# Start Cloudflare tunnel (No auth token required!)
tunnel = try_cloudflare(port=8000)

print("\n" + "="*50)
print(f"🚀 API is live at: {tunnel.tunnel}/generate")
print("="*50 + "\n")

# Start Uvicorn
config = uvicorn.Config(app, host="0.0.0.0", port=8000)
server = uvicorn.Server(config)
await server.serve()  # <--- This cooperates with Colab's background loop!
