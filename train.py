# HF NETWORK RESILIENCY PATCH (Must be at the absolute top!)
import os  
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

import re
import urllib.request
import shutil

# unsloth must be imported before trl/transformers/peft
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

# Text-processing helpers and reward functions live in a dependency-light,
# side-effect-free module so they can be unit tested in isolation.
from rewards import (
    LUAU_ANALYZE_BIN,
    ROBLOX_DEFS_PATH,
    SYSTEM_PROMPT,
    format_conversational,
    code_fence_reward_func,
    think_format_reward_func,
    luau_syntax_reward_func,
    no_explanation_reward_func,
)


# 0. SYSTEM PRE-FLIGHT CHECKS
if not shutil.which(LUAU_ANALYZE_BIN):
    raise FileNotFoundError(f"[CRITICAL] '{LUAU_ANALYZE_BIN}' not found. Please check your PATH.")

# FIX: local filename now matches the source file's real extension (.d.luau, not .d.lua)
ROBLOX_DEFS_URL = "https://raw.githubusercontent.com/JohnnyMorganz/luau-lsp/master/scripts/globalTypes.d.luau"

if not os.path.exists(ROBLOX_DEFS_PATH):
    print(f"{ROBLOX_DEFS_PATH} not found, downloading official Roblox definitions...")
    try:
        urllib.request.urlretrieve(ROBLOX_DEFS_URL, ROBLOX_DEFS_PATH)
    except Exception as e:
        raise RuntimeError(f"[CRITICAL] Could not download definitions: {e}")
else:
    print(f"[INFO] Using Roblox definitions at: {ROBLOX_DEFS_PATH}")

torch.cuda.empty_cache()

# 1. INITIALIZE MODEL
max_seq_length = 4096  
base_model_name = "luau_grpo_final_clean"

print(f"[INFO] Loading model: {base_model_name}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=False,  
)

print("🧠 QUALITY OVERRIDE: Expanding LoRA Rank to 128 for ultra-dense logic retention...")
model = FastLanguageModel.get_peft_model(
    model,
    r=128,               # MAXIMUM QUALITY: Massive parameter footprint for capturing deep Luau nuances
    lora_alpha=128,      
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ],                   
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
)

# INCREASED TEMPERATURE: Forces the model to explore creative algorithmic solutions 
# rather than defaulting to the most basic, lazy code loops.
model.generation_config.temperature = 0.85 
model.generation_config.do_sample = True

# 2. DATASET STREAMING
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

# FIX: compute the sanity stat from the raw dataset BEFORE overwriting it with
# format_conversational's output, instead of loading the jsonl file a second time.
_matched = sum(
    1 for ex in dataset if re.search(r'Task:\s*\n', ex["prompt"], re.DOTALL)
)
_total = len(dataset)

dataset = dataset.map(format_conversational, num_proc=os.cpu_count() or 1)

print(f"[INFO] Boilerplate stripped cleanly for {_matched}/{_total} rows.")


# 3. CONFIGURATION

print("SETTING UP SETTINGS...")
training_args = GRPOConfig(
    output_dir="luau_grpo_outputs_continued",
    
    # QUALITY FOCUS: Lower learning rate with a cosine decay curve. 
    # Slower, smoother gradient updates prevent the model from "forgetting" its base intelligence.
    learning_rate=2e-6,
    lr_scheduler_type="cosine", 
    warmup_ratio=0.05,
    
    # KL PENALTY: Ensures the model doesn't destroy its natural language just to hack the reward score.
    beta=0.08, 
    
    num_generations=16,          
    generation_batch_size=16,    
    
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    dataloader_drop_last=True,
    logging_steps=1,
    max_steps=1800, # Extended training horizon to let the cosine scheduler breathe
    save_steps=100,
    
    max_completion_length=2048, 
    
    report_to="none",
    temperature=0.85, # Encourages diverse exploration
    top_p=0.9,
    repetition_penalty=1.05,
    
    reward_weights=[1.0, 1.0, 1.5, 0.5], 
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[code_fence_reward_func, think_format_reward_func, luau_syntax_reward_func, no_explanation_reward_func],
    args=training_args,
    train_dataset=dataset,
)

# ==========================================
# PIPELINE SANITY CHECK
# ==========================================
print("\n" + "="*50)
print("PIPELINE SANITY CHECK")
print("="*50)
print(f"Dataset size: {len(dataset)} examples")
print(f"Roblox defs path exists: {os.path.exists(ROBLOX_DEFS_PATH)}")

# FIX: switch into Unsloth's inference mode before generating. Without this,
# gradient checkpointing (enabled above via use_gradient_checkpointing="unsloth")
# is incompatible with KV-caching during generate(), which either errors out or
# silently forces a slow/uncached generation that isn't representative of real
# inference quality.
FastLanguageModel.for_inference(model)

_test_inputs = tokenizer.apply_chat_template(
    dataset[0]["prompt"], tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to(model.device)
_test_output = model.generate(_test_inputs, max_new_tokens=512, do_sample=True, temperature=0.85)
_test_text = tokenizer.decode(_test_output[0][_test_inputs.shape[1]:], skip_special_tokens=True)
print(f"\n[SANITY] code_fence reward: {code_fence_reward_func([[{'role': 'assistant', 'content': _test_text}]])}")
print(f"[SANITY] no_explanation reward: {no_explanation_reward_func([[{'role': 'assistant', 'content': _test_text}]])}")
print("="*50 + "\n")

# FIX: switch back into training mode before trainer.train() picks the model
# back up — otherwise training resumes with the model still configured for
# inference (cache enabled, checkpointing effectively bypassed).
FastLanguageModel.for_training(model)

# ==========================================
# START TRAINING FROM CHECKPOINT
# ==========================================
checkpoint_path = None
if os.path.exists(training_args.output_dir):
    checkpoints = [
        os.path.join(training_args.output_dir, d) 
        for d in os.listdir(training_args.output_dir) 
        if d.startswith("checkpoint-") and d.split("-")[-1].isdigit()
    ]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
        checkpoint_path = checkpoints[-1]

if checkpoint_path:
    print(f"[INFO] Resuming GRPO Alignment Phase from latest checkpoint: {checkpoint_path}...")
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    print(f"[INFO] Starting GRPO training using: {base_model_name}...")
    trainer.train()

model.save_pretrained_merged("luau_grpo_final_clean_continued", tokenizer, save_method="merged_16bit")
print("[SUCCESS] Model successfully compiled and saved.")
