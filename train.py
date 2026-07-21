# HF NETWORK RESILIENCY PATCH (Must be at the absolute top!)
import os  
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

import re
import subprocess
import tempfile
import urllib.request
import shutil
from concurrent.futures import ThreadPoolExecutor

# unsloth must be imported before trl/transformers/peft
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer


# 0. SYSTEM PRE-FLIGHT CHECKS
LUAU_ANALYZE_BIN = "luau-lsp"
if not shutil.which(LUAU_ANALYZE_BIN):
    raise FileNotFoundError(f"[CRITICAL] '{LUAU_ANALYZE_BIN}' not found. Please check your PATH.")

# FIX: local filename now matches the source file's real extension (.d.luau, not .d.lua)
ROBLOX_DEFS_PATH = os.path.abspath("globalTypes.d.luau")
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
print("🧠 QUALITY OVERRIDE: Loading native unquantized weights...")

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
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# INCREASED TEMPERATURE: Forces the model to explore creative algorithmic solutions 
# rather than defaulting to the most basic, lazy code loops.
model.generation_config.temperature = 0.85 
model.generation_config.do_sample = True

# 2. DATASET STREAMING
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

SYSTEM_PROMPT = (
    "You are a helpful Roblox Luau assistant named LuauBot made by divinerblx. You must map out "
    "your logic, constraints, and thoughts step-by-step inside "
    "<think> tags. Once your thinking is done, write your final code inside a "
    "```lua code block. Do NOT provide prose or explanations outside these blocks."
    "Do NOT write multiple code blocks to explain your code."
)

def format_conversational(example):
    raw_prompt = example["prompt"]
    match = re.search(r'Task:\s*\n(.*)', raw_prompt, re.DOTALL)
    task_only = match.group(1).strip() if match else raw_prompt.strip()

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_only}
        ]
    }

# FIX: compute the sanity stat from the raw dataset BEFORE overwriting it with
# format_conversational's output, instead of loading the jsonl file a second time.
_matched = sum(
    1 for ex in dataset if re.search(r'Task:\s*\n', ex["prompt"], re.DOTALL)
)
_total = len(dataset)

dataset = dataset.map(format_conversational, num_proc=os.cpu_count() or 1)

print(f"[INFO] Boilerplate stripped cleanly for {_matched}/{_total} rows.")


# 3. EXTRACTION & REWARDS
def safe_get_text(completion):
    if isinstance(completion, list) and len(completion) > 0:
        return completion[-1].get("content", "")
    return str(completion)

CODE_FENCE_RE = re.compile(r'```(?:lua|luau)\s*\n(.*?)```', re.DOTALL | re.IGNORECASE)

def strip_think_block(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text

def extract_luau_code(completion_text):
    clean_text = strip_think_block(completion_text)
    match = CODE_FENCE_RE.search(clean_text)
    if match:
        return match.group(1).strip()
    return ""

def code_fence_reward_func(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = strip_think_block(safe_get_text(completion))
        fence_count = len(CODE_FENCE_RE.findall(text))
        if fence_count == 1:
            rewards.append(1.0)
        elif fence_count > 1:
            rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards

def think_format_reward_func(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = safe_get_text(completion)
        match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
        # PENALIZE LAZY THINKING: Requires at least 150 characters of thought logic.
        if match and len(match.group(1).strip()) >= 150 and re.search(r'[a-zA-Z0-9]', match.group(1)):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def _evaluate_single_completion(completion):
    """Runs the disk-write + luau-lsp subprocess check for exactly one completion.
    Pulled out of luau_syntax_reward_func so it can be fanned out across threads —
    subprocess.run() releases the GIL while blocked on the child process, so
    ThreadPoolExecutor gives real concurrency here without needing multiprocessing."""
    code = extract_luau_code(safe_get_text(completion))
    code_clean = re.sub(r'--.*', '', code).strip()

    if len(code_clean) < 15 or not re.search(r'\b(local|function|if|for|while|return|table\.|task\.)\b', code_clean):
        return 0.0

    is_trivial = len(code_clean.split('\n')) < 3 and not ("function" in code_clean or "if" in code_clean)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.luau', delete=False, encoding='utf-8') as f:
        f.write(code)
        path = f.name

    try:
        res = subprocess.run(
            [LUAU_ANALYZE_BIN, "analyze", f"--definitions:@roblox={ROBLOX_DEFS_PATH}",
             "--no-strict-dm-types", path],
            capture_output=True, text=True, timeout=5,
        )
        if res.returncode == 0:
            return 0.2 if is_trivial else 1.0
        else:
            combined_output = res.stdout + res.stderr
            diagnostic_lines = combined_output.split('\n')
            # FIX: match the phrasing luau-lsp actually emits ("Unused function
            # parameter", "unused local", etc.) instead of the concatenated,
            # never-matching "functionunused" substring.
            real_issues = [
                line for line in diagnostic_lines
                if re.search(r'\(\d+,\d+\):', line)
                and "unknown global" not in line.lower()
                and "unused" not in line.lower()
            ]
            issue_count = len(real_issues)

            continuous_score = max(0.0, 0.6 - 0.08 * issue_count)
            if is_trivial:
                # FIX: a trivial script that still fails static analysis must
                # never outscore a trivial script that passes cleanly (0.2 in
                # the res.returncode == 0 branch above). Halving alone wasn't
                # enough to guarantee that at low issue_count, so clamp here.
                continuous_score = min(continuous_score * 0.5, 0.15)
            return round(continuous_score, 4)
    except Exception as e:
        print(f"[WARN] luau-lsp invocation failed: {e}")
        return 0.0
    finally:
        if os.path.exists(path):
            os.remove(path)


def luau_syntax_reward_func(completions, **kwargs):
    # PARALLELIZED: each completion's tempfile-write + luau-lsp subprocess call
    # is independent I/O-bound work, so run them concurrently across threads
    # instead of blocking the GPU step on 16 sequential shell-outs.
    max_workers = min(len(completions), os.cpu_count() or 4) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rewards = list(executor.map(_evaluate_single_completion, completions))
    return rewards

def no_explanation_reward_func(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = safe_get_text(completion)
        clean_text = strip_think_block(text)

        fence_match = CODE_FENCE_RE.search(clean_text)
        outside_len = 0
        
        if fence_match:
            before = clean_text[:fence_match.start()].strip()
            after = clean_text[fence_match.end():].strip()
            outside_len = len(before) + len(after)
        else:
            outside_len = len(clean_text.strip())

        if outside_len <= 25:
            explanation_penalty = 0.0
        else:
            explanation_penalty = min(1.0, (outside_len - 25) / 200)

        code = extract_luau_code(text)
        lines = code.strip().split('\n') if code else []
        comment_lines = sum(1 for line in lines if line.strip().startswith('--'))
        
        comment_penalty = 0.0
        if len(lines) > 4:
            comment_ratio = comment_lines / len(lines)
            if comment_ratio > 0.55:
                comment_penalty = 0.5

        total_penalty = max(-1.0, -(explanation_penalty + comment_penalty))
        rewards.append(round(total_penalty, 4))
        
    return rewards

# 4. CONFIGURATION

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
