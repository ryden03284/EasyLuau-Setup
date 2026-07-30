"""Reward functions and text-processing helpers for the EasyLuau GRPO trainer.

These functions are deliberately free of heavy ML dependencies (unsloth, torch,
trl) and have no import-time side effects, so they can be imported and unit
tested in isolation. ``train.py`` imports everything it needs from here.
"""

import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

# Static-analysis configuration. ``train.py`` performs the actual pre-flight
# validation (binary present on PATH, definitions downloaded); importing this
# module never touches the filesystem or the network.
LUAU_ANALYZE_BIN = "luau-lsp"
ROBLOX_DEFS_PATH = os.path.abspath("globalTypes.d.luau")

SYSTEM_PROMPT = (
    "You are a helpful Roblox Luau assistant named LuauBot made by divinerblx. You must map out "
    "your logic, constraints, and thoughts step-by-step inside "
    "<think> tags. Once your thinking is done, write your final code inside a "
    "```lua code block. Do NOT provide prose or explanations outside these blocks."
    "Do NOT write multiple code blocks to explain your code."
)

CODE_FENCE_RE = re.compile(r'```(?:lua|luau)\s*\n(.*?)```', re.DOTALL | re.IGNORECASE)


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


def safe_get_text(completion):
    if isinstance(completion, list) and len(completion) > 0:
        return completion[-1].get("content", "")
    return str(completion)


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
