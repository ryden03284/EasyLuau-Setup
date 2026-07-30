"""Unit tests for the reward functions and helpers in ``rewards.py``.

``rewards.py`` holds all of the dependency-light, side-effect-free logic that
was previously buried inside ``train.py`` (which cannot be imported without a
GPU, unsloth, torch and trl). These tests give that logic real coverage.

The luau-lsp static-analysis path (``_evaluate_single_completion`` /
``luau_syntax_reward_func``) shells out to an external binary, so those tests
mock ``subprocess.run`` to exercise every scoring branch deterministically
without needing luau-lsp installed.
"""

from unittest import mock

import pytest

import rewards


def _assistant(text):
    """Wrap raw text the way GRPOTrainer hands completions to reward funcs."""
    return [{"role": "assistant", "content": text}]


# --------------------------------------------------------------------------
# format_conversational
# --------------------------------------------------------------------------

class TestFormatConversational:
    def test_extracts_text_after_task_marker(self):
        out = rewards.format_conversational({"prompt": "Boilerplate\nTask:\nDo the thing"})
        assert out["prompt"][0] == {"role": "system", "content": rewards.SYSTEM_PROMPT}
        assert out["prompt"][1] == {"role": "user", "content": "Do the thing"}

    def test_falls_back_to_whole_prompt_when_no_marker(self):
        out = rewards.format_conversational({"prompt": "  just a prompt  "})
        assert out["prompt"][1]["content"] == "just a prompt"

    def test_task_marker_captures_multiline_remainder(self):
        out = rewards.format_conversational({"prompt": "Task:\nline1\nline2"})
        assert out["prompt"][1]["content"] == "line1\nline2"

    def test_result_always_has_system_then_user(self):
        out = rewards.format_conversational({"prompt": "anything"})
        roles = [m["role"] for m in out["prompt"]]
        assert roles == ["system", "user"]


# --------------------------------------------------------------------------
# safe_get_text
# --------------------------------------------------------------------------

class TestSafeGetText:
    def test_returns_content_of_last_message(self):
        completion = [{"role": "system", "content": "x"}, {"role": "assistant", "content": "y"}]
        assert rewards.safe_get_text(completion) == "y"

    def test_missing_content_key_yields_empty_string(self):
        assert rewards.safe_get_text([{"role": "assistant"}]) == ""

    def test_empty_list_falls_back_to_str(self):
        assert rewards.safe_get_text([]) == "[]"

    def test_non_list_is_stringified(self):
        assert rewards.safe_get_text("hello") == "hello"
        assert rewards.safe_get_text(123) == "123"


# --------------------------------------------------------------------------
# strip_think_block
# --------------------------------------------------------------------------

class TestStripThinkBlock:
    def test_removes_closed_think_block(self):
        assert rewards.strip_think_block("<think>reasoning</think>code") == "code"

    def test_removes_unclosed_think_block_to_end(self):
        assert rewards.strip_think_block("before<think>dangling") == "before"

    def test_is_case_insensitive(self):
        assert rewards.strip_think_block("<THINK>x</THINK>done") == "done"

    def test_leaves_text_without_think_untouched(self):
        assert rewards.strip_think_block("plain text") == "plain text"


# --------------------------------------------------------------------------
# extract_luau_code
# --------------------------------------------------------------------------

class TestExtractLuauCode:
    def test_extracts_lua_fence(self):
        text = "```lua\nlocal x = 1\n```"
        assert rewards.extract_luau_code(text) == "local x = 1"

    def test_extracts_luau_fence(self):
        text = "```luau\nlocal y = 2\n```"
        assert rewards.extract_luau_code(text) == "local y = 2"

    def test_strips_think_block_before_extracting(self):
        text = "<think>```lua\nfake\n```</think>```lua\nreal\n```"
        assert rewards.extract_luau_code(text) == "real"

    def test_returns_empty_when_no_fence(self):
        assert rewards.extract_luau_code("no code here") == ""


# --------------------------------------------------------------------------
# code_fence_reward_func
# --------------------------------------------------------------------------

class TestCodeFenceReward:
    def test_single_fence_scores_one(self):
        assert rewards.code_fence_reward_func([_assistant("```lua\ncode\n```")]) == [1.0]

    def test_multiple_fences_score_low(self):
        text = "```lua\na\n```\n```lua\nb\n```"
        assert rewards.code_fence_reward_func([_assistant(text)]) == [0.2]

    def test_no_fence_scores_zero(self):
        assert rewards.code_fence_reward_func([_assistant("just prose")]) == [0.0]

    def test_fence_inside_think_is_ignored(self):
        text = "<think>```lua\nx\n```</think>```lua\ny\n```"
        # only the post-think fence should count -> exactly one
        assert rewards.code_fence_reward_func([_assistant(text)]) == [1.0]

    def test_batches_are_scored_independently(self):
        completions = [
            _assistant("```lua\nok\n```"),
            _assistant("no fence"),
        ]
        assert rewards.code_fence_reward_func(completions) == [1.0, 0.0]


# --------------------------------------------------------------------------
# think_format_reward_func
# --------------------------------------------------------------------------

class TestThinkFormatReward:
    def test_long_meaningful_think_scores_one(self):
        think = "a" * 150
        text = f"<think>{think}</think>```lua\nx\n```"
        assert rewards.think_format_reward_func([_assistant(text)]) == [1.0]

    def test_short_think_scores_zero(self):
        text = "<think>too short</think>code"
        assert rewards.think_format_reward_func([_assistant(text)]) == [0.0]

    def test_missing_think_scores_zero(self):
        assert rewards.think_format_reward_func([_assistant("no think tag")]) == [0.0]

    def test_long_but_non_alphanumeric_think_scores_zero(self):
        text = "<think>" + ("." * 200) + "</think>"
        assert rewards.think_format_reward_func([_assistant(text)]) == [0.0]

    def test_exactly_150_alnum_chars_scores_one(self):
        text = "<think>" + ("b" * 150) + "</think>"
        assert rewards.think_format_reward_func([_assistant(text)]) == [1.0]


# --------------------------------------------------------------------------
# no_explanation_reward_func
# --------------------------------------------------------------------------

class TestNoExplanationReward:
    def test_clean_code_only_has_no_penalty(self):
        text = "```lua\nlocal x = 1\n```"
        assert rewards.no_explanation_reward_func([_assistant(text)]) == [0.0]

    def test_short_prose_within_tolerance(self):
        text = "Here:\n```lua\nlocal x = 1\n```"
        assert rewards.no_explanation_reward_func([_assistant(text)]) == [0.0]

    def test_long_explanation_incurs_penalty(self):
        prose = "x" * 225  # 225 outside chars -> (225-25)/200 == 1.0 penalty
        text = f"{prose}\n```lua\nlocal x = 1\n```"
        assert rewards.no_explanation_reward_func([_assistant(text)]) == [-1.0]

    def test_partial_explanation_penalty_is_proportional(self):
        prose = "y" * 125  # (125-25)/200 == 0.5
        text = f"{prose}```lua\nlocal x = 1\n```"
        assert rewards.no_explanation_reward_func([_assistant(text)]) == [-0.5]

    def test_comment_heavy_code_incurs_comment_penalty(self):
        code = "\n".join([f"-- comment {i}" for i in range(5)] + ["local x = 1"])
        text = f"```lua\n{code}\n```"
        # 5 of 6 lines are comments (>0.55) -> -0.5, no prose penalty
        assert rewards.no_explanation_reward_func([_assistant(text)]) == [-0.5]

    def test_no_fence_penalizes_full_prose_length(self):
        # no code fence at all -> the whole text counts as "outside" prose
        text = "x" * 225
        assert rewards.no_explanation_reward_func([_assistant(text)]) == [-1.0]

    def test_penalty_is_clamped_at_negative_one(self):
        prose = "z" * 500
        code = "\n".join([f"-- c{i}" for i in range(5)] + ["local x = 1"])
        text = f"{prose}```lua\n{code}\n```"
        assert rewards.no_explanation_reward_func([_assistant(text)]) == [-1.0]


# --------------------------------------------------------------------------
# _evaluate_single_completion / luau_syntax_reward_func (subprocess mocked)
# --------------------------------------------------------------------------

def _fence(code):
    return _assistant(f"```lua\n{code}\n```")


def _mock_run(returncode=0, stdout="", stderr=""):
    return mock.Mock(returncode=returncode, stdout=stdout, stderr=stderr)


class TestEvaluateSingleCompletion:
    def test_too_short_code_short_circuits_to_zero(self):
        with mock.patch.object(rewards.subprocess, "run") as run:
            assert rewards._evaluate_single_completion(_fence("local x")) == 0.0
            run.assert_not_called()

    def test_code_without_keywords_short_circuits_to_zero(self):
        # long enough but has no local/function/if/... keyword
        with mock.patch.object(rewards.subprocess, "run") as run:
            assert rewards._evaluate_single_completion(_fence("print('hello world here')")) == 0.0
            run.assert_not_called()

    def test_clean_nontrivial_code_scores_one(self):
        code = "local function greet()\n    return 'hi there everyone'\nend"
        with mock.patch.object(rewards.subprocess, "run", return_value=_mock_run(0)):
            assert rewards._evaluate_single_completion(_fence(code)) == 1.0

    def test_clean_trivial_code_scores_low(self):
        # single-line, no function/if -> trivial; passes analysis -> 0.2
        code = "local x = 1 + 2 + 3 + 4"
        with mock.patch.object(rewards.subprocess, "run", return_value=_mock_run(0)):
            assert rewards._evaluate_single_completion(_fence(code)) == 0.2

    def test_failing_analysis_scores_continuous_by_issue_count(self):
        code = "local function greet()\n    return undefined_symbol\nend"
        stderr = "file.luau(2,5): Unknown symbol\nfile.luau(3,1): Another problem"
        with mock.patch.object(rewards.subprocess, "run", return_value=_mock_run(1, stderr=stderr)):
            # 2 real issues -> 0.6 - 0.08*2 == 0.44
            assert rewards._evaluate_single_completion(_fence(code)) == pytest.approx(0.44)

    def test_unused_and_unknown_global_lines_are_not_counted(self):
        code = "local function greet()\n    return value_here_now\nend"
        stderr = (
            "file.luau(1,1): Unused local 'x'\n"
            "file.luau(2,2): Unknown global 'game'\n"
            "file.luau(3,3): Real type error\n"
        )
        with mock.patch.object(rewards.subprocess, "run", return_value=_mock_run(1, stderr=stderr)):
            # only 1 real issue counts -> 0.6 - 0.08 == 0.52
            assert rewards._evaluate_single_completion(_fence(code)) == pytest.approx(0.52)

    def test_failing_trivial_code_is_clamped_below_passing_trivial(self):
        code = "local x = nonexistent_thing_value"
        stderr = "file.luau(1,1): boom"
        with mock.patch.object(rewards.subprocess, "run", return_value=_mock_run(1, stderr=stderr)):
            score = rewards._evaluate_single_completion(_fence(code))
        # trivial+failing must never beat trivial+passing (0.2), and is clamped <=0.15
        assert score <= 0.15

    def test_subprocess_exception_scores_zero(self):
        code = "local function greet()\n    return 'value goes here'\nend"
        with mock.patch.object(rewards.subprocess, "run", side_effect=OSError("boom")):
            assert rewards._evaluate_single_completion(_fence(code)) == 0.0

    def test_tempfile_is_cleaned_up(self):
        code = "local function greet()\n    return 'value goes here'\nend"
        created = {}
        real_ntf = rewards.tempfile.NamedTemporaryFile

        def _spy(*args, **kwargs):
            handle = real_ntf(*args, **kwargs)
            created["path"] = handle.name
            return handle

        with mock.patch.object(rewards.tempfile, "NamedTemporaryFile", side_effect=_spy):
            with mock.patch.object(rewards.subprocess, "run", return_value=_mock_run(0)):
                rewards._evaluate_single_completion(_fence(code))

        import os
        assert not os.path.exists(created["path"])


class TestLuauSyntaxRewardFunc:
    def test_maps_over_all_completions(self):
        good = "local function greet()\n    return 'value goes here'\nend"
        completions = [_fence(good), _assistant("no code at all")]
        with mock.patch.object(rewards.subprocess, "run", return_value=_mock_run(0)):
            result = rewards.luau_syntax_reward_func(completions)
        assert result == [1.0, 0.0]

    def test_handles_empty_batch(self):
        assert rewards.luau_syntax_reward_func([]) == []
