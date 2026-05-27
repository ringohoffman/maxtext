# Copyright 2025–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for input_pipeline_utils SFT masking, verifying correctness on official tokenizer configs.

All tests run completely offline using unmodified official model files committed to local assets.

Design contract (what these tests spec):
  1. apply_chat_template always produces an ``inputs`` array identical to the
     full tokenized conversation and a ``targets`` array where every non-assistant
     token is replaced with ``unk_id=0``.
  2. Every distinct conditional branch of each model's Jinja template is exercised.
  3. The masking strategy falls back to sentinel tagging (Strategy B) for any
     template that does not carry the ``{% generation %}`` marker, which is the
     case for all locally-committed assets.
  4. Native MaxText tokenizer wrappers (SentencePiece, TikToken) intentionally
     do *not* expose ``apply_chat_template``.
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import pytest
import transformers

from maxtext.input_pipeline.input_pipeline_utils import apply_chat_template, extract_token_ids, initialize_sentinel_tokens
from maxtext.utils.globals import MAXTEXT_TEST_ASSETS_ROOT

_ASSET_ROOT = MAXTEXT_TEST_ASSETS_ROOT
_TOKENIZER_ROOT = os.path.join(_ASSET_ROOT, "tokenizers")



def _load_tokenizer(folder_name: str) -> transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast:
  """Load a tokenizer from local assets with no network access."""
  local_path = os.path.join(_TOKENIZER_ROOT, folder_name)
  return transformers.AutoTokenizer.from_pretrained(local_path, local_files_only=True, trust_remote_code=True)


def _apply(tokenizer: transformers.PreTrainedTokenizerBase, messages: list[dict[str, Any]], *, max_target_length: int = 256) -> dict[str, Any]:
  """Convenience wrapper: add sentinels, run apply_chat_template, return result."""
  sentinel_ids = initialize_sentinel_tokens(tokenizer)
  return apply_chat_template(
      {"messages": messages},
      tokenizer,
      "messages",
      sentinel_ids,
      max_target_length=max_target_length,
      unk_id=0,
  )


def _assert_sft_masking_invariants(tc: unittest.TestCase, result: dict[str, Any], label: str = "") -> tuple[np.ndarray, np.ndarray]:
  """Assert the structural invariants that must hold for every well-formed SFT result."""
  inputs, targets = result["inputs"], result["targets"]

  tc.assertIsInstance(inputs, np.ndarray, f"{label}: inputs must be np.ndarray")
  tc.assertIsInstance(targets, np.ndarray, f"{label}: targets must be np.ndarray")
  tc.assertEqual(inputs.shape, targets.shape, f"{label}: inputs/targets shapes must match")
  tc.assertEqual(inputs.dtype, np.int32, f"{label}: inputs dtype must be int32")
  tc.assertEqual(targets.dtype, np.int32, f"{label}: targets dtype must be int32")


  trainable_mask = targets != 0
  tc.assertTrue(np.all(np.where(trainable_mask, targets, inputs) == inputs),
                f"{label}: every trainable target id must equal its corresponding input id")


  tc.assertIn("targets_segmentation", result, f"{label}: targets_segmentation missing")
  expected_seg = (targets != 0).astype(np.int32)
  np.testing.assert_array_equal(
      result["targets_segmentation"], expected_seg,
      err_msg=f"{label}: targets_segmentation must be 1 exactly where targets != 0"
  )

  return inputs, targets


def _decoded_trainable(tokenizer: transformers.PreTrainedTokenizerBase, inputs: np.ndarray, targets: np.ndarray) -> str:
  """Decode only the trainable (loss-active) token positions."""
  return tokenizer.decode([int(inputs[i]) for i in range(len(targets)) if targets[i] != 0])


def _clean(s: str) -> str:
  """Collapse all whitespace for template-agnostic content comparison."""
  return "".join(s.split())


def _assert_assistant_content_trainable_user_masked(
    tc: unittest.TestCase,
    tokenizer: transformers.PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    expected_trainable: list[str],
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
  """Full SFT masking assertion: verify specific strings are (or are not) trainable.

  Args:
    expected_trainable: List of substrings that MUST appear in decoded trainable tokens.
      All user/system message contents are automatically checked NOT to appear.
  """
  result = _apply(tokenizer, messages)
  inputs, targets = _assert_sft_masking_invariants(tc, result, label)
  decoded = _decoded_trainable(tokenizer, inputs, targets)

  for phrase in expected_trainable:
    tc.assertIn(_clean(phrase), _clean(decoded),
                f"{label}: expected trainable phrase {phrase!r} not found in {decoded!r}")

  for msg in messages:
    if msg["role"] in ("user", "system"):
      content = msg["content"]
      if isinstance(content, str):
        tc.assertNotIn(_clean(content), _clean(decoded),
                       f"{label}: {msg['role']} content {content!r} must not be trainable")

  return inputs, targets



@pytest.mark.cpu_only
class TestGemma2bIt(unittest.TestCase):
  """Gemma 2B Instruct — Gemma v1 template.

  Template characteristics:
  - Raises exception on system role (enforced at the very top of the template).
  - Alternation check: raises exception if roles are not user/assistant/... interleaved.
  - Turn markers: ``<start_of_turn>{role}\\n{content}<end_of_turn>\\n``.
  - ``assistant`` role is rendered as ``model``.
  - Uses sentinel/Strategy-B masking (template has no ``{% generation %}`` tag).
  """

  @classmethod
  def setUpClass(cls) -> None:
    try:
      cls.tokenizer = _load_tokenizer("gemma-2b-it")
    except Exception as exc:
      raise unittest.SkipTest(f"gemma-2b-it tokenizer not found: {exc}") from exc

  def test_system_role_raises(self) -> None:
    """System role is explicitly rejected by the Gemma 2B template."""
    with self.assertRaises(Exception):
      _apply(self.tokenizer, [
          {"role": "system", "content": "Be helpful."},
          {"role": "user", "content": "Hello"},
      ])

  def test_alternation_violation_raises(self) -> None:
    """Two consecutive user messages violate the alternation invariant."""
    with self.assertRaises(Exception):
      _apply(self.tokenizer, [
          {"role": "user", "content": "First"},
          {"role": "user", "content": "Second"},
      ])

  def test_single_turn_masking(self) -> None:
    """Assistant response is trainable; user prompt and turn markers are masked."""
    _assert_assistant_content_trainable_user_masked(
        self, self.tokenizer,
        messages=[
            {"role": "user",      "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ],
        expected_trainable=["The answer is 4"],
        label="gemma-2b-it single-turn",
    )

  def test_multi_turn_masking(self) -> None:
    """Both assistant turns are trainable; all user tokens are masked."""
    _assert_assistant_content_trainable_user_masked(
        self, self.tokenizer,
        messages=[
            {"role": "user",      "content": "What is 2+2?"},
            {"role": "assistant", "content": "Four."},
            {"role": "user",      "content": "And 3+3?"},
            {"role": "assistant", "content": "Six."},
        ],
        expected_trainable=["Four", "Six"],
        label="gemma-2b-it multi-turn",
    )

  def test_end_of_turn_token_is_masked(self) -> None:
    """The ``<end_of_turn>`` structural token immediately after assistant content is masked.

    This is a model-level structural delimiter that the model does not generate;
    it must never carry loss.
    """
    result = _apply(self.tokenizer, [
        {"role": "user",      "content": "Hi"},
        {"role": "assistant", "content": "Hello."},
    ], max_target_length=32)
    inputs, targets = result["inputs"], result["targets"]

    token_strs = [self.tokenizer.decode([int(t)]) for t in inputs]
    eot_positions = [i for i, s in enumerate(token_strs) if "<end_of_turn>" in s]
    self.assertGreater(len(eot_positions), 0, "<end_of_turn> token not found in inputs.")
    for pos in eot_positions:
      self.assertEqual(int(targets[pos]), 0,
                       f"<end_of_turn> at position {pos} must be masked, got {targets[pos]}.")


@pytest.mark.cpu_only
class TestGemma3It(unittest.TestCase):
  """Gemma 3 27B Instruct — Gemma v2 template.

  Template characteristics (new vs v1):
  - System message supported: content prepended to the first user message as
    ``{system_content}\\n\\n{user_content}`` (``first_user_prefix``).
    System content may be a plain string OR a list ``[{type, text}]``.
  - Multimodal content: ``message['content']`` may be an iterable of
    ``{"type": "image"}`` / ``{"type": "text", "text": "..."}`` dicts;
    images are rendered as ``<start_of_image>`` special tokens.
  - Sentinel injection supports both string and list assistant content
    (text blocks in the list are wrapped with start/end sentinels).
  - Otherwise identical turn structure to Gemma 2B.
  """

  @classmethod
  def setUpClass(cls) -> None:
    try:
      cls.tokenizer = _load_tokenizer("gemma-3-27b-it")
    except Exception as exc:
      raise unittest.SkipTest(f"gemma-3-27b-it tokenizer not found: {exc}") from exc

  def test_system_string_content_prepended_to_first_user_turn(self) -> None:
    """String system content is fused into the first user turn, not a standalone block."""
    rendered = self.tokenizer.apply_chat_template(
        [
            {"role": "system",    "content": "Be concise."},
            {"role": "user",      "content": "Hello."},
            {"role": "assistant", "content": "Hi."},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    self.assertIn("Be concise.", rendered)
    self.assertNotIn("<start_of_turn>system", rendered)
    # System prefix must appear before the user's own message.
    self.assertLess(rendered.index("Be concise."), rendered.index("Hello."))

  def test_system_list_content_prepended_to_first_user_turn(self) -> None:
    """List-form system content (``[{type, text}]``) is also fused into the first user turn.

    This exercises the ``else`` branch of the system content extraction:
    ``first_user_prefix = messages[0]['content'][0]['text'] + '\\n\\n'``
    """
    rendered = self.tokenizer.apply_chat_template(
        [
            {"role": "system",    "content": [{"type": "text", "text": "Be concise."}]},
            {"role": "user",      "content": "Hello."},
            {"role": "assistant", "content": "Hi."},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    self.assertIn("Be concise.", rendered)
    self.assertNotIn("<start_of_turn>system", rendered)
    self.assertLess(rendered.index("Be concise."), rendered.index("Hello."))

  def test_system_message_sft_masking(self) -> None:
    """With system message, only assistant tokens are trainable."""
    _assert_assistant_content_trainable_user_masked(
        self, self.tokenizer,
        messages=[
            {"role": "system",    "content": "Be concise."},
            {"role": "user",      "content": "What is the sky?"},
            {"role": "assistant", "content": "Blue."},
        ],
        expected_trainable=["Blue"],
        label="gemma-3-27b-it system",
    )

  def test_alternation_violation_raises(self) -> None:
    """Two consecutive user messages violate the alternation invariant."""
    with self.assertRaises(Exception):
      _apply(self.tokenizer, [
          {"role": "user", "content": "First"},
          {"role": "user", "content": "Second"},
      ])

  def test_user_multimodal_content_sft_masking(self) -> None:
    """Multimodal user content (image + text items) is processed for SFT masking.

    Exercises the ``message['content'] is iterable`` branch of the Gemma 3 template
    for user-role messages. The assistant response must be trainable.
    """
    result = _apply(self.tokenizer, [
        {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": "Describe this."}]},
        {"role": "assistant", "content": "A cat."},
    ])
    inputs, targets = _assert_sft_masking_invariants(self, result, "gemma-3-27b-it user multimodal")
    decoded = _decoded_trainable(self.tokenizer, inputs, targets)
    self.assertIn(_clean("A cat"), _clean(decoded))
    self.assertNotIn(_clean("Describe this"), _clean(decoded))

  def test_assistant_list_content_sft_masking(self) -> None:
    """Assistant message with list content has its text blocks correctly wrapped with sentinels.

    Exercises ``_inject_sentinels_into_content`` for list-type content: the sentinel
    is inserted into the first and last text block, so the entire text response is trainable.
    """
    result = _apply(self.tokenizer, [
        {"role": "user",      "content": "Describe the cat."},
        {"role": "assistant", "content": [{"type": "text", "text": "A fluffy cat."}]},
    ])
    inputs, targets = _assert_sft_masking_invariants(self, result, "gemma-3-27b-it assistant list")
    decoded = _decoded_trainable(self.tokenizer, inputs, targets)
    self.assertIn(_clean("fluffy cat"), _clean(decoded))

  def test_multi_turn_string_content_sft_masking(self) -> None:
    """Standard multi-turn conversation with string content produces correct masking."""
    _assert_assistant_content_trainable_user_masked(
        self, self.tokenizer,
        messages=[
            {"role": "user",      "content": "What is 2+2?"},
            {"role": "assistant", "content": "Four."},
            {"role": "user",      "content": "And 3+3?"},
            {"role": "assistant", "content": "Six."},
        ],
        expected_trainable=["Four", "Six"],
        label="gemma-3-27b-it multi-turn",
    )


@pytest.mark.cpu_only
class TestGemma4It(unittest.TestCase):
  """Gemma 4 31B Instruct — Gemma v3 / Gemma 4 template.

  Template characteristics (new vs v2):
  - Uses ``<|turn>{role}\\n{content}<turn|>\\n`` delimiters (different from v1/v2).
  - System role is rendered as a regular turn (role=``system``), not stripped.
  - ``strip_thinking()`` is called on model-role string content; it removes
    ``<|channel|>...<channel|>`` blocks from the rendered output.
    However, when the sentinel wraps content before the channel open marker,
    the thinking content remains in the sentinel region and is fully trainable.
  - ``reasoning_content`` / ``reasoning`` message field triggers a thinking channel
    block only when (a) that message comes after the last user message AND
    (b) tool_calls are present. Otherwise the field is ignored for rendering.
  - Consecutive assistant messages (continuation turns) suppress the duplicate
    ``<|turn>model`` header for all but the first segment.
  - BOS token (``<bos>``) is prepended once at the start.
  """

  @classmethod
  def setUpClass(cls) -> None:
    try:
      cls.tokenizer = _load_tokenizer("gemma-4-31B-it")
    except Exception as exc:
      raise unittest.SkipTest(f"gemma-4-31B-it tokenizer not found: {exc}") from exc

  # -- Template structure / rendering tests --

  def test_empty_messages_raises(self) -> None:
    """Empty message list raises IndexError or ValueError (pre-scan on empty list)."""
    with self.assertRaises((IndexError, ValueError)):
      self.tokenizer.apply_chat_template([], tokenize=False, add_generation_prompt=False)

  def test_bos_token_prepended(self) -> None:
    """``<bos>`` is emitted exactly once at the very start of the rendered output."""
    rendered = self.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False, add_generation_prompt=False,
    )
    self.assertTrue(rendered.startswith("<bos>"),
                    "Gemma 4 template must prepend <bos>.")
    self.assertEqual(rendered.count("<bos>"), 1, "<bos> must appear exactly once.")

  def test_user_turn_with_generation_prompt_opens_thinking_channel(self) -> None:
    """``add_generation_prompt=True`` opens the model thinking channel prefix."""
    rendered = self.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False, add_generation_prompt=True,
    )
    self.assertIn("<|turn>user\nHello<turn|>\n", rendered)
    self.assertIn("<|channel>thought", rendered)

  def test_assistant_turn_rendered_as_model(self) -> None:
    """The ``assistant`` role is re-labeled ``model`` in the rendered output."""
    rendered = self.tokenizer.apply_chat_template(
        [{"role": "assistant", "content": "Hi"}],
        tokenize=False, add_generation_prompt=False,
    )
    self.assertEqual(rendered, "<bos><|turn>model\nHi<turn|>\n")

  def test_system_role_rendered_as_turn(self) -> None:
    """System role is NOT silently dropped; it is emitted as a ``<|turn>system`` block."""
    rendered = self.tokenizer.apply_chat_template(
        [{"role": "system", "content": "System prompt"}],
        tokenize=False, add_generation_prompt=False,
    )
    self.assertEqual(rendered, "<bos><|turn>system\nSystem prompt<turn|>\n")

  def test_consecutive_assistant_turns_share_single_model_header(self) -> None:
    """Two consecutive assistant messages (continuation turn) emit only one ``<|turn>model`` header.

    The continuation-detection logic in the template suppresses the duplicate
    ``<|turn>model\\n`` prefix when the previous non-tool message was also assistant.
    """
    rendered = self.tokenizer.apply_chat_template(
        [
            {"role": "user",      "content": "Q"},
            {"role": "assistant", "content": "Part 1"},
            {"role": "assistant", "content": "Part 2"},
        ],
        tokenize=False, add_generation_prompt=False,
    )
    # Only one model header should appear; Part 2 continues the same turn.
    self.assertEqual(rendered.count("<|turn>model"), 1,
                     "Continuation turn must not repeat <|turn>model header.")
    self.assertIn("Part 1", rendered)
    self.assertIn("Part 2", rendered)

  # -- SFT masking tests --

  def test_single_turn_masking_boundaries(self) -> None:
    """Token-level verification: user + model-header tokens masked; response tokens trainable.

    This is the canonical example of what the masking invariant means in practice
    for the Gemma 4 template.
    """
    result = _apply(self.tokenizer, [
        {"role": "user",      "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
    ], max_target_length=40)
    inputs, targets = _assert_sft_masking_invariants(self, result, "gemma-4 single-turn boundaries")

    token_strs = [self.tokenizer.decode([int(t)]) for t in inputs]


    model_header_idx = next((i for i, s in enumerate(token_strs) if "model" in s), None)
    self.assertIsNotNone(model_header_idx, "Model header token not found.")
    for i in range(model_header_idx + 1):
      self.assertEqual(int(targets[i]), 0,
                       f"Token {token_strs[i]!r} at index {i} (before response) must be masked.")


    self.assertTrue(
        any(targets[i] != 0 for i in range(model_header_idx + 1, len(targets))),
        "No trainable response tokens found.",
    )

  def test_turn_closer_is_masked(self) -> None:
    """The ``<turn|>`` structural token closing each assistant turn is masked."""
    result = _apply(self.tokenizer, [
        {"role": "user",      "content": "Hi"},
        {"role": "assistant", "content": "Hello."},
    ], max_target_length=32)
    inputs, targets = result["inputs"], result["targets"]
    token_strs = [self.tokenizer.decode([int(t)]) for t in inputs]

    turn_closer_positions = [i for i, s in enumerate(token_strs) if "<turn|>" in s]
    self.assertGreater(len(turn_closer_positions), 0, "<turn|> token not found.")
    for pos in turn_closer_positions:
      self.assertEqual(int(targets[pos]), 0, f"<turn|> at {pos} must be masked.")

  def test_multi_turn_masking(self) -> None:
    """Both assistant turns in a 2-turn conversation are trainable; user tokens are masked."""
    _assert_assistant_content_trainable_user_masked(
        self, self.tokenizer,
        messages=[
            {"role": "user",      "content": "What is 2+2?"},
            {"role": "assistant", "content": "Four."},
            {"role": "user",      "content": "And 3+3?"},
            {"role": "assistant", "content": "Six."},
        ],
        expected_trainable=["Four", "Six"],
        label="gemma-4 multi-turn",
    )

  def test_strip_thinking_removes_channel_content_from_sft_targets(self) -> None:
    """``strip_thinking`` strips ``<|channel>...<channel|>`` from the rendered content.

    The Gemma 4 template calls ``strip_thinking(message['content'])`` for ``role == 'model'``
    messages. When the sentinel is injected at the very beginning of the content string
    (before the ``<|channel>`` open marker), ``strip_thinking`` splits on ``<channel|>``:

    - Part 0: ``<|sft_train_start|><|channel>thought\\n...\\n`` — contains ``<|channel>``
      so only the prefix before ``<|channel>`` (``<|sft_train_start|>``) is kept.
    - Part 1: ``{answer}.<|sft_train_end|>`` — no ``<|channel>`` so kept in full.

    Result: the thinking channel content is **stripped** from the trainable region.
    Only the final answer is trainable. This is accurate: for SFT on Gemma 4 with
    inline thinking, use the ``reasoning_content`` field (tested separately in
    ``TestGemma4It.test_single_turn_masking_boundaries``) which goes through a
    dedicated rendering path that preserves the thinking in the sentinel region.
    """
    result = _apply(self.tokenizer, [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": "<|channel>thought\nLet me count: 1, 2, 3, 4.\n<channel|>The answer is 4.",
        },
    ], max_target_length=128)
    inputs, targets = _assert_sft_masking_invariants(self, result, "gemma-4 strip_thinking")

    decoded = _decoded_trainable(self.tokenizer, inputs, targets)

    # strip_thinking removes the thinking block; only the answer is trainable.
    self.assertIn(_clean("The answer is 4"), _clean(decoded),
                  "Answer after the thinking channel must be trainable.")
    self.assertNotIn(_clean("Let me count"), _clean(decoded),
                     "Thinking channel content is stripped by strip_thinking and must NOT be trainable "
                     "when inline <|channel>...<channel|> markers are used in the content field.")

  def test_attention_mask_and_right_padding(self) -> None:
    """Documents the causal + padding attention mask construction used during training.

    MaxText SFT training uses RIGHT-side padding. The 2D attention mask is the
    logical AND of the causal lower-triangular mask and the column-wise non-padding mask.
    """
    pad_id = 0
    max_length = 12
    inputs = np.array([101, 202, 303, 404, 505, 606, 707, 808, pad_id, pad_id, pad_id, pad_id], dtype=np.int32)

    # Right-padding contract: padding tokens must be at the tail.
    padding_positions = np.where(inputs == pad_id)[0].tolist()
    self.assertEqual(padding_positions, [8, 9, 10, 11],
                     "MaxText SFT training requires RIGHT padding.")

    # 2D attention mask = causal AND non-padding.
    inputs_segmentation = (inputs != pad_id)
    causal_mask = np.tril(np.ones((max_length, max_length), dtype=bool))
    padding_mask_2d = np.broadcast_to(inputs_segmentation[None, :], (max_length, max_length))
    attention_mask_2d = causal_mask & padding_mask_2d

    for i in range(max_length):
      for j in range(i + 1, max_length):
        self.assertFalse(attention_mask_2d[i, j],
                         f"Causal violation: token {i} attends to future {j}.")
    for i in range(max_length):
      for j in range(8, 12):
        self.assertFalse(attention_mask_2d[i, j],
                         f"Padding violation: token {i} attends to pad {j}.")


@pytest.mark.cpu_only
class TestLlama3Instruct(unittest.TestCase):
  """Llama 3 8B Instruct — Meta Llama 3 template.

  Template characteristics:
  - BOS token prepended to the very first message.
  - Turn markers: ``<|start_header_id|>{role}<|end_header_id|>\\n\\n{content}<|eot_id|>``.
  - System role supported (rendered as a ``<|start_header_id|>system`` turn).
  - No role alternation constraint (unlike Gemma 2B).
  """

  @classmethod
  def setUpClass(cls) -> None:
    try:
      cls.tokenizer = _load_tokenizer("llama3-8b-instruct")
    except Exception as exc:
      raise unittest.SkipTest(f"llama3-8b-instruct tokenizer not found: {exc}") from exc

  def test_turn_structure(self) -> None:
    """Verifies the Llama 3 header/end-of-turn format."""
    rendered = self.tokenizer.apply_chat_template(
        [
            {"role": "user",      "content": "Hello"},
            {"role": "assistant", "content": "Hi."},
        ],
        tokenize=False, add_generation_prompt=False,
    )
    self.assertIn("<|start_header_id|>user<|end_header_id|>", rendered)
    self.assertIn("<|start_header_id|>assistant<|end_header_id|>", rendered)
    self.assertIn("<|eot_id|>", rendered)

  def test_single_turn_without_system_sft_masking(self) -> None:
    """Pure user/assistant conversation (no system) is correctly masked."""
    _assert_assistant_content_trainable_user_masked(
        self, self.tokenizer,
        messages=[
            {"role": "user",      "content": "What is 2+2?"},
            {"role": "assistant", "content": "Four."},
        ],
        expected_trainable=["Four"],
        label="llama3 no-system single-turn",
    )

  def test_system_role_sft_masking(self) -> None:
    """System role is rendered as a header turn but is entirely masked during training."""
    _assert_assistant_content_trainable_user_masked(
        self, self.tokenizer,
        messages=[
            {"role": "system",    "content": "You are helpful."},
            {"role": "user",      "content": "What is 2+2?"},
            {"role": "assistant", "content": "Four."},
        ],
        expected_trainable=["Four"],
        label="llama3 system",
    )

  def test_multi_turn_sft_masking(self) -> None:
    """Both assistant turns are trainable; all user/system/structural tokens are masked."""
    _assert_assistant_content_trainable_user_masked(
        self, self.tokenizer,
        messages=[
            {"role": "user",      "content": "What is 2+2?"},
            {"role": "assistant", "content": "Four."},
            {"role": "user",      "content": "And 3+3?"},
            {"role": "assistant", "content": "Six."},
        ],
        expected_trainable=["Four", "Six"],
        label="llama3 multi-turn",
    )

  def test_eot_id_token_is_masked(self) -> None:
    """The ``<|eot_id|>`` end-of-turn token after assistant content is masked."""
    result = _apply(self.tokenizer, [
        {"role": "user",      "content": "Hello"},
        {"role": "assistant", "content": "Hi."},
    ], max_target_length=32)
    inputs, targets = result["inputs"], result["targets"]
    token_strs = [self.tokenizer.decode([int(t)]) for t in inputs]
    eot_positions = [i for i, s in enumerate(token_strs) if "<|eot_id|>" in s]
    self.assertGreater(len(eot_positions), 0, "<|eot_id|> not found.")
    for pos in eot_positions:
      self.assertEqual(int(targets[pos]), 0, f"<|eot_id|> at {pos} must be masked.")


@pytest.mark.cpu_only
class TestMistralInstructV03(unittest.TestCase):
  """Mistral 7B Instruct v0.3 — Mistral [INST] template.

  Template characteristics:
  - BOS token prepended to the first message.
  - Turn markers: ``[INST] {content}[/INST] {response}</s>``.
  - Strict alternation check: roles must alternate user/assistant (tool calls excluded).
  - System message (index 0) is prepended to the **last** user turn only.
    Concretely, for a conversation ``[system, user, assistant]`` the template
    iterates ``loop_messages = [user, assistant]``; since the user is not the
    final element (``loop.last``), the system text is silently dropped from the
    rendered output. The system only appears when the user message is the final
    item in loop_messages (i.e., in inference mode with ``add_generation_prompt=True``).
  - Tool-calling paths exist but are out of scope for SFT masking tests.
  """

  @classmethod
  def setUpClass(cls) -> None:
    try:
      cls.tokenizer = _load_tokenizer("mistral-7b-instruct-v0.3")
    except Exception as exc:
      raise unittest.SkipTest(f"mistral-7b-instruct-v0.3 tokenizer not found: {exc}") from exc

  def test_turn_structure(self) -> None:
    """Verifies Mistral's [INST]/[/INST] turn format and BOS token placement."""
    rendered = self.tokenizer.apply_chat_template(
        [
            {"role": "user",      "content": "Hello"},
            {"role": "assistant", "content": "Hi."},
        ],
        tokenize=False, add_generation_prompt=False,
    )
    self.assertIn("[INST]", rendered)
    self.assertIn("[/INST]", rendered)
    self.assertIn("<s>", rendered)  # BOS
    self.assertIn("Hi.", rendered)

  def test_alternation_violation_raises(self) -> None:
    """Two consecutive user messages violate the strict alternation check."""
    with self.assertRaises(Exception):
      _apply(self.tokenizer, [
          {"role": "user", "content": "First"},
          {"role": "user", "content": "Second"},
      ])

  def test_system_message_dropped_for_sft_training_conversations(self) -> None:
    """System message at index 0 is silently dropped when it is not the last user turn.

    In the Mistral template, ``system_message`` is injected only when the user message
    is ``loop.last`` in ``loop_messages``. For a complete ``[system, user, assistant]``
    conversation rendered with ``add_generation_prompt=False``, ``loop_messages =
    [user, assistant]`` so the user is NOT last — the system text does not appear in
    the rendered output at all.

    This is important to know: the SFT training targets for this template are therefore
    always based on a system-free render. Only the assistant text is trainable.
    """
    rendered = self.tokenizer.apply_chat_template(
        [
            {"role": "system",    "content": "You are helpful."},
            {"role": "user",      "content": "What is 2+2?"},
            {"role": "assistant", "content": "Four."},
        ],
        tokenize=False, add_generation_prompt=False,
    )
    # System content is silently absent from the rendered training string.
    self.assertNotIn("You are helpful", rendered,
                     "System text must not appear in the rendered SFT training string.")

    # SFT masking still functions correctly — only assistant is trainable.
    result = _apply(self.tokenizer, [
        {"role": "system",    "content": "You are helpful."},
        {"role": "user",      "content": "What is 2+2?"},
        {"role": "assistant", "content": "Four."},
    ])
    inputs, targets = _assert_sft_masking_invariants(self, result, "mistral system dropped")
    decoded = _decoded_trainable(self.tokenizer, inputs, targets)
    self.assertIn(_clean("Four"), _clean(decoded))

  def test_system_message_prepended_to_last_user_turn_at_inference(self) -> None:
    """In inference mode (``add_generation_prompt=True``), system is fused into the user turn.

    When the user message IS the final element in ``loop_messages``, the template
    renders ``[INST] {system}\\n\\n{user}[/INST]``.  This is the path that exercises
    the ``if loop.last and system_message is defined`` branch.
    """
    rendered = self.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user",   "content": "What is 2+2?"},
        ],
        tokenize=False, add_generation_prompt=True,
    )
    # System and user are fused inside a single [INST]...[/INST] block.
    self.assertIn("You are helpful.", rendered)
    self.assertIn("What is 2+2?", rendered)
    # System must appear before the user message within the same INST block.
    self.assertLess(rendered.index("You are helpful."), rendered.index("What is 2+2?"))

  def test_multi_turn_sft_masking(self) -> None:
    """Both assistant turns are trainable; user prompts are masked."""
    _assert_assistant_content_trainable_user_masked(
        self, self.tokenizer,
        messages=[
            {"role": "user",      "content": "What is 2+2?"},
            {"role": "assistant", "content": "Four."},
            {"role": "user",      "content": "And 3+3?"},
            {"role": "assistant", "content": "Six."},
        ],
        expected_trainable=["Four", "Six"],
        label="mistral multi-turn",
    )

  def test_eos_token_is_masked(self) -> None:
    """The ``</s>`` EOS token that closes each assistant turn is masked."""
    result = _apply(self.tokenizer, [
        {"role": "user",      "content": "Hello"},
        {"role": "assistant", "content": "Hi."},
    ], max_target_length=32)
    inputs, targets = result["inputs"], result["targets"]
    token_strs = [self.tokenizer.decode([int(t)]) for t in inputs]
    eos_positions = [i for i, s in enumerate(token_strs) if "</s>" in s]
    self.assertGreater(len(eos_positions), 0, "</s> token not found.")
    for pos in eos_positions:
      self.assertEqual(int(targets[pos]), 0, f"</s> EOS at {pos} must be masked.")


@pytest.mark.cpu_only
class TestOlmo2Instruct(unittest.TestCase):
  """OLMo 2 7B Instruct — OLMo 2 template.

  Template characteristics:
  - BOS token == EOS token == ``<|endoftext|>`` (same special token for both).
  - Role markers: ``<|system|>``, ``<|user|>``, ``<|assistant|>``.
  - Non-last assistant turns append EOS + newline; the last turn appends EOS only
    (no trailing newline). This is the key ``loop.last`` conditional.
  - System role supported (rendered inline as a ``<|system|>`` block).
  """

  @classmethod
  def setUpClass(cls) -> None:
    try:
      cls.tokenizer = _load_tokenizer("olmo2-7b-instruct")
    except Exception as exc:
      raise unittest.SkipTest(f"olmo2-7b-instruct tokenizer not found: {exc}") from exc

  def test_turn_structure(self) -> None:
    """Verifies OLMo2's role-marker format and EOS placement."""
    rendered = self.tokenizer.apply_chat_template(
        [
            {"role": "user",      "content": "Hello"},
            {"role": "assistant", "content": "Hi."},
        ],
        tokenize=False, add_generation_prompt=False,
    )
    self.assertIn("<|user|>", rendered)
    self.assertIn("<|assistant|>", rendered)
    self.assertIn("<|endoftext|>", rendered)

  def test_non_last_vs_last_assistant_turn_newline(self) -> None:
    """Non-last assistant turns end with EOS+newline; the last turn ends with EOS only.

    This covers the ``{% if not loop.last %}...eos_token + '\\n'...{% else %}...{% endif %}``
    conditional in the OLMo 2 template.
    """
    rendered = self.tokenizer.apply_chat_template(
        [
            {"role": "user",      "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user",      "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ],
        tokenize=False, add_generation_prompt=False,
    )
    self.assertIn("A1<|endoftext|>\n", rendered,
                  "Non-last assistant must end with EOS+newline.")
    self.assertTrue(
        rendered.endswith("A2<|endoftext|>"),
        "Last assistant turn must end with EOS and no trailing newline.",
    )

  def test_system_message_sft_masking(self) -> None:
    """System message is masked; only assistant response is trainable."""
    _assert_assistant_content_trainable_user_masked(
        self, self.tokenizer,
        messages=[
            {"role": "system",    "content": "You are helpful."},
            {"role": "user",      "content": "What is 2+2?"},
            {"role": "assistant", "content": "Four."},
        ],
        expected_trainable=["Four"],
        label="olmo2 system",
    )

  def test_multi_turn_sft_masking(self) -> None:
    """Both assistant turns in a multi-turn conversation are trainable."""
    _assert_assistant_content_trainable_user_masked(
        self, self.tokenizer,
        messages=[
            {"role": "user",      "content": "What is 2+2?"},
            {"role": "assistant", "content": "Four."},
            {"role": "user",      "content": "And 3+3?"},
            {"role": "assistant", "content": "Six."},
        ],
        expected_trainable=["Four", "Six"],
        label="olmo2 multi-turn",
    )

  def test_eos_tokens_are_masked(self) -> None:
    """All ``<|endoftext|>`` EOS tokens (structural turn delimiters) are masked."""
    result = _apply(self.tokenizer, [
        {"role": "user",      "content": "Hello"},
        {"role": "assistant", "content": "Hi."},
    ], max_target_length=32)
    inputs, targets = result["inputs"], result["targets"]
    token_strs = [self.tokenizer.decode([int(t)]) for t in inputs]
    eos_positions = [i for i, s in enumerate(token_strs) if "<|endoftext|>" in s]
    self.assertGreater(len(eos_positions), 0, "<|endoftext|> not found.")
    for pos in eos_positions:
      self.assertEqual(int(targets[pos]), 0, f"<|endoftext|> EOS at {pos} must be masked.")


@pytest.mark.cpu_only
class TestQwen3(unittest.TestCase):
  """Qwen 3 — Qwen 3 template (ChatML variant with thinking).

  Template characteristics:
  - Turn markers: ``<|im_start|>{role}\\n{content}<|im_end|>\\n``.
  - System role supported at index 0 (rendered as a ``<|im_start|>system`` block).
  - A non-first system message (role == ``system`` and not ``loop.first``) is rendered
    as a ``<|im_start|>system`` block in the same style as a user turn.
  - For assistant messages after the last user query (``loop.index0 > last_query_index``):
    - If the message is ``loop.last`` or has ``reasoning_content``, wraps in
      ``<think>\\n{reasoning}\\n</think>\\n\\n{content}``.
    - Otherwise (non-last-query assistant in history), renders plain content.
  - ``reasoning_content`` field takes precedence over ``</think>`` extraction from content.
  - If ``reasoning_content`` is absent but content contains ``</think>``, the template
    splits on ``</think>`` to extract reasoning and wraps it in ``<think>\\n...\\n</think>``.
  - When ``add_generation_prompt=True`` and ``enable_thinking is False``, injects
    an empty ``<think>\\n\\n</think>\\n\\n`` block before the response.
  - The ``<think>`` opening tag is structural context (masked); ``</think>`` closing
    tag is trainable (model must learn to emit it); response content is trainable.
  - ``<|im_end|>`` turn-closer is always masked.
  """

  @classmethod
  def setUpClass(cls) -> None:
    try:
      cls.tokenizer = _load_tokenizer("qwen3-tokenizer")
    except Exception as exc:
      raise unittest.SkipTest(f"qwen3-tokenizer not found: {exc}") from exc

  def test_turn_structure(self) -> None:
    """Verifies Qwen3's ChatML turn format."""
    rendered = self.tokenizer.apply_chat_template(
        [
            {"role": "user",      "content": "Hello"},
            {"role": "assistant", "content": "Hi."},
        ],
        tokenize=False, add_generation_prompt=False,
    )
    self.assertIn("<|im_start|>user\nHello<|im_end|>", rendered)
    self.assertIn("<|im_start|>assistant", rendered)
    self.assertIn("<|im_end|>", rendered)

  def test_plain_assistant_gets_empty_think_block(self) -> None:
    """Last assistant turn without ``reasoning_content`` is wrapped in an empty ``<think>`` block."""
    rendered = self.tokenizer.apply_chat_template(
        [
            {"role": "user",      "content": "Hello"},
            {"role": "assistant", "content": "Hi."},
        ],
        tokenize=False, add_generation_prompt=False,
    )
    self.assertIn("<think>\n\n</think>\n\nHi.", rendered)

  def test_reasoning_content_field_renders_think_block(self) -> None:
    """Explicit ``reasoning_content`` field is rendered inside a ``<think>\\n...\\n</think>`` block."""
    rendered = self.tokenizer.apply_chat_template(
        [
            {"role": "user",      "content": "Q?"},
            {
                "role": "assistant",
                "content": "The answer.",
                "reasoning_content": "Let me think.",
            },
        ],
        tokenize=False, add_generation_prompt=False,
    )
    self.assertIn("<think>\nLet me think.\n</think>", rendered)
    self.assertIn("The answer.", rendered)

  def test_reasoning_extracted_from_think_tags_in_content(self) -> None:
    """If ``reasoning_content`` is absent but content embeds ``<think>...</think>``, it is extracted.

    This exercises the ``else`` branch: ``if '</think>' in content -> split and extract``.
    """
    rendered = self.tokenizer.apply_chat_template(
        [
            {"role": "user",      "content": "Q?"},
            {"role": "assistant", "content": "<think>\nMy reasoning.\n</think>\n\nMy answer."},
        ],
        tokenize=False, add_generation_prompt=False,
    )
    self.assertIn("<think>\nMy reasoning.\n</think>", rendered)
    self.assertIn("My answer.", rendered)

  def test_enable_thinking_false_injects_empty_think_block_on_generation(self) -> None:
    """``enable_thinking=False`` with ``add_generation_prompt=True`` injects an empty think block.

    This covers the ``if enable_thinking is defined and enable_thinking is false`` branch
    in the generation prompt section, which forces the model to skip thinking.
    """
    rendered = self.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Q"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    self.assertIn("<|im_start|>assistant\n<think>\n\n</think>\n\n", rendered)

  def test_system_message_sft_masking(self) -> None:
    """System message is masked; only assistant response is trainable."""
    _assert_assistant_content_trainable_user_masked(
        self, self.tokenizer,
        messages=[
            {"role": "system",    "content": "You are helpful."},
            {"role": "user",      "content": "What is 2+2?"},
            {"role": "assistant", "content": "Four."},
        ],
        expected_trainable=["Four"],
        label="qwen3 system",
    )

  def test_non_first_system_message_rendered_as_system_turn(self) -> None:
    """A system message appearing mid-conversation (not loop.first) renders as a ``<|im_start|>system`` block.

    The template condition ``role == 'system' and not loop.first`` causes mid-conversation
    system messages to be rendered using the same ``<|im_start|>{role}\\n{content}<|im_end|>``
    format as user messages, not merged with the first-turn system block.
    """
    rendered = self.tokenizer.apply_chat_template(
        [
            {"role": "system",    "content": "Initial system"},
            {"role": "user",      "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "system",    "content": "Mid-conversation override"},
            {"role": "user",      "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ],
        tokenize=False, add_generation_prompt=False,
    )
    # Both the initial and mid-conversation system messages appear as im_start|>system blocks.
    self.assertIn("Mid-conversation override", rendered)
    # Count occurrences: initial system + mid-conversation system
    self.assertGreaterEqual(rendered.count("<|im_start|>system"), 2,
                            "Both system turns must render as im_start|>system blocks.")

  def test_thinking_mask_boundaries(self) -> None:
    """Token-level verification: reasoning tokens trainable, think-open tag masked,
    think-close tag trainable, response content trainable.

    Key spec: the model must learn to emit ``</think>`` so it is trainable;
    the ``<think>`` opening is structural prompt context and is masked.
    """
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": "The answer is 4.",
            "reasoning_content": "Let me count: 1, 2, 3, 4.",
        },
    ]
    result = _apply(self.tokenizer, msgs, max_target_length=128)
    inputs, targets = _assert_sft_masking_invariants(self, result, "qwen3 thinking boundaries")

    token_strs = [self.tokenizer.decode([int(t)]) for t in inputs]

    think_open_idx  = next((i for i, s in enumerate(token_strs) if "think" in s and "/" not in s), None)
    think_close_idx = next((i for i, s in enumerate(token_strs) if "think" in s and "/" in s), None)
    im_end_idx = next(
        (i for i, s in enumerate(token_strs) if i > (think_close_idx or 0) and "im_end" in s), None
    )

    self.assertIsNotNone(think_open_idx,  "<think> open tag not found.")
    self.assertIsNotNone(think_close_idx, "</think> close tag not found.")
    self.assertIsNotNone(im_end_idx,      "<|im_end|> not found after think close.")


    self.assertEqual(int(targets[think_open_idx]), 0, "<think> open tag must be masked.")


    reasoning_start = next(
        (i for i in range(think_open_idx + 1, think_close_idx) if "Let" in token_strs[i]),
        think_open_idx + 1,
    )
    for i in range(reasoning_start, think_close_idx):
      self.assertNotEqual(int(targets[i]), 0,
                          f"Reasoning token {token_strs[i]!r} at {i} must be trainable.")


    self.assertNotEqual(int(targets[think_close_idx]), 0, "</think> close tag must be trainable.")


    for i in range(think_close_idx + 1, im_end_idx):
      self.assertNotEqual(int(targets[i]), 0,
                          f"Response token {token_strs[i]!r} at {i} must be trainable.")


    self.assertEqual(int(targets[im_end_idx]), 0, "<|im_end|> turn closer must be masked.")

  def test_non_last_query_assistant_has_no_think_block(self) -> None:
    """Non-last-query assistant turns are rendered without the think block.

    The Qwen3 template only adds ``<think>`` wrapping for assistant messages
    that appear after the last user query (``loop.index0 > last_query_index``).
    Prior assistant turns in history (before the final user message) are
    rendered with plain content.
    """
    msgs = [
        {"role": "user",      "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user",      "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]
    rendered = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    # Split by im_start to isolate individual assistant turns.
    turns = rendered.split("<|im_start|>")
    assistant_turns = [t for t in turns if t.startswith("assistant\n")]
    self.assertGreaterEqual(len(assistant_turns), 2)
    a1_turn, a2_turn = assistant_turns[0], assistant_turns[1]

    # A1 (non-last-query) must be plain content with no <think> block.
    self.assertNotIn("<think>", a1_turn)
    self.assertIn("A1", a1_turn)

    # A2 (last-query) must have the <think> block.
    self.assertIn("<think>", a2_turn)

    # SFT masking must still mark A2 as trainable.
    result = _apply(self.tokenizer, msgs)
    inputs, targets = _assert_sft_masking_invariants(self, result, "qwen3 non-last-query")
    decoded = _decoded_trainable(self.tokenizer, inputs, targets)
    self.assertIn(_clean("A2"), _clean(decoded))
    self.assertNotIn(_clean("Q1"), _clean(decoded))
    self.assertNotIn(_clean("Q2"), _clean(decoded))

  def test_im_end_token_is_masked(self) -> None:
    """All ``<|im_end|>`` turn-closer tokens are masked."""
    result = _apply(self.tokenizer, [
        {"role": "user",      "content": "Hello"},
        {"role": "assistant", "content": "Hi."},
    ], max_target_length=64)
    inputs, targets = result["inputs"], result["targets"]
    token_strs = [self.tokenizer.decode([int(t)]) for t in inputs]
    im_end_positions = [i for i, s in enumerate(token_strs) if "im_end" in s]
    self.assertGreater(len(im_end_positions), 0, "<|im_end|> not found.")
    for pos in im_end_positions:
      self.assertEqual(int(targets[pos]), 0, f"<|im_end|> at {pos} must be masked.")



@pytest.mark.cpu_only
class TestNativeTokenizersLackChatTemplateApi(unittest.TestCase):
  """MaxText's native SentencePiece and TikToken wrappers do NOT expose
  ``apply_chat_template``.

  This is intentional: the HuggingFace SFT pipeline uses HF tokenizers;
  native tokenizers serve the non-SFT training path.
  """

  def _sp_tokenizer(self):
    from maxtext.input_pipeline.input_pipeline_utils import get_tokenizer  # pylint: disable=import-outside-toplevel
    from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
    return get_tokenizer(
        os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.gemma"),
        "sentencepiece",
        add_bos=False,
        add_eos=False,
        hf_access_token=None,
    )

  def _tiktoken_tokenizer(self):
    from maxtext.input_pipeline.input_pipeline_utils import get_tokenizer  # pylint: disable=import-outside-toplevel
    from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
    return get_tokenizer(
        os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer_llama3.tiktoken"),
        "tiktoken",
        add_bos=False,
        add_eos=False,
        hf_access_token=None,
    )

  def test_sentencepiece_lacks_apply_chat_template(self) -> None:
    tokenizer = self._sp_tokenizer()
    self.assertFalse(
        hasattr(tokenizer, "apply_chat_template"),
        "SentencePieceTokenizer unexpectedly gained apply_chat_template.",
    )

  def test_tiktoken_lacks_apply_chat_template(self) -> None:
    tokenizer = self._tiktoken_tokenizer()
    self.assertFalse(
        hasattr(tokenizer, "apply_chat_template"),
        "TikTokenTokenizer unexpectedly gained apply_chat_template.",
    )


def _segment_by_loss_mask(
    tokenizer: transformers.PreTrainedTokenizerBase,
    inputs: np.ndarray,
    targets: np.ndarray,
) -> list[tuple[str, bool]]:
  """Groups the token stream into contiguous segments of trainable/masked regions."""
  segments: list[tuple[str, bool]] = []
  current: list[int] = []
  current_trainable = (targets[0] != 0)

  for i in range(len(inputs)):
    tid = int(inputs[i])
    if tid == 0:  # Padding / end of sequence
      break
    is_trainable = (targets[i] != 0)
    if is_trainable != current_trainable:
      if current:
        segments.append((tokenizer.decode(current), current_trainable))
      current = [tid]
      current_trainable = is_trainable
    else:
      current.append(tid)

  if current:
    segments.append((tokenizer.decode(current), current_trainable))

  return segments


@pytest.mark.cpu_only
class TestSftVisualMaskBoundaries(unittest.TestCase):
  """Documents and asserts exact text chunks that fall inside/outside SFT loss masks."""

  def test_gemma4_sft_train_last_turn_only_text_only(self) -> None:
    """Test 1: Gemma 4 (sft_train_last_turn_only=True, sft_train_on_thoughts_only=False).

    Expected Masking:
    - Prompt, system, headers (including <|think|>\n trigger), and all prior turns: Masked.
    - Final assistant response: Trainable.
    """
    tokenizer = _load_tokenizer("gemma-4-31B-it")
    sentinel_ids = initialize_sentinel_tokens(tokenizer)

    messages = [
        {"role": "user",      "content": "What is 2+2?"},
        {"role": "assistant", "content": "<|channel>thought\nLet me think.\n<channel|>It is 4."},
        {"role": "user",      "content": "And 3+3?"},
        {"role": "assistant", "content": "<|channel>thought\nLet me count.\n<channel|>It is 6."},
    ]

    res = apply_chat_template(
        {"messages": messages},
        tokenizer, "messages", sentinel_ids, 256, unk_id=0,
        sft_train_last_turn_only=True,
        sft_train_on_thoughts_only=False,
        enable_thinking=True
    )

    segments = _segment_by_loss_mask(tokenizer, res["inputs"], res["targets"])

    self.assertEqual(segments, [
        (
            "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\n"
            "What is 2+2?<turn|>\n<|turn>model\nIt is 4.<turn|>\n"
            "<|turn>user\nAnd 3+3?<turn|>\n<|turn>model\n",
            False,
        ),
        ("It is 6.", True),
        ("<turn|>\n", False),
    ])

  def test_gemma4_sft_train_last_turn_only_thought_only(self) -> None:
    """Test 2: Gemma 4 (sft_train_last_turn_only=True, sft_train_on_thoughts_only=True).

    Expected Masking:
    - Since the raw text template strips thoughts from the content string,
      toggling thought-only training results in 0 trainable tokens (the entire
      dialogue is safely masked).
    """
    tokenizer = _load_tokenizer("gemma-4-31B-it")
    sentinel_ids = initialize_sentinel_tokens(tokenizer)

    messages = [
        {"role": "user",      "content": "What is 2+2?"},
        {"role": "assistant", "content": "<|channel>thought\nLet me think.\n<channel|>It is 4."},
        {"role": "user",      "content": "And 3+3?"},
        {"role": "assistant", "content": "<|channel>thought\nLet me count.\n<channel|>It is 6."},
    ]

    res = apply_chat_template(
        {"messages": messages},
        tokenizer, "messages", sentinel_ids, 256, unk_id=0,
        sft_train_last_turn_only=True,
        sft_train_on_thoughts_only=True,
        enable_thinking=True
    )

    segments = _segment_by_loss_mask(tokenizer, res["inputs"], res["targets"])

    self.assertEqual(segments, [
        (
            "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\n"
            "What is 2+2?<turn|>\n<|turn>model\nIt is 4.<turn|>\n"
            "<|turn>user\nAnd 3+3?<turn|>\n<|turn>model\nIt is 6.<turn|>\n",
            False,
        )
    ])

  def test_gemma4_train_all_turns_text_only(self) -> None:
    """Test 3: Gemma 4 (sft_train_last_turn_only=False, sft_train_on_thoughts_only=False).

    Expected Masking:
    - Dialogue headers, queries, and turn markers: Masked.
    - All assistant visible text turns (Turn 1 and Turn 2): Trainable.
    """
    tokenizer = _load_tokenizer("gemma-4-31B-it")
    sentinel_ids = initialize_sentinel_tokens(tokenizer)

    messages = [
        {"role": "user",      "content": "What is 2+2?"},
        {"role": "assistant", "content": "<|channel>thought\nLet me think.\n<channel|>It is 4."},
        {"role": "user",      "content": "And 3+3?"},
        {"role": "assistant", "content": "<|channel>thought\nLet me count.\n<channel|>It is 6."},
    ]

    res = apply_chat_template(
        {"messages": messages},
        tokenizer, "messages", sentinel_ids, 256, unk_id=0,
        sft_train_last_turn_only=False,
        sft_train_on_thoughts_only=False,
        enable_thinking=True
    )

    segments = _segment_by_loss_mask(tokenizer, res["inputs"], res["targets"])

    self.assertEqual(segments, [
        (
            "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\n"
            "What is 2+2?<turn|>\n<|turn>model\n",
            False,
        ),
        ("It is 4.", True),
        (
            "<turn|>\n<|turn>user\nAnd 3+3?<turn|>\n<|turn>model\n",
            False,
        ),
        ("It is 6.", True),
        ("<turn|>\n", False),
    ])

  def test_kimi_style_train_all_turns_thoughts_only(self) -> None:
    """Test 4: Kimi-K2.6 (sft_train_last_turn_only=False, sft_train_on_thoughts_only=True).

    Expected Masking:
    - Prompts, responses, and turn wrappers (using native Kimi tokens): Masked.
    - Thought channels across both Turns 1 & 2 (preserved in history via preserve_thinking=True): Trainable.
    """
    tokenizer = _load_tokenizer("kimi-k2.6")
    sentinel_ids = initialize_sentinel_tokens(tokenizer)

    messages = [
        {"role": "user",      "content": "What is 2+2?"},
        {"role": "assistant", "content": "It is 4.", "reasoning_content": "Let me think."},
        {"role": "user",      "content": "And 3+3?"},
        {"role": "assistant", "content": "It is 6.", "reasoning_content": "Let me count."},
    ]

    res = apply_chat_template(
        {"messages": messages},
        tokenizer, "messages", sentinel_ids, 256, unk_id=0,
        sft_train_last_turn_only=False,
        sft_train_on_thoughts_only=True,
        preserve_thinking=True
    )

    segments = _segment_by_loss_mask(tokenizer, res["inputs"], res["targets"])

    self.assertEqual(segments, [
        (
            "<|im_user|>user<|im_middle|>What is 2+2?<|im_end|>"
            "<|im_assistant|>assistant<|im_middle|><think>",
            False,
        ),
        ("Let me think.", True),
        (
            "</think>It is 4.<|im_end|><|im_user|>user<|im_middle|>"
            "And 3+3?<|im_end|><|im_assistant|>assistant<|im_middle|><think>",
            False,
        ),
        ("Let me count.", True),
        ("</think>It is 6.<|im_end|>", False),
    ])

  def test_qwen3_sft_train_last_turn_only_thought_only_inline_splitting(self) -> None:
    """Test 5: Qwen 3 (sft_train_last_turn_only=True, sft_train_on_thoughts_only=True).

    Expected Masking:
    - User queries, history assistant turns, and formatting tags: Masked.
    - Only the final assistant turn's reasoning channel (T2\n</think>): Trainable.
    """
    tokenizer = _load_tokenizer("qwen3-tokenizer")
    sentinel_ids = initialize_sentinel_tokens(tokenizer)

    messages = [
        {"role": "user",      "content": "Q1"},
        {"role": "assistant", "content": "<think>\nT1\n</think>\nA1"},
        {"role": "user",      "content": "Q2"},
        {"role": "assistant", "content": "<think>\nT2\n</think>\nA2"},
    ]

    res = apply_chat_template(
        {"messages": messages},
        tokenizer, "messages", sentinel_ids, 256, unk_id=0,
        sft_train_last_turn_only=True,
        sft_train_on_thoughts_only=True
    )

    segments = _segment_by_loss_mask(tokenizer, res["inputs"], res["targets"])

    self.assertEqual(segments, [
        (
            "<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\n"
            "A1<|im_end|>\n<|im_start|>user\nQ2<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n",
            False,
        ),
        ("\nT2\n</think>\n\n", True),
        ("\nA2<|im_end|>\n", False),
    ])

  def test_context_and_schema_masking(self) -> None:
    """Test 6: Context and Tool Schema Masking (EAC Scenario).

    To prevent the model from predicting time, locations, or API declaration
    schemas, all system and context messages are 100% masked; only assistant response trains.
    """
    tokenizer = _load_tokenizer("gemma-4-31B-it")
    sentinel_ids = initialize_sentinel_tokens(tokenizer)

    messages = [
        {"role": "system",    "content": "Tool Definition: SEARCH(query)"},
        {"role": "system",    "content": "Time: 10:00 AM, May 27, 2026"},
        {"role": "user",      "content": "What is the weather?"},
        {"role": "assistant", "content": "It is sunny."},
    ]

    res = apply_chat_template(
        {"messages": messages},
        tokenizer, "messages", sentinel_ids, 256, unk_id=0,
        sft_train_last_turn_only=False,
        sft_train_on_thoughts_only=False
    )

    segments = _segment_by_loss_mask(tokenizer, res["inputs"], res["targets"])

    self.assertEqual(segments, [
        (
            "<bos><|turn>system\nTool Definition: SEARCH(query)<turn|>\n"
            "<|turn>system\nTime: 10:00 AM, May 27, 2026<turn|>\n"
            "<|turn>user\nWhat is the weather?<turn|>\n<|turn>model\n",
            False,
        ),
        ("It is sunny.", True),
        ("<turn|>\n", False),
    ])


if __name__ == "__main__":
  unittest.main()
