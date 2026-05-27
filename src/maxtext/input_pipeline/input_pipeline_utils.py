# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input pipeline utilities for Grain, HuggingFace, and SFT masking transforms."""

from __future__ import annotations

import dataclasses
import warnings
from threading import current_thread
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
  import datasets
  import tensorflow as tf
  import transformers

import grain.python as grain
import numpy as np
from grain._src.python.dataset.sources.tfrecord_dataset import (_TFRecordDatasetIterator,  # pylint: disable=protected-access
                                                                _TFRecordReader)
from grain.experimental import TFRecordIterDataset

from maxtext.input_pipeline import tokenizer
from maxtext.input_pipeline.protos import example_pb2
from maxtext.multimodal import processor as mm_processor
from maxtext.multimodal import utils as mm_utils
from maxtext.utils import gcs_utils, max_logging

Features = dict[str, Any]
INPUT_TOKENS_KEY = "input_ids"


########## Functions used by TFDS pipeline


def normalize_features(x, column_name):
  return {"inputs": x[column_name], "targets": x[column_name]}


def get_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token=None):
  # Load tokenizer
  tokenizer_model = tokenizer.build_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token)
  return tokenizer_model


def truncate_to_max_allowable_length(x, max_length):
  return {k: v[:max_length] for k, v in x.items()}


def shift_data_by_truncation(x):
  x["inputs"] = x["inputs"][:-1]
  x["targets"] = x["targets"][1:]
  return x


def add_segmentation_and_position(x, data_columns, padding_token=0):
  import tensorflow as tf  # pylint: disable=import-outside-toplevel

  for data_column in data_columns:
    x[f"{data_column}_segmentation"] = tf.cast(x[data_column] != padding_token, tf.int32)
    x[f"{data_column}_position"] = tf.broadcast_to(
        tf.range(x[data_column].shape[-1], dtype=np.int32)[None, :], x[data_column].shape
    )
  return x


def TokenizeOp(tokenizer_model, features: Features, data_keys: Iterable[str] = ("inputs", "targets")) -> Features:
  """Op for tokenization"""
  import tensorflow as tf  # pylint: disable=import-outside-toplevel

  def _process_string(string_tensor):
    # Extract string value and decode it if necessary
    string_value = string_tensor.numpy().decode("utf-8")
    # encode and extract the tokenized integers
    modified_string = tokenizer_model.encode(string_value)
    return [modified_string]

  for k in data_keys:
    features[k] = tf.py_function(_process_string, [features[k]], Tout=[tf.int32])[0]
  return features


########## Functions used by HF pipeline


def reformat_prompt(example, column, image_placeholder, model_name):
  """reformat prompt for multimodal SFT"""
  if isinstance(example["images"], list):
    num_images = len(example["images"])
  else:
    num_images = 1
  example[column] = mm_processor.reformat_prompt(example[column], image_placeholder, model_name, num_images)
  return example


def reformat_response(example, column, model_name):
  """reformat response for multimodal SFT"""
  example[column] = mm_processor.reformat_response(example[column][0], model_name)
  return example


def merge_image_columns(example, image_columns, max_num_images_per_example):
  """Merge multiple image columns into a single list of images."""
  images = []
  for col in image_columns:
    if isinstance(example[col], list):
      images.extend(example[col])
    else:
      images.append(example[col])

  example["images"] = images[:max_num_images_per_example] if max_num_images_per_example > 0 else images
  return example


def pre_process_image_sft(example, image_column, model_name):
  """pre-process image for multimodal SFT"""

  def _process_image_fn(image):
    if isinstance(image, list):
      image = [np.array(mm_utils.convert_to_RGB(img)) for img in image]
    else:
      image = np.array(mm_utils.convert_to_RGB(image))

    image = mm_processor.preprocess_image_for_training(image, model_name)
    return image

  example[image_column] = _process_image_fn(example[image_column])
  return example


def prepare_text_for_image_fusion(example, column_name, config):
  """prepare text for image fusion for multimodal SFT"""
  example[column_name] = mm_processor.prepare_text_for_image_fusion(
      tokens=example[column_name], config=config, processor_output=example["images"]
  )
  return example


def combine_columns(example, columns, data_column):
  """Combine columns such as 'prompt' and 'completion' for sft training"""
  assert len(columns) > 1
  combined = []
  for i in range(len(example[columns[0]])):
    for c in columns:
      combined.append(example[c][i])
  example[data_column] = combined
  return example


def is_conversational(features, data_columns):
  """Check if data is in a conversational format.
  Examples:

  features = {'prompt': [{'content': Value(dtype='string', id=None), 'role': Value(dtype='string', id=None)}],
              'completion': [{'content': Value(dtype='string', id=None), 'role': Value(dtype='string', id=None)}]}
  data_columns = ["prompt", "completion"]
  is_conversational(features, data_columns) return True.

  features = {'prompt': [Value(dtype='string', id=None)], 'completion': [Value(dtype='string', id=None)]}
  data_columns = ["prompt", "completion"]
  is_conversational(features, data_columns) returns False.
  """
  import datasets  # pylint: disable=import-outside-toplevel

  for column in data_columns:
    messages = features[column]
    if isinstance(messages, datasets.Sequence):
      if isinstance(messages.feature, dict) and "role" in messages.feature and "content" in messages.feature:
        return True

  return False


def extract_token_ids(tokens):
  """Extracts token IDs from various tokenizer output formats.

  This helper function standardizes the extraction of tokenized integer IDs
  from common return types of Hugging Face tokenizers, including
  `BatchEncoding` objects, dictionaries, or simple lists.

  Args:
    tokens: The object containing token IDs. Supported types include:
      - A list of integers.
      - A dictionary containing the `INPUT_TOKENS_KEY`.
      - An object (e.g., `BatchEncoding`) with an attribute named `INPUT_TOKENS_KEY`.

  Returns:
    A list of integer token IDs.

  Raises:
    ValueError: If the input type is not supported or does not contain the expected key.
  """
  # attention masks in BatchEncoding are effectively ignored
  if hasattr(tokens, INPUT_TOKENS_KEY):
    return getattr(tokens, INPUT_TOKENS_KEY)
  elif isinstance(tokens, dict) and INPUT_TOKENS_KEY in tokens:
    return tokens[INPUT_TOKENS_KEY]
  elif isinstance(tokens, list):
    return tokens
  else:
    raise ValueError(f"Can't extract token_ids from type {type(tokens)}")


def tokenization(example, hf_tokenizer, truncation, max_length, column_names):
  """Tokenize a HuggingFace dataset"""
  for column_name in column_names:
    if isinstance(example[column_name], list):
      example[column_name] = [
          hf_tokenizer(x, truncation=truncation, max_length=max_length)["input_ids"] for x in example[column_name]
      ]
    elif isinstance(example[column_name], str):
      example[column_name] = hf_tokenizer(example[column_name], truncation=truncation, max_length=max_length)["input_ids"]
  return example


@dataclasses.dataclass
class SFTPromptMaskingVision(grain.MapTransform):
  """SFT prompt masking for multimodal"""

  def __init__(self, query_column, response_column, max_target_length, pad_id):
    self.query_column = query_column
    self.response_column = response_column
    self.max_target_length = max_target_length
    self.pad_id = pad_id

  def map(self, element):
    inputs = np.concatenate((element[self.query_column], element[self.response_column]))
    targets = np.concatenate((np.asarray([self.pad_id] * len(element[self.query_column])), element[self.response_column]))
    return {
        "inputs": np.asarray(inputs[: self.max_target_length], dtype=np.int32),
        "targets": np.asarray(targets[: self.max_target_length], dtype=np.int32),
        "images": element["images"],
    }


def _supports_assistant_tokens_mask(tokenizer_model: transformers.PreTrainedTokenizerBase) -> bool:
  """Returns True iff the tokenizer's apply_chat_template supports return_assistant_tokens_mask."""
  import inspect  # pylint: disable=import-outside-toplevel
  if not hasattr(tokenizer_model, "apply_chat_template"):
    return False
  try:
    sig = inspect.signature(tokenizer_model.apply_chat_template)
    return "return_assistant_tokens_mask" in sig.parameters
  except (ValueError, TypeError):
    return False


_START_SENTINEL_STR = "<|sft_train_start|>"
_END_SENTINEL_STR   = "<|sft_train_end|>"
_SENTINEL_TOKENS = [_START_SENTINEL_STR, _END_SENTINEL_STR]


def _inject_sentinels_into_content(
    content: str | list[dict[str, Any]]
) -> str | list[dict[str, Any]]:
  """Wraps assistant message content with SFT sentinel markers.

  For string content: prepends START and appends END to the whole string.
  For list content (multimodal): inserts START at the start of the first
  text block and END at the end of the last text block, leaving non-text
  blocks (images etc.) untouched.

  Raises ValueError if list content contains no text blocks (nothing to mask).
  """
  if isinstance(content, str):
    return f"{_START_SENTINEL_STR}{content}{_END_SENTINEL_STR}"
  if not isinstance(content, list):
    raise TypeError(f"Unexpected content type {type(content)!r}; expected str or list.")
  text_block_indices = [i for i, b in enumerate(content) if b.get("type") == "text"]
  if not text_block_indices:
    raise ValueError("Assistant message has list content with no text blocks; cannot inject sentinels.")
  result = [dict(b) for b in content]
  first, last = text_block_indices[0], text_block_indices[-1]
  result[first]["text"] = _START_SENTINEL_STR + result[first]["text"]
  result[last]["text"] = result[last]["text"] + _END_SENTINEL_STR
  return result


def initialize_sentinel_tokens(tokenizer_model: transformers.PreTrainedTokenizerBase) -> tuple[int, int]:
  """Adds sentinel special tokens to the tokenizer and returns their IDs.

  Must be called once at pipeline setup time. The returned IDs are used by
  `apply_chat_template` to locate trainable regions in the token stream.
  The tokenizer's vocabulary is mutated; these IDs must never reach the model.
  """
  tokenizer_model.add_special_tokens({"additional_special_tokens": _SENTINEL_TOKENS})
  start_id = tokenizer_model.convert_tokens_to_ids(_START_SENTINEL_STR)
  end_id = tokenizer_model.convert_tokens_to_ids(_END_SENTINEL_STR)
  if start_id == tokenizer_model.unk_token_id:
    raise ValueError(
        f"Sentinel {_START_SENTINEL_STR!r} resolved to unk_token_id={tokenizer_model.unk_token_id}; "
        "token was not added correctly."
    )
  return start_id, end_id


# pylint: disable=too-many-positional-arguments
def apply_chat_template(
    example: dict[str, Any],
    tokenizer_model: transformers.PreTrainedTokenizerBase,
    data_column_name: str,
    sentinel_ids: tuple[int, int],
    max_target_length: int,
    unk_id: int = 0,
    chat_template: str | None = None,
    sft_train_last_turn_only: bool = False,
    sft_train_on_thoughts_only: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
  """Formats a conversation and creates SFT loss targets via sentinel-based masking.

  Injects sentinel markers around assistant content in text space, renders the
  conversation through the chat template, tokenizes as a single string, then
  strips sentinel tokens to produce aligned inputs/targets arrays.

  Args:
      example: Dataset element with a list of {role, content} messages.
      tokenizer_model: HuggingFace tokenizer. Must have sentinel IDs already added
          via `initialize_sentinel_tokens`.
      data_column_name: Key in example holding the message list.
      sentinel_ids: (start_id, end_id) returned by `initialize_sentinel_tokens`.
      max_target_length: Max length for inputs and targets arrays.
      unk_id: Token ID used to represent masked (non-trainable) positions in targets.
      chat_template: Optional Jinja chat template override string.
      sft_train_last_turn_only: If True, only trains on the final assistant turn in the sequence.
      sft_train_on_thoughts_only: If True, only trains on the reasoning/thinking blocks.

  Returns:
      Dict with 'inputs' (np.int32), 'targets' (np.int32, prompt positions filled
      with unk_id), segmentation, and position arrays.
  """
  if chat_template == "":
    chat_template = None
  raw_messages: list[dict[str, Any]] = example[data_column_name]
  start_sentinel_id, end_sentinel_id = sentinel_ids

  # Native HF assistant token masking does not support advanced turn or reasoning masking.
  # If any advanced masking option is active, we bypass native masking.
  if not sft_train_last_turn_only and not sft_train_on_thoughts_only and _supports_assistant_tokens_mask(tokenizer_model):
    try:
      result = tokenizer_model.apply_chat_template(
          raw_messages,
          tokenize=True,
          return_assistant_tokens_mask=True,
          return_dict=True,
          chat_template=chat_template,
      )
      input_ids = np.asarray(result["input_ids"], dtype=np.int32)
      asst_mask = np.asarray(result["assistant_masks"], dtype=bool)
      if np.any(asst_mask):
        targets      = np.where(asst_mask, input_ids, np.int32(unk_id))
        inputs_slice = input_ids[:max_target_length]
        targets_slice = targets[:max_target_length]
        asst_mask_slice = asst_mask[:max_target_length]
        ret = {
            "inputs": inputs_slice,
            "targets": targets_slice,
            "inputs_segmentation": np.ones(len(inputs_slice), dtype=np.int32),
            "targets_segmentation": np.where(asst_mask_slice, np.int32(1), np.int32(0)),
            "inputs_position": np.arange(len(inputs_slice), dtype=np.int32),
            "targets_position": np.arange(len(targets_slice), dtype=np.int32),
        }
        if "images" in example:
          ret["images"] = example["images"]
        return ret
    except Exception as exc:  # pylint: disable=broad-except
      max_logging.log(
          f"Native assistant_tokens_mask failed ({type(exc).__name__}: {exc}); "
          "falling back to sentinel-based masking."
      )

  tagged_messages = []
  last_asst_idx = -1
  for idx, msg in enumerate(raw_messages):
    if msg["role"] == "assistant":
      last_asst_idx = idx

  for idx, msg in enumerate(raw_messages):
    tagged = dict(msg)
    if msg["role"] == "assistant":
      if sft_train_last_turn_only and idx != last_asst_idx:
        # Keep prior assistant turns completely sentinel-free (so they are masked out)
        pass
      else:
        if sft_train_on_thoughts_only:
          if "reasoning_content" in msg and msg["reasoning_content"]:
            # Wrap ONLY the reasoning content chunk inside boundaries
            tagged["reasoning_content"] = f"{_START_SENTINEL_STR}{msg['reasoning_content']}{_END_SENTINEL_STR}"
          elif "</think>" in msg["content"]:
            # Support inline split-thought masking (e.g. Qwen3/Deepseek style)
            content = msg["content"]
            parts = content.split("</think>", 1)
            if "<think>" in parts[0]:
              think_parts = parts[0].split("<think>", 1)
              tagged_content = f"<think>{_START_SENTINEL_STR}{think_parts[1]}</think>{_END_SENTINEL_STR}{parts[1]}"
            else:
              tagged_content = f"{_START_SENTINEL_STR}{parts[0]}</think>{_END_SENTINEL_STR}{parts[1]}"
            tagged["content"] = tagged_content
          else:
            pass
        else:
          if "reasoning_content" in msg and msg["reasoning_content"]:
            tagged["reasoning_content"] = f"{_START_SENTINEL_STR}{msg['reasoning_content']}"
            tagged["content"] = f"{msg['content']}{_END_SENTINEL_STR}"
          else:
            tagged["content"] = _inject_sentinels_into_content(msg["content"])
    tagged_messages.append(tagged)

  if hasattr(tokenizer_model, "apply_chat_template"):
    rendered_str = tokenizer_model.apply_chat_template(
        tagged_messages,
        add_generation_prompt=False,
        tokenize=False,
        chat_template=chat_template,
        **kwargs,
    )
  else:
    if chat_template is None:
      wrapper_type = type(tokenizer_model).__name__
      raise TypeError(
          "apply_chat_template expects a Hugging Face tokenizer supporting the "
          "'apply_chat_template' method, but received a tokenizer model of "
          f"type '{wrapper_type}' without a custom 'chat_template'.\n\nTo "
          "format conversational/chat datasets on a native binary tokenizer, you must "
          "explicitly provide a custom chat template file (e.g. setting "
          "'chat_template_path')."
      )
    
    from jinja2 import Environment  # pylint: disable=import-outside-toplevel
    env = Environment()
    
    bos_token = ""
    eos_token = ""
    if hasattr(tokenizer_model, "bos_id") and tokenizer_model.bos_id is not None:
      try:
        bos_token = tokenizer_model.decode([tokenizer_model.bos_id])
      except Exception:
        pass
    if hasattr(tokenizer_model, "eos_id") and tokenizer_model.eos_id is not None:
      try:
        eos_token = tokenizer_model.decode([tokenizer_model.eos_id])
      except Exception:
        pass
        
    try:
      template_obj = env.from_string(chat_template)
      rendered_str = template_obj.render(
          messages=tagged_messages,
          bos_token=bos_token,
          eos_token=eos_token,
          add_generation_prompt=False,
          **kwargs
      )
    except Exception as e:
      raise ValueError(f"Failed to functionally render custom Jinja chat template: {e}") from e

  if callable(tokenizer_model):
    tokenized = tokenizer_model(rendered_str, add_special_tokens=False)
  else:
    tokenized = tokenizer_model.encode(rendered_str)

  token_ids: list[int] = extract_token_ids(tokenized)

  clean_inputs: list[int] = []
  clean_targets: list[int] = []
  in_assistant_turn = False
  for token_id in token_ids:
    if token_id == start_sentinel_id:
      if in_assistant_turn:
        raise ValueError("Nested START sentinel detected; check message content for accidental sentinel strings.")
      in_assistant_turn = True
      continue
    if token_id == end_sentinel_id:
      if not in_assistant_turn:
        raise ValueError("END sentinel without matching START; token stream is corrupted.")
      in_assistant_turn = False
      continue
    clean_inputs.append(token_id)
    clean_targets.append(token_id if in_assistant_turn else unk_id)

  if in_assistant_turn:
    raise ValueError("START sentinel without matching END; assistant turn is not closed.")

  inputs_arr = np.asarray(clean_inputs[:max_target_length],  dtype=np.int32)
  targets_arr = np.asarray(clean_targets[:max_target_length], dtype=np.int32)
  ret = {
      "inputs": inputs_arr,
      "targets": targets_arr,
      "inputs_segmentation": np.ones(len(inputs_arr), dtype=np.int32),
      "targets_segmentation": np.where(targets_arr != unk_id, np.int32(1), np.int32(0)),
      "inputs_position": np.arange(len(inputs_arr), dtype=np.int32),
      "targets_position": np.arange(len(targets_arr), dtype=np.int32),
  }
  if "images" in example:
    ret["images"] = example["images"]
  return ret


@dataclasses.dataclass
class SFTPromptMasking(grain.MapTransform):
  """Single-pass SFT prompt masking as a Grain MapTransform for HuggingFace pipelines.

  Delegates to `apply_chat_template` for each element. The tokenizer must
  have sentinel IDs already registered via `initialize_sentinel_tokens`.

  Args:
      tokenizer_model: HuggingFace tokenizer with sentinel tokens already added.
      data_column_name: Key in each element holding the list of message dicts.
      sentinel_ids: `(start_id, end_id)` tuple returned by `initialize_sentinel_tokens`.
      max_target_length: Maximum sequence length for inputs and targets arrays.
      pad_id: Token ID used for masked (non-trainable) positions in targets.
      chat_template: Optional Jinja chat template override string.
      sft_train_last_turn_only: If True, only the final assistant turn carries loss.
      sft_train_on_thoughts_only: If True, only reasoning/thinking blocks carry loss.
      **kwargs: Extra keyword arguments forwarded to `apply_chat_template`.
  """

  # pylint: disable=too-many-positional-arguments
  def __init__(
      self,
      tokenizer_model: transformers.PreTrainedTokenizerBase,
      data_column_name: str,
      sentinel_ids: tuple[int, int],
      max_target_length: int,
      pad_id: int,
      chat_template: str | None = None,
      sft_train_last_turn_only: bool = False,
      sft_train_on_thoughts_only: bool = False,
      **kwargs: Any,
  ) -> None:
    self.tokenizer_model = tokenizer_model
    self.data_column_name = data_column_name
    self.sentinel_ids = sentinel_ids
    self.max_target_length = max_target_length
    self.pad_id = pad_id
    self.chat_template = chat_template
    self.sft_train_last_turn_only = sft_train_last_turn_only
    self.sft_train_on_thoughts_only = sft_train_on_thoughts_only
    self.kwargs = kwargs

  def map(self, element: dict[str, Any]) -> dict[str, Any]:
    res = apply_chat_template(
        example=element,
        tokenizer_model=self.tokenizer_model,
        data_column_name=self.data_column_name,
        sentinel_ids=self.sentinel_ids,
        max_target_length=self.max_target_length,
        unk_id=self.pad_id,
        chat_template=self.chat_template,
        sft_train_last_turn_only=self.sft_train_last_turn_only,
        sft_train_on_thoughts_only=self.sft_train_on_thoughts_only,
        **self.kwargs,
    )
    filtered = {k: res[k] for k in ("inputs", "targets") if k in res}
    if "images" in res:
      filtered["images"] = res["images"]
    return filtered



class HFDataSource(grain.RandomAccessDataSource):
  """A class that makes HuggingFace IterableDataset a grain datasource without random access support"""

  def __init__(
      self,
      dataset: "datasets.IterableDataset",
      dataloading_host_index: int,
      dataloading_host_count: int,
      num_threads: int,
      max_target_length: int,
      data_column_names: list[str],
  ):
    from datasets.distributed import split_dataset_by_node  # pylint: disable=import-outside-toplevel

    self._split_dataset_by_node = split_dataset_by_node
    self.dataset = dataset
    self.num_threads = num_threads
    self.dataloading_host_count = dataloading_host_count
    self.dataloading_host_index = dataloading_host_index
    self.max_target_lenth = max_target_length
    self.data_column_names = data_column_names
    if hasattr(dataset, "n_shards"):
      self.n_shards = dataset.n_shards
    else:
      self.n_shards = 1
    self._check_shard_count()
    self.dataset_shards = [dataloading_host_index * self.num_threads + i for i in range(self.num_threads)]
    self.datasets = [self._split_dataset_by_node(dataset, world_size=self.n_shards, rank=x) for x in self.dataset_shards]
    self.data_iters = []

  def _check_shard_count(self):
    if self.n_shards < (self.dataloading_host_count * self.num_threads):
      warnings.warn(
          f"WARNING: Inefficient dataloading. Your train or eval dataset contains {self.n_shards} shards, "
          "smaller than number of host loading data. This is known to lead to inefficient dataloading. See"
          "github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md#multihost-dataloading-best-practice"
      )
      self.n_shards = self.dataloading_host_count * self.num_threads

  def _update_shard(self, idx):
    """update shard"""
    new_shard = self.dataset_shards[idx] + self.dataloading_host_count * self.num_threads
    if new_shard < self.n_shards:
      max_logging.log(
          f"Updating host {self.dataloading_host_index} dataset {idx}, was on shard {self.dataset_shards[idx]}"
      )
      max_logging.log(f"New shard is {new_shard}")
      self.dataset_shards[idx] = new_shard
      self.datasets[idx] = self._split_dataset_by_node(
          self.dataset, world_size=self.n_shards, rank=self.dataset_shards[idx]
      )
      self.data_iters[idx] = iter(self.datasets[idx])
    else:
      raise StopIteration(f"Run out of shards on host {self.dataloading_host_index}, shard {new_shard} is not available")

  def __len__(self):
    """Return length of the HF dataset. Since HuggingFace IterableDataset does not have length,
    a fake length bigger than the dataset is returned"""
    return 10_000_000_000

  def __getitem__(self, index):
    """Since HuggingFace IterableDataset does not support random access by index.
    The next item in the iterator is returned."""
    if not self.data_iters:
      self.data_iters = [iter(x) for x in self.datasets]
    idx = int(current_thread().name.split("_")[1])

    while True:
      try:
        data = next(self.data_iters[idx])
        return data
      except StopIteration:
        self._update_shard(idx)


########## Functions used by Grain pipeline


class _GCSTFRecordReader(_TFRecordReader):
  """Extends Grain's _TFRecordReader to open TFRecord files from GCS via streaming BlobReader."""

  def __init__(self, path: str):
    # Skip parent __init__ (which calls open(path, "rb")) and open via GCS BlobReader instead.
    bucket_name, blob_name = gcs_utils.parse_gcs_bucket_and_prefix(path)
    self._reader = gcs_utils.storage.Client().bucket(bucket_name).blob(blob_name).open("rb")


class _GCSTFRecordDatasetIterator(_TFRecordDatasetIterator):
  """Extends Grain's _TFRecordDatasetIterator to use _GCSTFRecordReader for GCS paths."""

  def __init__(self, path: str):
    # Skip parent __init__ (which creates _TFRecordReader); use GCS-aware reader instead.
    grain.DatasetIterator.__init__(self)
    self._reader = _GCSTFRecordReader(path)


class GCSTFRecordIterDataset(TFRecordIterDataset):
  """Extends Grain's TFRecordIterDataset to support GCS paths."""

  def __iter__(self) -> grain.DatasetIterator:  # pylint: disable=non-iterator-returned
    return _GCSTFRecordDatasetIterator(self._path)


def make_tfrecord_iter_dataset(path: str):
  """Returns the appropriate TFRecordIterDataset for local or GCS paths."""
  if path.startswith("gs://"):
    return GCSTFRecordIterDataset(path)
  return TFRecordIterDataset(path)


@dataclasses.dataclass
class ParseFeatures(grain.MapTransform):
  """Parse serialized tf.train.Example protos for arrayrecord/tfrecord datasets.

  Also validates that the stored field type matches `tokenize`: raises
  ValueError if `tokenize=True` but the column contains integers (pre-tokenized)
  or if `tokenize=False` but the column contains bytes (raw text).
  """

  def __init__(self, data_columns, tokenize):
    self.data_columns = data_columns
    self.tokenize = tokenize

  def map(self, element):
    """Parse a serialized tf.train.Example proto and extract features."""
    example = example_pb2.Example()
    example.ParseFromString(element)
    features = example.features.feature

    missing = [c for c in self.data_columns if c not in features]
    if missing:
      raise ValueError(
          f"Column {missing} not found in dataset. Available columns: {sorted(features.keys())}. "
          "Please set train_data_columns or eval_data_columns accordingly."
      )

    parsed = {}
    for col in self.data_columns:
      f = features[col]
      if self.tokenize:
        if not f.bytes_list.value:
          raise ValueError(
              f"tokenize_data=True but column '{col}' has no text (bytes) data. "
              "Set tokenize_train_data or tokenize_eval_data to False if your dataset is already tokenized."
          )
        parsed[col] = np.array(f.bytes_list.value, dtype=object)
      else:
        if not f.int64_list.value:
          raise ValueError(
              f"tokenize_data=False but column '{col}' has no integer token data. "
              "Set tokenize_train_data or tokenize_eval_data to True if your dataset needs tokenization."
          )
        parsed[col] = np.array(f.int64_list.value, dtype=np.int32)
    return parsed


@dataclasses.dataclass
class NormalizeFeatures(grain.MapTransform):
  """Normalize text feature keys."""

  def __init__(self, column_names, tokenize):
    self.column_names = column_names
    self.tokenize = tokenize

  def map(self, element):
    if self.tokenize:
      return {col: element[col][0].decode() for col in self.column_names}
    else:
      return {col: element[col] for col in self.column_names}


@dataclasses.dataclass
class KeepFeatures(grain.MapTransform):
  """Filter dataset elements to specified features for parquet and other non-proto formats.

  Retains only the keys present in `feature_names`. Validates the stored value
  type against `tokenize`: raises ValueError if `tokenize=True` but a column
  contains integer data (pre-tokenized), or if `tokenize=False` but a column
  contains string/bytes data (raw text).
  """

  def __init__(self, feature_names: list[str], tokenize: bool = True):
    self.feature_names = feature_names
    self.tokenize = tokenize

  def map(self, element: dict[str, Any]) -> dict[str, Any]:
    """Applies the feature filtering to the input element."""
    missing = [n for n in self.feature_names if n not in element]
    if missing:
      raise ValueError(
          f"Column {missing} not found in dataset. Available columns: {sorted(element.keys())}. "
          "Please set train_data_columns or eval_data_columns accordingly."
      )
    filtered = {k: v for k, v in element.items() if k in self.feature_names}
    for col, val in filtered.items():
      if self.tokenize:
        if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.integer):
          raise ValueError(
              f"tokenize_data=True but column '{col}' contains integer (pre-tokenized) data. "
              "Set tokenize_train_data or tokenize_eval_data to False if your dataset is already tokenized."
          )
        if isinstance(val, (list, tuple)) and val and isinstance(val[0], (int, np.integer)):
          raise ValueError(
              f"tokenize_data=True but column '{col}' contains integer (pre-tokenized) data. "
              "Set tokenize_train_data or tokenize_eval_data to False if your dataset is already tokenized."
          )
      else:
        if isinstance(val, (str, bytes)):
          raise ValueError(
              f"tokenize_data=False but column '{col}' contains text data. "
              "Set tokenize_train_data or tokenize_eval_data to True if your dataset needs tokenization."
          )
    return filtered


@dataclasses.dataclass
class Rekey(grain.MapTransform):
  """Rename keys according to a mapping dict"""

  def __init__(self, mapping_dict, keep_old_keys=False):
    self.mapping_dict = mapping_dict
    self.keep_old_keys = keep_old_keys

  def map(self, element):
    old_keys = set()
    for new_key, old_key in self.mapping_dict.items():
      element[new_key] = element[old_key]
      old_keys.add(old_key)
    if not self.keep_old_keys:
      for key in old_keys:
        del element[key]
    return element


@dataclasses.dataclass
class ReformatPacking(grain.MapTransform):
  """Reformat packing outputs."""

  def __init__(self, column_names):
    self.column_names = column_names

  def map(self, element):
    ret = {}
    for col in self.column_names:
      ret[f"{col}"] = element[0][col]
      ret[f"{col}_segmentation"] = element[1][col]
      ret[f"{col}_position"] = element[2][col]
    return ret


@dataclasses.dataclass
class PadOrTrimToMaxLength(grain.MapTransform):
  """Pads or trims each input to the specified length.
  And optionally add true length for the input."""

  def __init__(
      self,
      max_length: int,
      pad_id: int = 0,
      config=None,
      add_true_length: bool = False,
      max_num_images_per_example: int = -1,
  ):
    self.max_length = max_length
    self.pad_id = pad_id
    self.config = config
    self.add_true_length = add_true_length
    self.max_num_images_per_example = max_num_images_per_example

  def _pad_text(self, x: np.ndarray, max_length: int, pad_id: int) -> np.ndarray:
    pad_amount = max(max_length - x.shape[0], 0)
    pad_amount = [(0, pad_amount)] + [(0, 0)] * (len(x.shape) - 1)
    return np.pad(x, pad_amount, constant_values=pad_id)[: self.max_length]

  def _pad_image_and_mask(self, preprocessed_image: mm_utils.PreprocessorOutput) -> mm_utils.PreprocessorOutput:
    """Pads the input tensors (image and mask) of a PreprocessorOutput to a maximum number of items.

    This function unifies padding logic for image tensors (standard or tiled) and
    mask tensors. It determines the tensor type based on its dimensions and applies
    the appropriate padding along the first axis.

    The maximum number of items is calculated based on model constraints or a
    user-defined limit, ensuring that sequence length limits are respected while
    reserving space for at least one text token. If the input tensor has fewer
    items than this maximum, it is padded with zeros.

    Args:
        preprocessed_image (multimodal_utils.PreprocessorOutput): The input numpy arrays to pad.
            - For masks, the expected shape is (num_masks, num_tiles).
            - For standard images, the shape is (num_images, H, W, C).
            - For tiled images, the shape is (num_images, num_tiles, H, W, C).

    Returns:
        np.ndarray: The tensor, padded with zeros up to the maximum number of
        items along the first axis.

    Raises:
        ValueError: If the input tensor's dimension is not 2, 4, or 5.
        ValueError: If the number of items in the input tensor exceeds the
        allowed maximum.

    Notes:
      - The computation of maximum images ensures that space is reserved in the sequence
        for at least one text token.
      - The dummy images used for padding are based on the image shape for initialization
        of this model (ignoring batch size).
    """
    if not isinstance(preprocessed_image, mm_utils.PreprocessorOutput):
      raise TypeError(f"Input must be multimodal_utils.PreprocessorOutput, but got {type(preprocessed_image)}")

    if preprocessed_image.pixel_values is None:
      raise ValueError("Input preprocessed_image must have pixel_values to pad images.")

    if self.config.model_name and self.config.model_name.startswith("qwen3-omni"):
      return preprocessed_image

    # Determine the maximum number of images/masks allowed.
    image_offsets = mm_processor.get_image_offsets(self.config, preprocessed_image)
    single_image_offset = image_offsets // preprocessed_image.pixel_values.shape[0]

    # Reserve space for at least one text token.
    max_num_items = (self.max_length - 1) // single_image_offset
    if self.max_num_images_per_example > 0:
      max_num_items = min(self.max_num_images_per_example, max_num_items)

    image_tensor = preprocessed_image.pixel_values
    mask_tensor = preprocessed_image.pixel_mask

    def _pad(tensor: np.ndarray) -> np.ndarray:
      # Validate tensor dimensions.
      if tensor.ndim in (4, 5):  # Standard or Tiled Image
        tensor_type = "images"
      elif tensor.ndim == 2:  # Mask
        tensor_type = "masks"
      else:
        raise ValueError(
            "Input tensor must be 2D (mask), 4D (image), or 5D (tiled image), " f"but got {tensor.ndim} dimensions."
        )

      # Assert that the input tensor does not exceed the maximum size.
      if tensor.shape[0] > max_num_items:
        raise ValueError(f"Number of {tensor_type} ({tensor.shape[0]}) exceeds the maximum allowed ({max_num_items}).")

      # Apply padding if the tensor is smaller than the maximum size.
      if tensor.shape[0] < max_num_items:
        pad_size = max_num_items - tensor.shape[0]
        pad_shape_suffix = tensor.shape[1:]
        pad_shape = (pad_size,) + pad_shape_suffix
        pad_tensor = np.zeros(pad_shape, dtype=tensor.dtype)

        if tensor.size > 0:
          tensor = np.concatenate([tensor, pad_tensor], axis=0)
        else:
          # If the input tensor is empty, the result is just the padding.
          tensor = pad_tensor

      return tensor

    preprocessed_image.pixel_values = _pad(image_tensor)

    if mask_tensor is not None:
      preprocessed_image.pixel_mask = _pad(mask_tensor)

    return preprocessed_image

  def map(
      self, element: dict[str, np.ndarray | mm_utils.PreprocessorOutput]
  ) -> dict[str, np.ndarray | mm_utils.PreprocessorOutput]:
    """map to each element"""
    data_columns = list(element.keys())
    for data_column in data_columns:
      if (
          data_column != "images"
          and not data_column.endswith("_position")
          and not data_column.endswith("_segmentation")
          and not data_column.endswith("_true_length")
      ):
        if isinstance(element[data_column], mm_utils.PreprocessorOutput):
          raise TypeError("Only 'images' column can be of type PreprocessorOutput.")

        element[f"{data_column}_segmentation"] = element[data_column] != self.pad_id
        element[f"{data_column}_segmentation"] = element[f"{data_column}_segmentation"].astype(np.int32)
        element[f"{data_column}_position"] = np.arange(element[data_column].shape[0], dtype=np.int32)
        if self.add_true_length:
          element[f"{data_column}_true_length"] = np.array([element[data_column].shape[0]], dtype=np.int32)

    for key, _ in element.items():
      if key == "images":
        if self.config.model_name is None:
          raise ValueError("model_name must be provided when padding images")

        element["images"] = self._pad_image_and_mask(element["images"])

      elif "true_length" not in key:
        element[key] = self._pad_text(element[key], self.max_length, self.pad_id)
    return element


@dataclasses.dataclass
class ExtractImagesAndMasks(grain.MapTransform):
  """Extracts images and masks from a PreprocessorOutput object.

  This transform is used in multi-modal data pipelines to extract the image
  tensors and their corresponding masks from a PreprocessorOutput object.
  The extracted images and masks are then added to the data element under
  the keys 'images' and 'image_masks', respectively.

  If the 'images' key is not present in the input element, the transform
  returns the element unchanged.
  """

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Applies the extraction transformation to the 'images' field if present."""
    preprocessed_image = element.get("images")
    if preprocessed_image is None:
      return element

    if not isinstance(preprocessed_image, mm_utils.PreprocessorOutput):
      raise TypeError(f"'images' must be of type PreprocessorOutput, but got {type(preprocessed_image)}")

    output = element.copy()
    output["images"] = preprocessed_image.pixel_values
    if preprocessed_image.pixel_mask is not None:
      output["image_masks"] = preprocessed_image.pixel_mask

    return output


@dataclasses.dataclass
class FoldImagesIntoBatch(grain.MapTransform):
  """Folds the 'image' dimension into the batch dimension.

  This transform is used in multi-modal data pipelines where each data example
  might have multiple associated images. For model processing, it's often
  efficient to treat each image as a separate item in a larger batch.

  This operation reshapes the 'images' tensor from a shape like
  (B, N, T, H, W, C) to (B * N, T, H, W, C), where B is the batch size, N is
  the number of images per example, and T is the number of image tiles.

  The transformation is triggered only if the input 'images' tensor has more
  dimensions than the expected batched image tensor.
  """

  model_name: str | None = None

  def __post_init__(self):
    """Initializes the target shape after the dataclass is created."""
    self.target_shape = mm_processor.get_dummy_image_shape_for_init(self.model_name)

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Applies the folding transformation to the 'images' field if present."""
    images = element.get("images")
    if images is None:
      return element

    # If ndim is greater than the expected ndim for a batched image tensor,
    # it implies an extra dimension (e.g., number of images per example)
    # that needs to be folded into the batch dimension.
    if images.ndim > len(self.target_shape):
      # Compute the new shape by merging the batch and image count dimensions.
      trailing_dims = self.target_shape[1:]

      # Reshape merges the leading dimensions (B, N) into one (-1) and
      # appends the correct trailing dimensions.
      element["images"] = images.reshape(-1, *trailing_dims)

    return element


def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  slices = [
      slice(None),
  ] * len(x.shape)
  slices[axis] = slice(0, -1)
  padded = np.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(0))
  return padded[tuple(slices)]


def shift_left(x, pad_id, axis=1):
  """Shift to the left and pad."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (0, 1)
  slices = [
      slice(None),
  ] * len(x.shape)
  slices[axis] = slice(1, None)
  padded = np.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(pad_id))
  return padded[tuple(slices)]


def shift_and_refine(x, ignored_ids, axis=1):
  """Shift inputs, set segmentation to 0 when target element is in ignored_ids if provided"""
  x["targets"] = shift_left(x["targets"], ignored_ids[0], axis=axis)
  x["targets_segmentation"] = shift_left(x["targets_segmentation"], 0, axis=axis)
  for ignore_id in ignored_ids:
    x["targets_segmentation"] = np.where(x["targets"] != ignore_id, x["targets_segmentation"], 0)

  return x


@dataclasses.dataclass
class ShiftData(grain.MapTransform):
  """Shift inputs and refine annotations."""

  def __init__(self, ignored_ids, axis=1):
    self.ignored_ids = ignored_ids
    self.axis = axis

  def map(self, element):
    return shift_and_refine(element, ignored_ids=self.ignored_ids, axis=self.axis)


@dataclasses.dataclass
class ComputeQwen3OmniPositions(grain.MapTransform):
  """Computes 3D position IDs for Qwen3-Omni multimodal sequences.

  This transform replaces the standard 1D sequential positions with 3D
  positions (temporal, height, width) for multimodal models like Qwen3-Omni.

  For text-only sequences, all 3 dimensions receive the same sequential values.
  For multimodal sequences with vision/audio, vision tokens get true 3D positions
  and text tokens continue sequentially from max(vision_pos) + 1.

  The actual position computation is delegated to multimodal_utils.get_rope_index(),
  which can be tested and modified independently.
  """

  def __init__(
      self,
      data_column: str = "inputs",
      spatial_merge_size: int = 2,
      position_id_per_seconds: int = 25,
      use_audio_in_video: bool = False,
  ):
    """Initialize the Qwen3-Omni position computation transform.

    Args:
      data_column: Name of the data column to compute positions for (default: "inputs").
      spatial_merge_size: Number of patches merged spatially (e.g., 2 for 2x2→1).
      position_id_per_seconds: Temporal granularity (tokens per second, typically 25).
      use_audio_in_video: If True, audio tokens are interleaved with video tokens.
    """
    self.data_column = data_column
    self.spatial_merge_size = spatial_merge_size
    self.position_id_per_seconds = position_id_per_seconds
    self.use_audio_in_video = use_audio_in_video

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute 3D position IDs for the batch element.

    Args:
      element: Dictionary containing:
        - {data_column}: Token IDs with shape (batch, seq_len)
        - {data_column}_segmentation: Attention mask (1=real, 0=padding)
        - image_grid_thw: Optional (num_images, 3) array
        - video_grid_thw: Optional (num_videos, 3) array
        - audio_lengths: Optional (num_audios,) array
        - second_per_grids: Optional (num_videos,) array

    Returns:
      element with {data_column}_position updated to shape (3, batch, seq_len)
      for 3D positions (always 3D, even for text-only sequences).
    """

    # Extract inputs and metadata
    input_ids = element[self.data_column]
    attention_mask = element.get(f"{self.data_column}_segmentation")

    # Extract multimodal metadata (if present)
    image_grid_thw = element.get("image_grid_thw")
    video_grid_thw = element.get("video_grid_thw")
    audio_lengths = element.get("audio_lengths")
    second_per_grids = element.get("second_per_grids")

    # Call the standalone get_rope_index function from multimodal_utils
    from maxtext.multimodal import processor_qwen3_omni  # pylint: disable=import-outside-toplevel

    # TODO(jfacevedo/hengtaoguo): Now get_rope_index is Qwen3-Omni specific. We should generalize it for other models
    position_ids, mrope_position_deltas = processor_qwen3_omni.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        use_audio_in_video=self.use_audio_in_video,
        audio_lengths=audio_lengths,
        second_per_grids=second_per_grids,
        spatial_merge_size=self.spatial_merge_size,
        position_id_per_seconds=self.position_id_per_seconds,
    )

    # Update element with 3D positions
    # Shape: (3, batch, seq_len) for multimodal, or (batch, seq_len) for text-only
    element[f"{self.data_column}_position"] = position_ids
    element[f"{self.data_column}_mrope_deltas"] = mrope_position_deltas

    return element
