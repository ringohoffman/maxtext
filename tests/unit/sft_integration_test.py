# Copyright 2025 Google LLC
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

"""Integration tests for the standard redesigned SFT chat template masking dataloading pipelines.

These tests run the full Grain and HuggingFace SFT preprocessing pipelines using
local assets (no GCS or HF Hub network access required).
"""

import os
import unittest
from typing import Any
from unittest.mock import MagicMock

import jax
import grain.python as grain
import ml_collections
import numpy as np
import pytest
import transformers
from datasets import Dataset
from jax.sharding import Mesh
from jax.experimental import mesh_utils

from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_PKG_DIR, MAXTEXT_CONFIGS_DIR, MAXTEXT_TEST_ASSETS_ROOT
from maxtext.input_pipeline import grain_data_processing
from maxtext.input_pipeline import hf_data_processing
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.input_pipeline.hf_data_processing import _get_pad_id


MESSAGES_DATA = [
    [
        {"content": "You are a helpful assistant.", "role": "system"},
        {"content": "What is 2+2?", "role": "user"},
        {"content": "The answer is 4.", "role": "assistant"},
    ],
    [
        {"content": "And 3+3?", "role": "user"},
        {"content": "That would be 6.", "role": "assistant"},
    ]
]


class LocalInMemoryDataSource(grain.RandomAccessDataSource):
  """Custom local DataSource that does not use shared memory to prevent multiprocessing KeyError."""

  def __init__(self, data: list[Any]) -> None:
    self._data = data

  def __len__(self) -> int:
    return len(self._data)

  def __getitem__(self, index: int) -> Any:
    return self._data[index]


@pytest.mark.cpu_only
class TestSFTDataPipelineIntegration(unittest.TestCase):
  """Integration tests running SFT pipelines with local tokenizers."""

  @classmethod
  def setUpClass(cls) -> None:
    cls.local_tokenizer_path = os.path.join(
        MAXTEXT_TEST_ASSETS_ROOT, "tokenizers", "qwen3-tokenizer"
    )
    if not os.path.exists(cls.local_tokenizer_path):
      raise unittest.SkipTest(f"Qwen3 local tokenizer not found at {cls.local_tokenizer_path}. Skipping integration test.")

  def _create_config(self) -> ml_collections.ConfigDict:
    config = pyconfig.initialize(
        [
            os.path.join(MAXTEXT_PKG_DIR, "sft_trainer"),
            os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "sft.yml"),
            "grain_num_threads=1",
        ],
        per_device_batch_size=1,
        run_name="test_run",
        mesh_axes=["data"],
        logical_axis_rules=[["batch", "data"]],
        data_sharding=["data"],
        base_output_directory="gs://test-experiments/",
        tokenizer_path=self.local_tokenizer_path,
        tokenizer_type="huggingface",
        train_split="train",
        enable_checkpointing=False,
        use_sft=True,
        enable_data_shuffling=False,
        max_target_length=64,
        max_prefill_predict_length=16,
        sft_train_on_completion_only=True,
        grain_file_type="",
    )
    return config

  def setUp(self) -> None:
    super().setUp()
    self.mesh_shape_1d = (len(jax.devices()),)
    self.mesh = Mesh(mesh_utils.create_device_mesh(self.mesh_shape_1d), ["data"])
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        self.local_tokenizer_path,
        add_bos_token=False,
        add_eos_token=False,
        legacy=False,
    )

  def test_huggingface_pipeline_redesign_integration(self) -> None:
    """Test HuggingFace preprocessing pipeline runs successfully with SFT masking."""
    config = self._create_config()
    dataset = Dataset.from_dict({"messages": MESSAGES_DATA * 32})
    
    process_indices = input_pipeline_interface.get_process_loading_real_data(
        config.data_sharding,
        config.global_batch_size_to_load,
        config.global_batch_size_to_train_on,
        config.max_target_length,
        self.mesh,
    )
    
    data_iter = hf_data_processing.preprocessing_pipeline(
        dataloading_host_index=process_indices.index(jax.process_index()),
        dataloading_host_count=len(process_indices),
        global_mesh=self.mesh,
        dataset=dataset,
        config=config,
        data_column_names=["messages"],
        tokenize=True,
        tokenizer_path=self.local_tokenizer_path,
        hf_access_token=None,
        global_batch_size=config.global_batch_size_to_load,
        max_target_length=config.max_target_length,
        shuffle=False,
        data_shuffle_seed=0,
        add_bos=False,
        add_eos=False,
        packing=False,
        generate_padding_batch=False,
        use_dpo=False,
        use_sft=True,
        sft_train_on_completion_only=True,
        grain_worker_count=0,
    )

    batch = next(data_iter)
    self.assertIn("inputs", batch)
    self.assertIn("targets", batch)
    self.assertEqual(batch["inputs"].shape, (1, 64))
    self.assertEqual(batch["targets"].shape, (1, 64))
    
    targets = batch["targets"][0]
    pad_id = _get_pad_id(self.tokenizer)
    active_positions = np.where((targets != pad_id) & (targets != 0))[0]
    self.assertGreater(len(active_positions), 0, "Targets should have some non-masked tokens.")
    self.assertLess(len(active_positions), 64, "Targets should have some masked tokens.")

  def test_grain_pipeline_redesign_integration(self) -> None:
    """Test Grain preprocessing pipeline runs successfully with SFT masking."""
    config = self._create_config()
    raw_source = LocalInMemoryDataSource(
        [{"messages": MESSAGES_DATA[i % 2]} for i in range(128)]
    )
    dataset = grain.MapDataset.source(raw_source)
    
    pipeline = grain_data_processing.sft_preprocessing_pipeline(
        dataset=dataset,
        config=config,
        data_columns=["messages"],
        tokenize=True,
        grain_worker_count=0,
        grain_per_worker_buffer_size=1,
    )
    
    data_iter = iter(pipeline)
    batch = None
    pad_id = _get_pad_id(self.tokenizer)
    active_row_targets = None
    
    for b_idx, b in enumerate(data_iter):
      targets_arr = b["targets"]
      for b_idx_row in range(targets_arr.shape[0]):
        tgt = targets_arr[b_idx_row]
        non_pad = np.sum((tgt != pad_id) & (tgt != 0))
        if non_pad > 0:
          batch = b
          active_row_targets = tgt
          break
        
    self.assertIsNotNone(batch, "SFT integration test failed: Sequence packing did not emit any batch containing trainable tokens.")

    self.assertIn("inputs", batch)
    self.assertIn("targets", batch)
    self.assertEqual(batch["inputs"].shape[1], 64)
    self.assertEqual(batch["targets"].shape[1], 64)
    
    active_positions = np.where((active_row_targets != pad_id) & (active_row_targets != 0))[0]
    self.assertGreater(len(active_positions), 0, "Targets should have some non-masked tokens.")
    self.assertLess(len(active_positions), 64, "Targets should have some masked tokens.")


if __name__ == "__main__":
  unittest.main()
