# coding=utf-8
# Copyright 2019 The Google UDA Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/bin/bash
vocab_file=bert_pretrained/NR_BASE_E20_SP_V32217/V32217.vocab
model_file=bert_pretrained/NR_BASE_E20_SP_V32217/V32217.model


# Preprocess supervised training set
python preprocess.py \
  --raw_data_dir=data/dataset/v2 \
  --output_base_dir=data/prepro_data/v2/train_239 \
  --data_type=sup \
  --sub_set=train \
  --sup_size=239 \
  --vocab_file=$vocab_file \
  --model_file=$model_file \
  $@

# Preprocess test set
python preprocess.py \
  --raw_data_dir=data/dataset/v2 \
  --output_base_dir=data/prepro_data/v2/dev \
  --data_type=sup \
  --sub_set=dev \
  --sup_size=60 \
  --vocab_file=$vocab_file \
  --model_file=$model_file \
  $@


# Preprocess unlabeled set
python preprocess.py \
  --raw_data_dir=data/dataset/v2 \
  --output_base_dir=data/prepro_data/v2/unsup \
  --back_translation_dir=data/dataset/v2/backtranslated \
  --data_type=unsup \
  --sub_set=unsup_in \
  --aug_ops=bt-1.0 \
  --aug_copy_num=0 \
  --vocab_file=$vocab_file \
  --model_file=$model_file \
  $@
