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
export CUDA_VISIBLE_DEVICES=0

python main.py \
  --use_tpu=False \
  --do_train=True \
  --do_eval=True \
  --sup_train_data_dir=data/prepro_data/v2/train_239 \
  --eval_data_dir=data/prepro_data/v2/dev \
  --raw_data_dir=data/dataset/v2 \
  --bert_config_file=bert_pretrained/NR_BASE_E20_SP_V32217/bert_config.json \
  --vocab_file=bert_pretrained/NR_BASE_E20_SP_V32217/V32217.vocab\
  --model_file=bert_pretrained/NR_BASE_E20_SP_V32217/V32217.model\
  --init_checkpoint=bert_pretrained/NR_BASE_E20_SP_V32217/model.ckpt \
  --task_name=pct \
  --model_dir=ckpt/base \
  --num_train_steps=3000 \
  --learning_rate=3e-05 \
  --num_warmup_steps=300 \
  $@
