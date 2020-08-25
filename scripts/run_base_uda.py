import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.system('python main.py \
  --use_tpu=False \
  --do_train=True \
  --do_eval=True \
  --sup_train_data_dir=data/prepro_data/v2/train_239 \
  --unsup_data_dir=data/prepro_data/v2/unsup \
  --eval_data_dir=data/prepro_data/v2/dev \
  --raw_data_dir=data/dataset/v2 \
  --bert_config_file=bert_pretrained/NR_BASE_E20_SP_V32217/bert_config.json \
  --vocab_file=bert_pretrained/NR_BASE_E20_SP_V32217/V32217.vocab\
  --model_file=bert_pretrained/NR_BASE_E20_SP_V32217/V32217.model\
  --init_checkpoint=bert_pretrained/NR_BASE_E20_SP_V32217/model.ckpt \
  --task_name=pct \
  --model_dir=ckpt/base_uda \
  --num_train_steps=10000 \
  --learning_rate=2e-05 \
  --num_warmup_steps=1000 \
  --unsup_ratio=3 \
  --tsa=linear_schedule \
  --aug_ops=bt-1.0 \
  --uda_confidence_thresh=0.8 \
  --aug_copy=1 \
  --uda_coeff=1')