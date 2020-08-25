import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.system('python main.py \
  --use_tpu=False \
  --do_train=True \
  --do_eval=True \
  --sup_train_data_dir=data/prepro_data/v2/train_239 \
  --eval_data_dir=data/prepro_data/v2/dev \
  --raw_data_dir=data/dataset/v2 \
  --bert_config_file=bert_pretrained/NR_BASE_E20_SP_V32217/_config.json \
  --vocab_file=bert_pretrained/NR_BASE_E20_SP_V32217/V32217.vocab\
  --model_file=bert_pretrained/NR_BASE_E20_SP_V32217/V32217.model\
  --init_checkpoint=bert_pretrained/NR_BASE_E20_SP_V32217/model.ckpt \
  --task_name=pct \
  --model_dir=ckpt/base \
  --num_train_steps=2000 \
  --learning_rate=3e-05 \
  --num_warmup_steps=300')