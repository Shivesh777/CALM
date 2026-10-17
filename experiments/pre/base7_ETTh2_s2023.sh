#!/bin/bash
export SEED_OFFSET=2   # their code: fix_seed = 2021 + ii + SEED_OFFSET -> seed 2023
# Proceed-protocol baselines (iTransformer, ETTh2, L=336) with prediction dumps for pooling as external experts.
W=/kaggle/working
chain () {
  H=$1; G=$2
  COMMON="--dataset ETTh2 --border_type online --model iTransformer --seq_len 336 --pred_len $H --itr 1 --gpu $G"
  python -u run.py $COMMON --learning_rate 0.0001 --batch_size 32 --save_opt > $W/baseline_logs/pretrain_H$H.log 2>&1
  DUMP_PREDS=$W/preds/GD__ETTh2__L336__H$H python -u run.py $COMMON --online_method Online --online_learning_rate 0.00003 \
      --pin_gpu True --reduce_bs False --save_opt --only_test > $W/baseline_logs/gd_H$H.log 2>&1
  DUMP_PREDS=$W/preds/Proceed__ETTh2__L336__H$H python -u run.py $COMMON --batch_size 32 --tune_mode down_up --concept_dim 50 \
      --val_online_lr --diff_online_lr --pretrain --save_opt --online_method Proceed --bottleneck_dim 48 \
      --online_learning_rate 0.00003 --learning_rate 0.003 --lradj type3 > $W/baseline_logs/proceed_H$H.log 2>&1
  grep -h "^mse:" $W/baseline_logs/*_H$H.log
}
chain 24 0 &
chain 48 1 &
wait
chain 96 0
ls -la $W/preds
