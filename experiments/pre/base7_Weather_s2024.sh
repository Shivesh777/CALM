#!/bin/bash
export SEED_OFFSET=3   # their code: fix_seed = 2021 + ii + SEED_OFFSET -> seed 2024
# Proceed-protocol baselines (iTransformer, L=512) with prediction dumps for pooling as external experts.
# Filled per dataset by make_pre7.py: DS (their name), PLR (pretrain lr), PBS, GDLR, PCBS, MID, BTL, LR, OLR24, OLR48, OLR96
W=/kaggle/working
chain () {
  H=$1; G=$2; OLR=$3
  COMMON="--dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len $H --itr 1 --gpu $G"
  python -u run.py $COMMON --learning_rate 0.00001 --batch_size 32 --save_opt > $W/baseline_logs/pretrain_Weather_H$H.log 2>&1
  DUMP_PREDS=$W/preds/GD__Weather__L512__H$H python -u run.py $COMMON --online_method Online --online_learning_rate 0.0000003 \
      --pin_gpu True --reduce_bs False --save_opt --only_test > $W/baseline_logs/gd_Weather_H$H.log 2>&1
  DUMP_PREDS=$W/preds/Proceed__Weather__L512__H$H python -u run.py $COMMON --batch_size 16 --tune_mode down_up \
      --concept_dim 100 --val_online_lr --diff_online_lr --pretrain --save_opt --online_method Proceed \
      --bottleneck_dim 24 --online_learning_rate $OLR --learning_rate 0.001 --lradj type3 > $W/baseline_logs/proceed_Weather_H$H.log 2>&1
  grep -h "^mse:" $W/baseline_logs/*_Weather_H$H.log
}
chain 24 0 0.000003 &
chain 48 1 0.000003 &
wait
chain 96 0 0.000003
ls -la $W/preds
