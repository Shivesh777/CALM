#!/bin/bash
# DynaME-protocol baselines (L=96, 20/5/75): pretrain, GD, Proceed, DynaME with prediction dumps.
W=/kaggle/working
M=iTransformer
chain () {
  H=$1; G=$2; OLR=$3
  COMMON="--dataset Weather --border_type online --model $M --seq_len 96 --pred_len $H --itr 1 --gpu $G"
  python -u run.py $COMMON --learning_rate 0.00001 --batch_size 32 --save_opt > $W/baseline_logs/pretrain_${M}_Weather_H$H.log 2>&1
  DUMP_PREDS=$W/preds/GD__${M}__Weather__L96__H$H python -u run.py $COMMON --online_method Online --online_learning_rate 0.0000003 \
      --pin_gpu True --reduce_bs False --save_opt --only_test > $W/baseline_logs/gd_${M}_Weather_H$H.log 2>&1
  DUMP_PREDS=$W/preds/Proceed__${M}__Weather__L96__H$H python -u run.py $COMMON --batch_size 16 --tune_mode down_up \
      --concept_dim 100 --val_online_lr --diff_online_lr --pretrain --save_opt --online_method Proceed \
      --bottleneck_dim 24 --online_learning_rate $OLR --learning_rate 0.001 --lradj type3 > $W/baseline_logs/proceed_${M}_Weather_H$H.log 2>&1
  DUMP_PREDS=$W/preds/DynaME__${M}__Weather__L96__H$H python -u run.py $COMMON --batch_size 16 --pretrain --save_opt \
      --online_method DynaME --online_learning_rate 0.0000003 --learning_rate 0.00001 --lradj type3 \
      --krr_train_num 8 --period_num 2 --beta 0.3 --past_num 672 --krr_lambda 0.0001 --delta 0.01 > $W/baseline_logs/dyname_${M}_Weather_H$H.log 2>&1
  grep -h "^mse:" $W/baseline_logs/*_${M}_Weather_H$H.log
}
chain 24 0 0.000003 &
chain 48 1 0.000003 &
wait
chain 96 0 0.000003
ls -la $W/preds
