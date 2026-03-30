#!/bin/bash
# ==============================================================================
# PhysioNet 2012 Forecasting Aggregation 对比实验 (LayerNorm)
# 对比 Encoder_overall, Encoder_zeros_flatten, Encoder_zeros_no_flatten
# 三者都是 Masked Encoder 架构，区别在于预测头的聚合方式:
#   - Encoder_overall:        flatten 全部 (input+output) patch → 线性映射
#   - Encoder_zeros_flatten:  只取 output patch → flatten → 线性映射
#   - Encoder_zeros_no_flatten: 只取 output patch → 逐 patch 线性映射 → flatten
# ==============================================================================

root_path_name=./dataset/
data_path_name=Physionet2012
model_id_name=Physionet2012
data_name=Physionet2012

enc_in=37
dec_in=37
c_out=37

gpu_num=0
random_seed=2021

# 单模块模型: 6 层
e_layers=6
d_layers=0

d_model=32
d_ff=64
n_heads=4
batch_size=64
learning_rate=0.001
train_epochs=30
patience=5
patch_len=6
stride=6

target=HR

# ==============================================================================
# 遍历 3 种 Masked Encoder 变体 × 不同 seq_len/pred_len 组合
# ==============================================================================

for model_name in Encoder_overall
do
# GraFITi Table 3: 24→12, 24→24, 36→6, 36→12
for pair in 24,12 24,24 36,6 36,12
do
    seq_len=${pair%,*}
    pred_len=${pair#*,}
    label_len=$pred_len

    echo "=============================================="
    echo "Running: $model_name | seq=$seq_len pred=$pred_len | LN"
    echo "=============================================="

    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_${model_name}_${seq_len}_${pred_len} \
      --model $model_name \
      --data $data_name \
      --features M \
      --target $target \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --d_layers $d_layers \
      --factor 3 \
      --enc_in $enc_in \
      --dec_in $dec_in \
      --c_out $c_out \
      --d_model $d_model \
      --d_ff $d_ff \
      --n_heads $n_heads \
      --des 'Exp' \
      --itr 1 \
      --learning_rate $learning_rate \
      --train_epochs $train_epochs \
      --patience $patience \
      --patch_len $patch_len \
      --stride $stride \
      --gpu $gpu_num \
      --batch_size $batch_size \
      --run_train --run_test \
      --norm layer \
      --time_aware \
      --time_embed_type continuous \
      --freq h

done
done