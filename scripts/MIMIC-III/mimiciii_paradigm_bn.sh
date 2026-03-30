#!/bin/bash
# ==============================================================================
# MIMIC-III 预测范式对比 Direct vs Autoregressive (BatchNorm)
# 64 变量, 48h ICU, 随机 70/20/10 划分
# 观测/预测: 24→12, 24→24, 36→6, 36→12
# ==============================================================================

root_path_name=./dataset/
data_path_name=MIMIC-III
model_id_name=MIMICIII
data_name=MIMICIII

enc_in=64
dec_in=64
c_out=64

gpu_num=0
random_seed=2021

e_layers=3
d_layers=3
d_model=128
d_ff=256
n_heads=4
batch_size=64
learning_rate=0.001
train_epochs=30
patience=5
patch_len=6
stride=6
target=HR

for model_name in Transformer Transformer_autoregressive Decoder Decoder_autoregressive
do
for pair in 24,12 24,24 36,6 36,12
do
    seq_len=${pair%,*}
    pred_len=${pair#*,}
    label_len=$pred_len

    if [ "$model_name" = "Decoder" ] || [ "$model_name" = "Decoder_autoregressive" ]; then
        cur_e_layers=0; cur_d_layers=6
    elif [ "$model_name" = "Transformer" ] || [ "$model_name" = "Transformer_autoregressive" ]; then
        cur_e_layers=3; cur_d_layers=3
    fi

    echo "=============================================="
    echo "Running: $model_name | seq=$seq_len pred=$pred_len | BN"
    echo "=============================================="

    python -u run_longExp.py \
      --random_seed $random_seed --is_training 1 \
      --root_path $root_path_name --data_path $data_path_name \
      --model_id ${model_id_name}_${model_name}_${seq_len}_${pred_len} \
      --model $model_name --data $data_name \
      --features M --target $target \
      --seq_len $seq_len --label_len $label_len --pred_len $pred_len \
      --e_layers $cur_e_layers --d_layers $cur_d_layers --factor 3 \
      --enc_in $enc_in --dec_in $dec_in --c_out $c_out \
      --d_model $d_model --d_ff $d_ff --n_heads $n_heads \
      --des 'Exp' --itr 1 \
      --learning_rate $learning_rate --train_epochs $train_epochs --patience $patience \
      --patch_len $patch_len --stride $stride \
      --gpu $gpu_num --batch_size $batch_size \
      --run_train --run_test \
      --norm batch --time_aware --time_embed_type continuous --freq h

done
done