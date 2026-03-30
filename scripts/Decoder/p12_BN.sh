#!/bin/bash
# 只跑 Decoder-only 架构的脚本

root_path_name=./dataset/
data_path_name=Physionet2012
model_id_name=Physionet2012
data_name=Physionet2012

enc_in=37
dec_in=37
c_out=37

gpu_num=0
random_seed=2021

# 论文设定：单模块模型 (Encoder-only, Decoder-only) 堆叠 6 层
e_layers=0  # Decoder-only 不需要 encoder 层
d_layers=6  # 设定为 6 层
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

# 直接指定模型为 Decoder (即 Decoder-only)
model_name=Decoder

for seq_len in 24 36
do
for pred_len in 6 12
do
    label_len=$pred_len

    echo "=============================================="
    echo "Running: $model_name | seq=$seq_len pred=$pred_len | BN"
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
      --des 'Exp_DecoderOnly' \
      --itr 1 \
      --learning_rate $learning_rate \
      --train_epochs $train_epochs \
      --patience $patience \
      --patch_len $patch_len \
      --stride $stride \
      --gpu $gpu_num \
      --batch_size $batch_size \
      --run_train --run_test \
      --norm batch \
      --time_aware \
      --time_embed_type continuous \
      --freq h
done
done