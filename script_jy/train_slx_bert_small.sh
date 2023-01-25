#!/bin/bash


export CUDA_VISIBLE_DEVICES=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID

DIR_OUTPUT=../model_results/v_a/slx_bert_small_lr5e5
mkdir $DIR_OUTPUT
mkdir $DIR_OUTPUT/data

pushd ../code

python train.py \
--task "cl_patent" \
--data_dir "$DIR_OUTPUT/data" \
--ckpt_dir "" \
--train_file "../../mega_trend_data/data_v_a/train.tsv" \
--dev_file "../../mega_trend_data/data_v_a/dev.tsv" \
--test_file "../../mega_trend_data/data_v_a/test.tsv" \
--model_type "saltluxbert" \
--model_name_or_path "../data/raw/patent-bert-small" \
--output_dir $DIR_OUTPUT \
--max_seq_len 512 \
--num_train_epochs 10 \
--gradient_accumulation_steps 1 \
--warmup_proportion 0 \
--max_steps -1 \
--seed 42 \
--train_batch_size 16 \
--eval_batch_size 16 \
--logging_steps -1 \
--save_steps -1 \
--weight_decay 0.0 \
--adam_epsilon 1e-8 \
--max_grad_norm 1.0 \
--learning_rate 5e-5 \
--do_train \
--do_eval \
--eval_all_checkpoints \



popd