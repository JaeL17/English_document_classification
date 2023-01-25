#!/bin/bash


export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

DPATH_OUTPUT= ../../inference_results
#/docker//data/mid/result/0221_output_test

mkdir $DPATH_OUTPUT/

pushd ../../code

python infer3.py \
--input_file "../../mega_trend_data/cls_data/test_v4.tsv" \
--output_file "$DPATH_OUTPUT/infer_classification_top1.txt" \
--model_type "saltluxbert" \
--model_name_or_path "../model_results_large/cls/slx_bert_large/checkpoint-27000" \
--max_seq_len 512 \
--eval_batch_size 32 \
--buff_size 4096 \
--top_n 3 \
--do_infer

popd

