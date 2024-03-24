#!/bin/bash

export NUM_TRAIN_EPOCHS=2
export WANDB_PROJECT=mistral-7b-hf-8k
export MODEL_PATH=./init_model/vietnamese-mistral-7b
export OUTPUT_DIR=./mistral-7b-hf-8k
export SAVE_STEPS=1000
export LOGGING_STEPS=100
export DATASET_PATH=./processed_data

deepspeed run_clm.py \
--deepspeed ./configs/ds_config_zero3.json \
--model_name_or_path $MODEL_PATH \
--per_device_train_batch_size 32 \
--gradient_accumulation_steps 1 \
--output_dir $OUTPUT_DIR \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs $NUM_TRAIN_EPOCHS \
--dataset_path $DATASET_PATH \
--logging_steps $LOGGING_STEPS \
--learning_rate 2e-5 \
--save_steps $SAVE_STEPS \
--save_total_limit 10 \
--gradient_checkpointing true