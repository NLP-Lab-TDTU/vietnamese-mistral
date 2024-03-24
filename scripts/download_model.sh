#!/bin/bash

export HF_HUB_ENABLE_HF_TRANSFER=1

files=(
    "config.json"
    "generation_config.json"
    "model-00001-of-00003.safetensors"
    "model-00002-of-00003.safetensors"
    "model-00003-of-00003.safetensors"
    "model.safetensors.index.json"
)

for file in "${files[@]}"
do
    echo "Downloading $file"
    huggingface-cli download \
    --repo-type model \
    --local-dir ./init_model/vietnamese-mistral-7b \
    --local-dir-use-symlinks False \
    mistralai/Mistral-7B-Instruct-v0.2 $file
done