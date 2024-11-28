#!/bin/bash

CKPT="llava-v1.5-7b"
FILE="llava_v1_5_mix1k.jsonl"

python -m llava.eval.analyze_attn_dispersion \
    --model-path /path/to/checkpoint/${CKPT} \
    --question-file ./playground/data/analysis/${FILE} \
    --image-folder ./playground/data/train \
    --output-folder ./playground/data/analysis/attn_dispersion \
    --visual-token-num 576 \
    --temperature 0 \
    --conv-mode vicuna_v1
