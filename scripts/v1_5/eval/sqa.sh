#!/bin/bash

CKPT="llava-v1.5-7b"
METHOD="fastervlm"
TOKEN=${1}
PARAM="n_${TOKEN}"

python -W ignore -m llava.eval.model_vqa_science \
    --model-path /path/to/checkpoint/${CKPT} \
    --question-file ./playground/data/eval/sqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/sqa/images/test \
    --answers-file ./playground/data/eval/sqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --single-pred-prompt \
    --visual-token-num ${TOKEN} \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/sqa \
    --result-file ./playground/data/eval/sqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --output-file ./playground/data/eval/sqa/answers/${CKPT}/${METHOD}/${PARAM}_output.jsonl \
    --output-result ./playground/data/eval/sqa/answers/${CKPT}/${METHOD}/${PARAM}_result.json \
