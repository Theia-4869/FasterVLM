#!/bin/bash

CKPT="llava-v1.6-vicuna-7b"
METHOD="fastervlm"
TOKEN=${1}
PARAM="n_${TOKEN}"

python -W ignore -m llava.eval.model_vqa_mmbench \
    --model-path /path/to/checkpoint/${CKPT} \
    --question-file ./playground/data/eval/mmbench_cn/mmbench_dev_cn_20231003.tsv \
    --answers-file ./playground/data/eval/mmbench_cn/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --lang cn \
    --single-pred-prompt \
    --visual-token-num ${TOKEN} \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench_cn/mmbench_dev_cn_20231003.tsv \
    --result-dir ./playground/data/eval/mmbench_cn/answers/${CKPT}/${METHOD} \
    --upload-dir ./playground/data/eval/mmbench_cn/answers_upload/${CKPT}/${METHOD} \
    --experiment ${PARAM}
