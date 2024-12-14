#!/bin/bash

CKPT="Video-LLaVA-7B"
METHOD="fastervlm"
TOKEN=${1}
PARAM="n_${TOKEN}"

GPT_Zero_Shot_QA="eval/GPT_Zero_Shot_QA"
output_name="outputs/${CKPT}/${METHOD}/${PARAM}"
pred_path="${GPT_Zero_Shot_QA}/MSVD_Zero_Shot_QA/${output_name}/merge.jsonl"
output_dir="${GPT_Zero_Shot_QA}/MSVD_Zero_Shot_QA/${output_name}/gpt-3.5-turbo"
output_json="${GPT_Zero_Shot_QA}/MSVD_Zero_Shot_QA/${output_name}/results.json"
api_key="sk-Jc2Ho5OAlNegdaCz9dBaCf3bEa954fB8AaF606A824D6D822"
api_key=""
api_base=""
num_tasks=8

python videollava/eval/video/eval_video_qa.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_key ${api_key} \
    --api_base ${api_base} \
    --num_tasks ${num_tasks}