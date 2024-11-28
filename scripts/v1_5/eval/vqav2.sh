#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b"
METHOD="fastervlm"
TOKEN=${1}
PARAM="n_${TOKEN}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -W ignore -m llava.eval.model_vqa_loader \
        --model-path /path/to/checkpoint/${CKPT} \
        --question-file ./playground/data/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl \
        --image-folder ./playground/data/eval/vqav2/test2015 \
        --answers-file ./playground/data/eval/vqav2/answers/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --visual-token-num ${TOKEN} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

VQAV2DIR="./playground/data/eval/vqav2"
output_file=${VQAV2DIR}/answers/${CKPT}/${METHOD}/${PARAM}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py \
    --dir ${VQAV2DIR} \
    --src answers/${CKPT}/${METHOD}/${PARAM}/merge.jsonl \
    --dst answers_upload/${CKPT}/${METHOD}/${PARAM}/upload.json

