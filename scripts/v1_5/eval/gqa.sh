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
        --question-file ./playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl \
        --image-folder ./playground/data/eval/gqa/data/images \
        --answers-file ./playground/data/eval/gqa/answers/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --visual-token-num ${TOKEN} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

GQADIR="./playground/data/eval/gqa/data"
output_file=./playground/data/eval/gqa/answers/${CKPT}/${METHOD}/${PARAM}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src "$output_file" --dst ${GQADIR}/${CKPT}/${METHOD}/${PARAM}/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced --method ${CKPT}/${METHOD}/${PARAM}
