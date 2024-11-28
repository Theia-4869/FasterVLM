# Evaluation

We evaluate FasterVLM with different [LLaVA](https://github.com/haotian-liu/LLaVA) models on a diverse set of 10 benchmarks. To ensure the reproducibility, we evaluate the models with greedy decoding following the originial LLaVA.

## Scripts

Before preparing task-specific data, **you MUST first download [eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing)**. It contains custom annotations, scripts, and the prediction files with vanilla LLaVA-1.5. Extract it to `./playground/data/eval`. This also provides a general structure for all datasets.

### VQAv2

1. Download [`test2015`](http://images.cocodataset.org/zips/test2015.zip) and put it under `./playground/data/eval/vqav2`.
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/vqav2.sh
```
3. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission): `./playground/data/eval/vqav2/answers_upload`.

### GQA

1. Download the [data](https://cs.stanford.edu/people/dorarad/gqa/download.html) and [evaluation scripts](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html) following the official instructions and put under `./playground/data/eval/gqa/data`. You may need to modify `eval.py` as [this](https://gist.github.com/haotian-liu/db6eddc2a984b4cbcc8a7f26fd523187) due to the missing assets in the GQA v1.2 release.
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/gqa.sh
```

### VisWiz

1. Download [`test.json`](https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip) and extract [`test.zip`](https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip) to `test`. Put them under `./playground/data/eval/vizwiz`.
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/vizwiz.sh
```
3. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/2185/my-submission): `./playground/data/eval/vizwiz/answers_upload`.

### ScienceQA

1. Under `./playground/data/eval/scienceqa`, download `images`, `pid_splits.json`, `problems.json` from the `data/scienceqa` folder of the ScienceQA [repo](https://github.com/lupantech/ScienceQA).
2. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/sqa.sh
```

### TextVQA

1. Download [`TextVQA_0.5.1_val.json`](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) and [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) and extract to `./playground/data/eval/textvqa`.
2. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/textvqa.sh
```

### POPE

1. Download `coco` from [POPE](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco) and put under `./playground/data/eval/pope`.
2. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/pope.sh
```

### MME

1. Download the data following the official instructions [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).
2. Downloaded images to `MME_Benchmark_release_version`.
3. put the official `eval_tool` and `MME_Benchmark_release_version` under `./playground/data/eval/MME`.
4. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme.sh
```

### MMBench

1. Download [`mmbench_dev_20230712.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv) and put under `./playground/data/eval/mmbench`.
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmbench.sh
```
3. Submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission): `./playground/data/eval/mmbench/answers_upload/mmbench_dev_20230712`.

### MMBench-CN

1. Download [`mmbench_dev_cn_20231003.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv) and put under `./playground/data/eval/mmbench`.
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmbench_cn.sh
```
3. Submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission): `./playground/data/eval/mmbench/answers_upload/mmbench_dev_cn_20231003`.

### MM-Vet

1. Extract [`mm-vet.zip`](https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip) to `./playground/data/eval/mmvet`.
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmvet.sh
```
3. Submit the results to the [evaluation server](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator): `./playground/data/eval/mm-vet/results`.

## Scripts with LLaVA-NeXT (LLaVA-1.6)

To evaluate FasterVLM with LLaVA-NeXT, you just need to replace the `v1_5` with `v1_6` in the shell scripts. For example, to evaluate VQAv2 with LLaVA-NeXT, you can run:

```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_6/eval/vqav2.sh
```

## Results

### LLaVA-1.5-7B

| Method       | Reduction Ratio | \# Token |  VQAv2 |   GQA  | VisWiz | SQA-IMG | TextVQA |  POPE  |    MME   |   MMB  | MMB-CN | MM-Vet | Average |
|--------------|:---------------:|:--------:|:------:|:------:|:------:|:-------:|:-------:|:------:|:--------:|:------:|:------:|:------:|:-------:|
| LLaVA-1.5-7B |        0%       |    576   | 78.52  | 61.94  | 50.06  |  69.51  |  58.21  | 85.87  | 1506.47  | 64.69  | 58.08  | 31.30  | 100.00% |
| FastV        |       25%       |    432   | 78.45  | 61.69  | 50.30  |  69.31  |  58.09  | 85.29  | 1528.16  | 64.52  | 58.76  | 32.80  | 100.60% |
| FitPrune     |       25%       |    432   | 78.49  | 61.93  | 50.07  |  69.51  |  58.27  | 85.94  | 1512.72  | 64.60  | 58.33  | 31.50  | 100.15% |
| SparseVLM    |       25%       |    432   | 78.13  | 61.38  | 50.38  |  68.62  |  57.44  | 84.56  | 1475.47  | 64.78  | 57.30  | 32.40  |  99.54% |
| FasterVLM    |       25%       |    432   | 78.39  | 61.51  | 50.01  |  68.57  |  57.91  | 85.92  | 1501.10  | 65.12  | 58.51  | 32.40  | 100.18% |
| FastV        |       50%       |    288   | 77.67  | 60.05  | 50.53  |  68.96  |  58.25  | 82.45  | 1513.06  | 64.26  | 58.16  | 31.70  |  99.33% |
| FitPrune     |       50%       |    288   | 78.41  | 61.70  | 50.04  |  69.16  |  58.26  | 85.37  | 1499.70  | 64.60  | 58.16  | 31.10  |  99.73% |
| SparseVLM    |       50%       |    288   | 76.67  | 58.78  | 51.03  |  68.57  |  57.49  | 83.87  | 1458.79  | 63.14  | 56.87  | 31.50  |  98.52% |
| FasterVLM    |       50%       |    288   | 77.86  | 60.64  | 50.45  |  68.37  |  57.90  | 86.20  | 1471.96  | 63.83  | 56.70  | 34.80  | 100.12% |
| FastV        |       75%       |    144   | 74.07  | 56.58  | 51.29  |  69.11  |  57.38  | 73.74  | 1463.39  | 64.00  | 57.22  | 28.60  |  95.80% |
| FitPrune     |       75%       |    144   | 76.14  | 59.38  | 51.30  |  69.01  |  56.49  | 80.75  | 1472.86  | 63.92  | 57.65  | 28.40  |  97.22% |
| SparseVLM    |       75%       |    144   | 72.76  | 55.11  | 51.46  |  69.36  |  55.99  | 77.57  | 1351.65  | 59.54  | 51.03  | 29.90  |  93.84% |
| FasterVLM    |       75%       |    144   | 76.19  | 58.34  | 51.97  |  67.92  |  57.07  | 83.46  | 1433.76  | 62.54  | 57.13  | 34.20  |  98.75% |
| FastV        |       90%       |    58    | 65.38  | 51.20  | 51.84  |  69.81  |  54.75  | 57.30  | 1210.36  | 59.97  | 51.72  | 27.20  |  87.97% |
| FitPrune     |       90%       |    58    | 62.76  | 49.96  | 50.85  |  68.22  |  50.35  | 53.81  | 1147.46  | 56.27  | 45.53  | 21.80  |  82.07% |
| SparseVLM    |       90%       |    58    | 62.90  | 48.86  | 49.36  |  67.23  |  48.99  | 65.82  | 1030.61  | 49.05  | 35.40  | 18.60  |  78.13% |
| FasterVLM    |       90%       |    58    | 71.92  | 54.91  | 53.01  |  68.91  |  55.28  | 75.85  | 1348.63  | 60.57  | 54.90  | 30.10  |  94.24% |
| FastV        |       95%       |    29    | 55.92  | 46.03  | 49.10  |  70.00  |  51.56  | 35.47  |  971.56  | 50.17  | 42.18  | 18.90  |  74.93% |
| FitPrune     |       95%       |    29    | 52.39  | 43.60  | 48.61  |  68.32  |  46.75  | 31.17  |  855.21  | 39.69  | 29.98  | 18.00  |  67.64% |
| FasterVLM    |       95%       |    29    | 66.75  | 51.51  | 52.67  |  69.56  |  53.09  | 67.24  | 1254.80  | 58.51  | 51.98  | 27.50  |  89.41% |
| LLaVA-1.5-7B |       100%      |     0    | 40.73  | 37.38  | 45.36  |  63.06  |  41.43  | 47.21  |  719.10  | 20.02  | 17.70  | 11.30  |  56.50% |

### LLaVA-1.5-13B

| Method        | Reduction Ratio | \# Token |  VQAv2 |   GQA  | VisWiz | SQA-IMG | TextVQA |  POPE  |    MME   |   MMB  | MMB-CN | MM-Vet | Average |
|---------------|:---------------:|:--------:|:------:|:------:|:------:|:-------:|:-------:|:------:|:--------:|:------:|:------:|:------:|:-------:|
| LLaVA-1.5-13B |        0%       |    576   | 80.00  | 63.25  | 53.61  |  72.78  |  61.19  | 85.99  | 1531.19  | 68.47  | 63.49  | 36.30  | 100.00% |
| FastV         |       25%       |    432   | 79.96  | 63.11  | 53.79  |  72.78  |  61.29  | 85.86  | 1543.97  | 68.38  | 63.40  | 35.20  |  99.76% |
| SparseVLM     |       25%       |    432   | 79.25  | 61.29  | 53.60  |  73.77  |  60.46  | 72.45  | 1481.40  | 68.30  | 61.77  | 35.50  |  97.20% |
| FasterVLM     |       25%       |    432   | 79.61  | 61.24  | 53.16  |  72.98  |  60.35  | 86.73  | 1487.34  | 67.61  | 63.40  | 36.70  |  99.21% |
| FastV         |       50%       |    288   | 79.54  | 62.59  | 54.34  |  73.13  |  60.86  | 85.15  | 1545.14  | 68.47  | 63.23  | 34.80  |  99.51% |
| SparseVLM     |       50%       |    288   | 78.48  | 59.90  | 53.08  |  74.02  |  59.48  | 71.30  | 1497.39  | 66.67  | 61.94  | 36.50  |  96.69% |
| FasterVLM     |       50%       |    288   | 79.03  | 61.01  | 52.65  |  73.62  |  59.99  | 86.05  | 1530.41  | 67.70  | 62.80  | 36.80  |  99.18% |
| FastV         |       75%       |    144   | 77.24  | 59.87  | 54.82  |  74.02  |  60.07  | 79.43  | 1493.51  | 67.27  | 62.63  | 33.20  |  97.16% |
| SparseVLM     |       75%       |    144   | 76.06  | 57.97  | 53.13  |  73.67  |  57.94  | 68.61  |  1499.49 | 64.52  | 59.11  | 35.00  |  94.32% |
| FasterVLM     |       75%       |    144   | 77.36  | 58.74  | 52.74  |  73.48  |  58.99  | 83.10  | 1467.00  | 67.10  | 62.54  | 36.30  |  97.43% |
| FastV         |       90%       |    58    | 70.27  | 54.92  | 54.78  |  72.43  |  55.64  | 67.26  | 1359.69  | 63.83  | 59.71  | 29.40  |  90.26% |
| SparseVLM     |       90%       |    58    | 68.27  | 54.43  | 50.45  |  70.35  |  52.56  | 62.63  | 1285.26  | 58.16  | 54.30  | 27.20  |  85.02% |
| FasterVLM     |       90%       |    58    | 73.08  | 55.98  | 54.00  |  73.72  |  57.35  | 74.71  | 1370.77  | 65.21  | 61.08  | 33.90  |  93.68% |
| FastV         |       95%       |    29    | 62.25  | 50.34  | 52.96  |  73.18  |  52.08  | 49.83  | 1165.70  | 56.44  | 51.29  | 24.00  |  80.53% |
| FasterVLM     |       95%       |    29    | 67.85  | 52.62  | 53.11  |  72.83  |  54.82  | 65.90  | 1267.09  | 62.11  | 56.87  | 31.60  |  88.35% |
| LLaVA-1.5-13B |       100%      |     0    | 41.40  | 38.40  | 45.15  |  66.63  |  43.51  |  1.45  |  622.91  | 22.77  | 18.13  | 13.20  |  49.99% |


### LLaVA-NeXT-7B

| Method        | Reduction Ratio | \# Token |  VQAv2 |   GQA  | VisWiz | SQA-IMG | TextVQA |  POPE  |    MME   |   MMB  | MMB-CN | MM-Vet | Average |
|---------------|:---------------:|:--------:|:------:|:------:|:------:|:-------:|:-------:|:------:|:--------:|:------:|:------:|:------:|:-------:|
| LLaVA-NeXT-7B |        0%       |   2880   | 81.21  | 62.93  | 55.21  |  69.66  |  59.59  | 86.32  | 1513.78  | 67.70  | 58.85  | 42.60  | 100.00% |
| FastV         |       25%       |   2160   | 81.12  | 62.50  | 55.05  |  69.31  |  59.69  | 86.27  | 1506.28  | 67.61  | 59.02  | 41.70  |  99.61% |
| SparseVLM     |       25%       |   2160   | 81.14  | 62.55  | 55.20  |  68.47  |  60.26  | 73.15  | 1507.75  | 66.07  | 58.59  | 41.90  |  97.86% |
| FasterVLM     |       25%       |   2160   | 81.18  | 62.81  | 56.12  |  70.70  |  59.65  | 86.26  | 1492.20  | 67.35  | 58.33  | 44.30  | 100.41% |
| FastV         |       50%       |   1440   | 80.71  | 61.76  | 54.89  |  69.06  |  59.55  | 85.46  | 1490.34  | 67.35  | 58.51  | 41.20  |  98.91% |
| SparseVLM     |       50%       |   1440   | 80.92  | 62.04  | 55.71  |  68.07  |  60.00  | 73.42  | 1484.92  | 65.72  | 58.85  | 39.90  |  97.14% |
| FasterVLM     |       50%       |   1440   | 80.72  | 62.66  | 55.82  |  69.41  |  59.72  | 86.66  | 1521.87  | 67.87  | 58.85  | 44.40  | 100.53% |
| FastV         |       75%       |    720   | 78.90  | 60.38  | 54.22  |  69.81  |  58.39  | 83.09  | 1477.31  | 65.64  | 57.04  | 41.10  |  97.37% |
| SparseVLM     |       75%       |    720   | 78.86  | 60.88  | 55.55  |  67.48  |  58.08  | 70.99  | 1446.10  | 63.83  | 57.04  | 38.00  |  94.95% |
| FasterVLM     |       75%       |    720   | 79.25  | 61.31  | 56.24  |  68.82  |  59.33  | 85.50  | 1480.68  | 67.53  | 59.19  | 40.40  |  98.73% |
| FastV         |       90%       |    290   | 71.94  | 55.86  | 53.07  |  69.26  |  55.69  | 71.66  | 1282.86  | 61.60  | 51.89  | 33.70  |  89.24% |
| SparseVLM     |       90%       |    290   | 71.62  | 56.12  | 53.16  |  68.62  |  51.97  | 63.23  | 1332.22  | 54.47  | 50.69  | 24.70  |  84.52% |
| FasterVLM     |       90%       |    290   | 75.21  | 58.12  | 56.92  |  68.12  |  57.57  | 80.00  | 1370.11  | 63.32  | 54.47  | 35.70  |  93.55% |
| FastV         |       95%       |    145   | 61.84  | 49.83  | 51.25  |  68.52  |  51.85  | 51.66  | 1079.46  | 54.90  | 45.36  | 21.90  |  77.43% |
| FasterVLM     |       95%       |    145   | 70.63  | 54.73  | 56.27  |  68.86  |  55.97  | 72.89  | 1225.96  | 60.48  | 53.09  | 31.90  |  88.85% |
| LLaVA-NeXT-7B |       100%      |     0    | 40.59  | 37.93  | 46.01  |  64.01  |  37.57  | 23.40  |  601.93  | 21.05  | 17.70  | 13.10  |  50.73% |
