# ⚡️ [CLS] Attention is All You Need for Training-Free Visual Token Pruning: Make VLM Inference Faster (Implementation with Video-LLaVA)

*A simple yet effective training-free token pruning method that evaluates the importance of visual tokens more accurately by [CLS] attentions, making VLM inference faster.*

[📄 [Paper](https://arxiv.org/abs/2412.01818)] [🎞️ [Project Page](https://theia4869.com/FasterVLM/)]

## 📹 FasterVLM with Video-LLaVA

In this repository, we apply our **FasterVLM** to **Video-LLaVA**, which accepts **videos** as input, and conduct experiments on **four** video question answering benchmarks, using ChatGPT assistant for evaluation.

## ⚙️ Setup

### 🏝️ Environment

1. Clone this repository.
```bash
git clone https://github.com/Theia-4869/FasterVLM.git
cd FasterVLM
```

2. Install necessary packages.
```bash
conda create -n fastervlm_video python=3.10 -y
conda activate fastervlm_video
pip install -e .
pip install decord opencv-python pytorchvideo
```

3. (Optional) Install FlashAttention for further inference acceleration.
```bash
pip install flash-attn --no-build-isolation
```

### 📦️ Model

1. Download LanguageBind checkpoints [LanguageBind_Image](https://huggingface.co/LanguageBind/LanguageBind_Image) and [LanguageBind_Video_merge](https://huggingface.co/LanguageBind/LanguageBind_Video_merge) from [Hugging Face](https://huggingface.co/LanguageBind) 🤗.

2. Download [Video-LLaVA](https://huggingface.co/LanguageBind/Video-LLaVA-7B) checkpoint and set the corresponding `mm_image_tower` and `mm_video_tower` paths in `config.json` to the downloaded checkpoints in step 1.

### 📊 Data

1. Download each dataset according to [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/TRAIN_AND_VALIDATE.md#data-for-validating):

| Dataset | Baidu Disk | Google Disk | Peking University Disk |
|----------|:----------:|:-----------:|:-----------:|
| TGIF-QA        | [Link](https://pan.baidu.com/s/11ubtWbTtubyBmN9UPvAyow?pwd=98yr) | [Link](https://drive.google.com/file/d/1so6L9rg_gdC8Segur7rKML-ffd4Ix_I6/view?usp=drive_link) | [Link](https://disk.pku.edu.cn/link/B9AB387EFE8817158F181FF3D7A97163) |
| MSVD-QA        | [Link](https://pan.baidu.com/s/1PJSHkjHG2BPl_ddUnBj9AA?pwd=jj34) | [Link](https://drive.google.com/file/d/1_q4eiSdb7i8P3Hmh4lCfgY1uBGyzU_7X/view?usp=drive_link) | [Link](https://disk.pku.edu.cn/link/8B0D01747D8AA65534820B7E60CBFEFC) |
| MSRVTT-QA      | [Link](https://pan.baidu.com/s/1QHUtwHXm4Vc-Wc12XFCFsA?pwd=1rj8) | [Link](https://drive.google.com/file/d/1yXh9lz7flQ5Ui2IRSd6Qi6RqSEeUJwl3/view?usp=drive_link) | - |
| ActivityNet-QA | [Link](https://pan.baidu.com/s/1d_AVx9Mz_57nA3exhQZGyA?pwd=9amr) | - | - |

2. Organize the data as follows in `./eval`:

```Shell
eval
└── GPT_Zero_Shot_QA
    ├── Activitynet_Zero_Shot_QA
    ├── MSRVTT_Zero_Shot_QA
    ├── MSVD_Zero_Shot_QA
    └── TGIF_Zero_Shot_QA
```

## 📋️ Evaluation

The main implementation of FasterVLM is highlighted with `FasterVLM` annotations, mainly in [`llava_llama.py`](videollava/model/language_model/llava_llama.py#L50), [`llava_arch.py`](videollava/model/llava_arch.py#L140), [`languagebind/__init__.py`](videollava/model/multimodal_encoder/languagebind/__init__.py#L204) and [`modeling_video.py`](videollava/model/multimodal_encoder/languagebind/video/modeling_video.py#L666).

We provide the evaluation scripts for each benchmark, you only need to set the remaining visual token number as the bash argument. For example, if you want to evaluate FasterVLM under 75% reduction ratio (256 * (1 - 0.75) = 64) on the TGIF-QA benchmark with 1 GPU, you can run the script `./scripts/eval/run_qa_tgif.sh` with argument `64` in the following command:
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/run_qa_tgif.sh 64
bash scripts/eval/eval_qa_tgif.sh 64
```

And if you want to evaluate FasterVLM under 90% reduction ratio (256 * (1 - 0.9) = 26) on the MSVD-QA benchmark with 8 GPUs, you can run the following command:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/run_qa_msvd.sh 26
bash scripts/eval/eval_qa_msvd.sh 26
```

 Due to the commercial API usage limits, we use the first 1K samples from each benchmark in our experiments and provide the corresponding samples in the `eval` directory as `test_q_1k.json` and `test_a_1k.json`. For the full evaluation, you just need to change the `gt_file_question` and `gt_file_answers` paths in each script to the full dataset.

<table>
<thead align="center">
  <tr>
    <th rowspan="2">Method</th>
    <th rowspan="2">Reduction Ratio</th>
    <th rowspan="2">\# Token</th>
    <th colspan="2">TGIF-QA</th>
    <th colspan="2">MSVD-QA</th>
    <th colspan="2">MSRVTT-QA</th>
    <th colspan="2">ActivityNet-QA</th>
    <th colspan="2">Average</th>
  </tr>
  <tr>
    <th>Acc.</th>
    <th>Score</th>
    <th>Acc.</th>
    <th>Score</th>
    <th>Acc.</th>
    <th>Score</th>
    <th>Acc.</th>
    <th>Score</th>
    <th>Acc.</th>
    <th>Score</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td>Video-LLaVA</td>
    <td>0%</td>
    <td>2048</td>
    <td>19.80 </td>
    <td>2.53 </td>
    <td>70.50 </td>
    <td>3.93 </td>
    <td>57.50 </td>
    <td>3.50 </td>
    <td>43.60 </td>
    <td>3.81 </td>
    <td>100.00%</td>
    <td>100.00%</td>
  </tr>
  <tr>
    <td>FastV</td>
    <td rowspan="2">25%</td>
    <td rowspan="2">1536</td>
    <td>19.60 </td>
    <td>2.54 </td>
    <td>70.50 </td>
    <td>3.94 </td>
    <td>55.40 </td>
    <td>3.47 </td>
    <td>44.20 </td>
    <td>3.86 </td>
    <td>99.18%</td>
    <td>100.24%</td>
  </tr>
  <tr>
    <td><b>FasterVLM</b></td>
    <td>19.80 </td>
    <td>2.53 </td>
    <td>72.00 </td>
    <td>3.96 </td>
    <td>57.10 </td>
    <td>3.48 </td>
    <td>45.00 </td>
    <td>3.88 </td>
    <td><b>101.16%</b></td>
    <td><b>100.45%</b></td>
  </tr>
  <tr>
    <td>FastV</td>
    <td rowspan="2">50%</td>
    <td rowspan="2">1024</td>
    <td>19.50 </td>
    <td>2.54 </td>
    <td>70.40 </td>
    <td>3.94 </td>
    <td>54.80 </td>
    <td>3.46 </td>
    <td>43.40 </td>
    <td>3.81 </td>
    <td>98.30%</td>
    <td>99.79%</td>
  </tr>
  <tr>
    <td><b>FasterVLM</b></td>
    <td>19.20 </td>
    <td>2.51 </td>
    <td>71.90 </td>
    <td>3.96 </td>
    <td>56.70 </td>
    <td>3.47 </td>
    <td>44.90 </td>
    <td>3.87 </td>
    <td><b>100.14%</b></td>
    <td><b>100.11%</b></td>
  </tr>
  <tr>
    <td>FastV</td>
    <td rowspan="2">75%</td>
    <td rowspan="2">512</td>
    <td>19.40 </td>
    <td>2.51 </td>
    <td>69.10 </td>
    <td>3.91 </td>
    <td>54.60 </td>
    <td>3.43 </td>
    <td>41.50 </td>
    <td>3.80 </td>
    <td>96.53%</td>
    <td>99.12%</td>
  </tr>
  <tr>
    <td><b>FasterVLM</b></td>
    <td>17.90 </td>
    <td>2.48 </td>
    <td>70.20 </td>
    <td>3.95 </td>
    <td>56.60 </td>
    <td>3.50 </td>
    <td>44.70 </td>
    <td>3.86 </td>
    <td><b>97.73%</b></td>
    <td><b>99.88%</b></td>
  </tr>
  <tr>
    <td>FastV</td>
    <td rowspan="2">90%</td>
    <td rowspan="2">208</td>
    <td>13.80 </td>
    <td>2.40 </td>
    <td>68.80 </td>
    <td>3.90 </td>
    <td>52.90 </td>
    <td>3.40 </td>
    <td>41.30 </td>
    <td>3.79 </td>
    <td>88.50%</td>
    <td>97.63%</td>
  </tr>
  <tr>
    <td><b>FasterVLM</b></td>
    <td>15.60 </td>
    <td>2.39 </td>
    <td>69.10 </td>
    <td>3.92 </td>
    <td>55.30 </td>
    <td>3.43 </td>
    <td>44.50 </td>
    <td>3.82 </td>
    <td><b>93.76%</b></td>
    <td><b>98.04%</b></td>
  </tr>
  <tr>
    <td>FastV</td>
    <td rowspan="2">95%</td>
    <td rowspan="2">104</td>
    <td>10.60 </td>
    <td>2.29 </td>
    <td>64.10 </td>
    <td>3.78 </td>
    <td>52.40 </td>
    <td>3.39 </td>
    <td>40.30 </td>
    <td>3.78 </td>
    <td>82.00%</td>
    <td>95.64%</td>
  </tr>
  <tr>
    <td><b>FasterVLM</b></td>
    <td>14.00 </td>
    <td>2.34 </td>
    <td>65.30 </td>
    <td>3.79 </td>
    <td>53.80 </td>
    <td>3.40 </td>
    <td>43.70 </td>
    <td>3.79 </td>
    <td><b>89.28%</b></td>
    <td><b>96.37%</b></td>
  </tr>
  <tr>
    <td>Video-LLaVA</td>
    <td>100%</td>
    <td>0</td>
    <td>5.70 </td>
    <td>1.96 </td>
    <td>23.40 </td>
    <td>1.77 </td>
    <td>24.50 </td>
    <td>1.87 </td>
    <td>39.10 </td>
    <td>3.66 </td>
    <td>48.57%</td>
    <td>68.01%</td>
  </tr>
</tbody>
</table>

## 🎗️ Citation

If you find FasterVLM useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{zhang2024fastervlm,
      title={[CLS] Attention is All You Need for Training-Free Visual Token Pruning: Make VLM Inference Faster}, 
      author={Zhang, Qizhe and Cheng, Aosong and Lu, Ming and Zhuo, Zhiyong and Wang, MinQi and Cao, Jiajun and Guo, Shaobo and She, Qi and Zhang, Shanghang},
      journal={arXiv preprint arXiv:2412.01818},
      year={2024},
}
```

## 🎟️ License

This project is released under the [Apache 2.0 license](LICENSE).

## 🎉 Acknowledgement

We appreciate the open-source efforts of [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA).