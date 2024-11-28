import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        question = line["conversations"][0]
        qs = question["value"].replace("<image>\n", "").replace("<image>", "")
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        _, image_tensor = process_images([image], self.image_processor, self.model_config)

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor[0], image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, visual_token_num=args.visual_token_num)

    # Data
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    os.makedirs(args.output_folder, exist_ok=True)

    cls_attn_dist = torch.zeros(576)
    last_attn_dist = torch.zeros(576)
    cls_attns, last_attns = [], []
    num = 0
    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids, v_token_num, cls_attn = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_attentions=True,
                return_dict_in_generate=True
            )
        
        cls_attn = cls_attn.cpu() # (B, 576)
        last_attn = output_ids['attentions'][0][1].mean(dim=1)[:, -1, 35:611].cpu() # (B, 576)

        cls_attn_dist += (torch.sort(cls_attn, dim=1, descending=True)[0] / cls_attn.sum(dim=1)).sum(dim=0)
        last_attn_dist += (torch.sort(last_attn, dim=1, descending=True)[0] / last_attn.sum(dim=1)).sum(dim=0)

        cls_attns.extend(cls_attn.flatten().numpy().tolist())
        last_attns.extend(last_attn.flatten().numpy().tolist())

    cls_attn_cum = cls_attn_dist.cumsum(dim=0) / len(questions)
    last_attn_cum = last_attn_dist.cumsum(dim=0) / len(questions)

    plt.figure(figsize=(6, 6))
    plt.grid()
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.85, top=0.85)
    plt.plot(np.arange(len(cls_attn_cum)) / len(cls_attn_cum) * 100, cls_attn_cum * 100, label="[CLS] attn")
    plt.plot(np.arange(len(cls_attn_cum)) / len(cls_attn_cum) * 100, last_attn_cum * 100, label="last attn")

    plt.title("Cumulative Distribution", fontsize=20)
    plt.xlabel("Token Proportion (%)", fontsize=16)
    plt.ylabel("Cumulative Attention (%)", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=16)

    plt.savefig(os.path.join(args.output_folder, "CDF.png"))

    cls_attns = np.array(cls_attns)
    cls_attns = cls_attns / cls_attns.max()
    counts, bin_edges = np.histogram(cls_attns, bins=200)
    proportions = counts / cls_attns.shape[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    proportions = 5 + np.log(proportions + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.85, top=0.85)
    plt.bar(bin_centers, proportions, width=bin_centers[1] - bin_centers[0])

    plt.xlim(-0.01, 0.16)
    plt.ylim(0, 4.9)

    plt.title("[CLS] Attention Distribution", fontsize=20)
    plt.xlabel("Attention Score (normalized)", fontsize=16)
    plt.ylabel("Proportion", fontsize=16)
    plt.xticks(ticks=[0, 0.05, 0.1, 0.15], labels=["0.00", "0.05", "0.10", "0.15"], fontsize=12)
    plt.yticks(ticks=[0, 1, 2, 3, 4], labels=["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"], fontsize=12)

    plt.savefig(os.path.join(args.output_folder, "cls_attn_dist.png"))
    plt.close()

    last_attns = np.array(last_attns)
    last_attns = last_attns / last_attns.max()
    counts, bin_edges = np.histogram(last_attns, bins=200)
    proportions = counts / last_attns.shape[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    proportions = 5 + np.log(proportions + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.85, top=0.85)
    plt.bar(bin_centers, proportions, width=bin_centers[1] - bin_centers[0])

    plt.xlim(-0.01, 0.16)
    plt.ylim(0, 4.9)

    plt.title("Last Attention Distribution", fontsize=20)
    plt.xlabel("Attention Score (normalized)", fontsize=16)
    plt.ylabel("Proportion", fontsize=16)
    plt.xticks(fontsize=12)
    plt.xticks(ticks=[0, 0.05, 0.1, 0.15], labels=["0.00", "0.05", "0.10", "0.15"], fontsize=12)
    plt.yticks(ticks=[0, 1, 2, 3, 4], labels=["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"], fontsize=12)

    plt.savefig(os.path.join(args.output_folder, "last_attn_dist.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--output-folder", type=str, default="")
    parser.add_argument("--param", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--visual-token-num", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
