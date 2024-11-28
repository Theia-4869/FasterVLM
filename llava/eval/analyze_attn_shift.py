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

    cls_attentions = torch.zeros(576, device='cuda')
    full_attentions = [torch.zeros(576, device='cuda') for _ in range(32)]
    image_attentions = [torch.zeros(576, device='cuda') for _ in range(32)]
    text_attentions = [torch.zeros(576, device='cuda') for _ in range(32)]
    last_attentions = [torch.zeros(576, device='cuda') for _ in range(32)]

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids, v_token_num, cls_attns = model.generate(
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

        all_attns = [attns[0, :, 35:, 35:611].mean(dim=0) for attns in output_ids['attentions'][0]] # (N, 576) * 32
        full_attns = [all_attn.sum(dim=0) / (torch.arange(576, 0, -1, device='cuda') + all_attn.shape[0] - 576) for all_attn in all_attns] # (576,) * 32
        image_attns = [all_attn[:576, :].sum(dim=0) / torch.arange(576, 0, -1, device='cuda') for all_attn in all_attns] # (576,) * 32
        text_attns = [all_attn[576:-1, :].sum(dim=0) / (all_attn.shape[0] - 576) for all_attn in all_attns] # (576,) * 32
        # full_attns = [all_attn.mean(dim=0) for all_attn in all_attns] # (576,) * 32
        # image_attns = [all_attn[:576, :].mean(dim=0) for all_attn in all_attns] # (576,) * 32
        # text_attns = [all_attn[576:-1, :].mean(dim=0) for all_attn in all_attns] # (576,) * 32
        last_attns = [all_attn[-1, :] for all_attn in all_attns] # (576,) * 32

        cls_attentions += cls_attns[0]
        full_attentions = [full_attns[i] + full_attentions[i] for i in range(32)]
        image_attentions = [image_attns[i] + image_attentions[i] for i in range(32)]
        text_attentions = [text_attns[i] + text_attentions[i] for i in range(32)]
        last_attentions = [last_attns[i] + last_attentions[i] for i in range(32)]

    cls_attentions = cls_attentions.cpu() / len(questions)
    full_attentions = [attn.cpu() / len(questions) for attn in full_attentions]
    image_attentions = [attn.cpu() / len(questions) for attn in image_attentions]
    text_attentions = [attn.cpu() / len(questions) for attn in text_attentions]
    last_attentions = [attn.cpu() / len(questions) for attn in last_attentions]

    # Save attentions
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Plot attentions
    cls_attn_dir = os.path.join(args.output_folder, "cls_attn")
    os.makedirs(cls_attn_dir, exist_ok=True)

    plt.figure(figsize=(6, 6), dpi=96)
    plt.bar(range(1, 577), cls_attentions.numpy())
    plt.ylim(0, 0.005)

    plt.title("[CLS] Attention Distribution", fontsize=16)
    plt.xlabel("Token Index", fontsize=14)
    plt.ylabel("Attention Score", fontsize=14)

    plt.savefig(os.path.join(cls_attn_dir, "dist.png"))
    plt.close()

    plt.figure(figsize=(6, 6), dpi=96)
    plt.imshow(cls_attentions.numpy().reshape(24, 24), cmap='viridis')
    plt.axis('off')

    plt.title('[CLS] Attention Map', fontsize=16)
    plt.savefig(os.path.join(cls_attn_dir, "map.png"))
    plt.close()
    
    full_attn_dist_dir = os.path.join(args.output_folder, "full_attn", "dist")
    os.makedirs(full_attn_dist_dir, exist_ok=True)
    full_attn_map_dir = os.path.join(args.output_folder, "full_attn", "map")
    os.makedirs(full_attn_map_dir, exist_ok=True)
    for i in range(32):
        # attention distribution
        plt.figure(figsize=(6, 6), dpi=96)
        plt.bar(range(1, 577), full_attentions[i].numpy())

        plt.title(f"Full Attention Distribution Layer {i+1}", fontsize=16)
        plt.xlabel("Token Index", fontsize=14)
        plt.ylabel("Attention Score", fontsize=14)

        plt.savefig(os.path.join(full_attn_dist_dir, f"layer_{i}.png"))
        plt.close()

        # attention heatmap
        plt.figure(figsize=(6, 6), dpi=96)
        plt.imshow(full_attentions[i].numpy().reshape(24, 24), cmap='viridis')
        plt.axis('off')

        plt.title('Full Attention Map Layer {}'.format(i+1), fontsize=16)
        plt.savefig(os.path.join(full_attn_map_dir, f"layer_{i}.png"))
        plt.close()
    
    image_attn_dist_dir = os.path.join(args.output_folder, "image_attn", "dist")
    os.makedirs(image_attn_dist_dir, exist_ok=True)
    image_attn_map_dir = os.path.join(args.output_folder, "image_attn", "map")
    os.makedirs(image_attn_map_dir, exist_ok=True)
    for i in range(32):
        # attention distribution
        plt.figure(figsize=(6, 6), dpi=96)
        plt.bar(range(1, 577), image_attentions[i].numpy())

        plt.title(f"Image Attention Distribution Layer {i+1}", fontsize=16)
        plt.xlabel("Token Index", fontsize=14)
        plt.ylabel("Attention Score", fontsize=14)

        plt.savefig(os.path.join(image_attn_dist_dir, f"layer_{i}.png"))
        plt.close()

        # attention heatmap
        plt.figure(figsize=(6, 6), dpi=96)
        plt.imshow(image_attentions[i].numpy().reshape(24, 24), cmap='viridis')
        plt.axis('off')

        plt.title('Image Attention Map Layer {}'.format(i+1), fontsize=16)
        plt.savefig(os.path.join(image_attn_map_dir, f"layer_{i}.png"))
        plt.close()
    
    text_attn_dist_dir = os.path.join(args.output_folder, "text_attn", "dist")
    os.makedirs(text_attn_dist_dir, exist_ok=True)
    text_attn_map_dir = os.path.join(args.output_folder, "text_attn", "map")
    os.makedirs(text_attn_map_dir, exist_ok=True)
    for i in range(32):
        # attention distribution
        plt.figure(figsize=(6, 6), dpi=96)
        plt.bar(range(1, 577), text_attentions[i].numpy())

        plt.title(f"Text Attention Distribution Layer {i+1}", fontsize=16)
        plt.xlabel("Token Index", fontsize=14)
        plt.ylabel("Attention Score", fontsize=14)

        plt.savefig(os.path.join(text_attn_dist_dir, f"layer_{i}.png"))
        plt.close()

        # attention heatmap
        plt.figure(figsize=(6, 6), dpi=96)
        plt.imshow(text_attentions[i].numpy().reshape(24, 24), cmap='viridis')
        plt.axis('off')

        plt.title('Text Attention Map Layer {}'.format(i+1), fontsize=16)
        plt.savefig(os.path.join(text_attn_map_dir, f"layer_{i}.png"))
        plt.close()

    last_attn_dist_dir = os.path.join(args.output_folder, "last_attn", "dist")
    os.makedirs(last_attn_dist_dir, exist_ok=True)
    last_attn_map_dir = os.path.join(args.output_folder, "last_attn", "map")
    os.makedirs(last_attn_map_dir, exist_ok=True)
    for i in range(32):
        # attention distribution
        plt.figure(figsize=(6, 6), dpi=96)
        plt.bar(range(1, 577), last_attentions[i].numpy())

        plt.title(f"Last Attention Distribution Layer {i+1}", fontsize=16)
        plt.xlabel("Token Index", fontsize=14)
        plt.ylabel("Attention Score", fontsize=14)

        plt.savefig(os.path.join(last_attn_dist_dir, f"layer_{i}.png"))
        plt.close()

        # attention heatmap
        plt.figure(figsize=(6, 6), dpi=96)
        plt.imshow(last_attentions[i].numpy().reshape(24, 24), cmap='viridis')
        plt.axis('off')

        plt.title('Last Attention Map Layer {}'.format(i+1), fontsize=16)
        plt.savefig(os.path.join(last_attn_map_dir, f"layer_{i}.png"))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--output-folder", type=str, default="")
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
