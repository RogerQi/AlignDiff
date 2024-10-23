from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline
import numpy as np
import cv2
import argparse
import multiprocessing as mp
import os
import argparse

from aligndiff.utils import parse_args, read_json_file
from aligndiff.ptp_utils import text2image_ldm_stable, AttentionStore, aggregate_attention

GUIDANCE_SCALE = 7.5
                        
def save_cross_attention(attention_store: AttentionStore, token_pos_list, mask_output_path: str, prompts=None):
    batch_idx = 0
    token_pos_start, token_pos_end = token_pos_list[batch_idx]
    
    from_where = ("up", "down")

    attention_maps_8s = aggregate_attention(attention_store, 8, ("up", "mid", "down"), True, batch_idx, prompts=prompts)
    attention_maps_8s = attention_maps_8s.sum(0) / attention_maps_8s.shape[0]
    
    attention_maps = aggregate_attention(attention_store, 16, from_where, True, batch_idx, prompts=prompts)
    attention_maps = attention_maps.sum(0) / attention_maps.shape[0]
    
    attention_maps_32 = aggregate_attention(attention_store, 32, from_where, True, batch_idx, prompts=prompts)
    attention_maps_32 = attention_maps_32.sum(0) / attention_maps_32.shape[0]
    
    attention_maps_64 = aggregate_attention(attention_store, 64, from_where, True, batch_idx, prompts=prompts)
    attention_maps_64 = attention_maps_64.sum(0) / attention_maps_64.shape[0]

    gt_kernel_final = np.zeros((512,512), dtype='float32')
    number_gt = 0

    for i in range(token_pos_start, token_pos_end):
        
        image_8 = attention_maps_8s[:, :, i].to(torch.float32)
        image_8 = cv2.resize(image_8.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
        image_8 = image_8 / image_8.max()
        
        image_16 = attention_maps[:, :, i].to(torch.float32)
        image_16 = cv2.resize(image_16.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
        image_16 = image_16 / image_16.max()
        
        image_32 = attention_maps_32[:, :, i].to(torch.float32)
        image_32 = cv2.resize(image_32.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
        image_32 = image_32 / image_32.max()
        
        image_64 = attention_maps_64[:, :, i].to(torch.float32)
        image_64 = cv2.resize(image_64.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
        image_64 = image_64 / image_64.max()
        
        # Roger(2024.10): DiffuMask uses different attentions for different prompts. We use uniform attention for all prompts.
        # if class_one == "sofa" or class_one == "train" or class_one == "tvmonitor":
        #     image = image_8
        # elif class_one == "diningtable":
        #     image = image_16
        image = (image_16 + image_32 + image_64) / 3
        
        gt_kernel_final += image.copy()
        number_gt += 1

    if number_gt != 0:
        gt_kernel_final = gt_kernel_final / number_gt

    # np.save(output, gt_kernel_final)
    mask = (gt_kernel_final * 255).astype(np.uint8)
    cv2.imwrite(mask_output_path, mask)

def generate(diffusion_pipeline, output_dir, text_prompt, category_name, img_cnt, generation_mode):
    tokenizer = diffusion_pipeline.tokenizer
    
    img_dir = os.path.join(output_dir, "generated_images")
    mask_dir = os.path.join(output_dir, "attention_masks")
    
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    
    for img_idx in range(img_cnt):
        g_cpu = torch.Generator().manual_seed(img_idx)

        if generation_mode == 'text':
            assert category_name in text_prompt
            prompts = [text_prompt]
            print(prompts)

            # Get idx of token representing category of interest
            token_pos_list = []
            for idx in range(len(prompts)):
                obj_tokens = tokenizer.encode(category_name)[1:-1]  # Remove start and end tokens
                tokens = tokenizer.encode(prompts[idx])
                token_idx = None
                for i in range(len(tokens) - len(obj_tokens) + 1):
                    if tokens[i:i + len(obj_tokens)] == obj_tokens:
                        token_idx = i
                        break
                assert token_idx is not None, 'Prompt does not contain object token'
                token_pos_list.append([token_idx, token_idx + len(obj_tokens)])
        elif generation_mode == 'NM_embedding':
            assert category_name in text_prompt
            prompts = [text_prompt]
            print(prompts)

            # Get idx of token representing category of interest
            token_pos_list = []
            for idx in range(len(prompts)):
                obj_tokens = tokenizer.encode(category_name)[1:-1]  # Remove start and end tokens
                tokens = tokenizer.encode(prompts[idx])
                token_idx = None
                for i in range(len(tokens) - len(obj_tokens) + 1):
                    if tokens[i:i + len(obj_tokens)] == obj_tokens:
                        token_idx = i
                        break
                assert token_idx is not None, 'Prompt does not contain object token'
                token_pos_list.append([token_idx, token_idx + len(obj_tokens)])
        else:
            raise ValueError('Invalid generation mode: {}'.format(generation_mode))
        
        controller = AttentionStore()

        images = text2image_ldm_stable(diffusion_pipeline,
                                        prompts,
                                        controller,
                                        guidance_scale=GUIDANCE_SCALE,
                                        generator=g_cpu)

        image_output_path = os.path.join(img_dir, "image_{}.jpg".format(str(img_idx).zfill(6)))
        images[0].save(image_output_path)

        save_cross_attention(controller,
                             token_pos_list,
                             mask_output_path=os.path.join(mask_dir, "image_{}.png".format(str(img_idx).zfill(6))),
                             prompts=prompts)

def main(pid, args, dataset):
    torch.cuda.set_device(pid)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    diffusion_pipeline = StableDiffusionPipeline.from_pretrained(args.diffusion_model, torch_dtype=torch.float16).to(device)

    for prompt_idx in range(len(dataset)):
        output_dir = os.path.join(args.dataset_dir, 'synthesized', 'aligndiff_{}'.format(str(prompt_idx).zfill(6)))
        os.makedirs(output_dir, exist_ok=True)
        if dataset[prompt_idx]['type'] == 'text':
            prompt = dataset[prompt_idx]['text']
            attention_str = dataset[prompt_idx]['category']
            generate(diffusion_pipeline,
                     output_dir,
                     prompt,
                     dataset[prompt_idx]['category'],
                     args.image_per_class, 'text')
            
        elif dataset[prompt_idx]['type'] == 'NM_embedding':
            prompt = "a photo of a <fss-token> {}".format(dataset[prompt_idx]['category'])
            learned_embedding_path = os.path.join(args.dataset_dir, dataset[prompt_idx]['path'], "learned_embeddings")
            diffusion_pipeline.load_textual_inversion(learned_embedding_path)
            attention_str = dataset[prompt_idx]['category']
            generate(diffusion_pipeline,
                        output_dir,
                        prompt,
                        attention_str,
                        args.image_per_class, 'NM_embedding')
        else:
            raise ValueError('Invalid dataset type: {}'.format(dataset[prompt_idx]['type']))
        
        # Reset diffusion
        diffusion_pipeline.unload_textual_inversion()

if __name__ == '__main__':
    args = parse_args()

    dataset_dir = args.dataset_dir
    dataset_json_file = os.path.join(dataset_dir, "dataset.json")
    dataset = read_json_file(dataset_json_file)
    
    # Multi-GPU generating
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        visible_gpu_cnt = len(os.environ.get('CUDA_VISIBLE_DEVICES').split(','))
        thread_num = visible_gpu_cnt
    else:
        thread_num = 1
    
    if thread_num == 1:
        main(0, args, dataset)
    else:
        # result_dict = mp.Manager().dict()
        mp = mp.get_context("spawn")
        processes = []

        print('Start Generation')
        for i in range(thread_num):
            p = mp.Process(target=main, args=(i, args, dataset))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # result_dict = dict(result_dict)
