import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="AlignDiff data synthesizer.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        required=True,
        help="The directory containing the dataset. The dataset should be in the format of the example dataset.",
    )
    # SDXL also potentially works
    parser.add_argument(
        "--diffusion_model",
        type=str,
        default="benjamin-paine/stable-diffusion-v1-5",
        help="The model identifier from huggingface.co/models.",
        choices=["CompVis/stable-diffusion-v1-4", "benjamin-paine/stable-diffusion-v1-5"],
    )
    # ===== inversion training =====
    parser.add_argument(
        "--inversion_train_bs",
        type=int,
        default=4,
        help="The batch size for the inversion training.",
    )
    parser.add_argument(
        "--inversion_lr",
        type=float,
        default=5e-04,
        help="The learning rate for the inversion training.",
    )
    parser.add_argument(
        "--inversion_steps",
        type=int,
        default=500,
        help="The number of steps for the inversion training.",
    )
    parser.add_argument(
        "--bg_lambda",
        type=float,
        default=0.6,
        help="The lambda for the background loss. (Eq. 3 in the paper)",
    )
    # ===== image generation settings =====
    parser.add_argument(
        "--image_per_class",
        type=int,
        default=100,
        help="The number of images to generate per class per GPU.",
    )
    # ===== mask generation settings =====
    parser.add_argument(
        "--fss_ckpt_path",
        type=str,
        default="coco_3_resnet101_5shot_39.86.pth",
        help="The path to the FSS checkpoint.",
    )
    parser.add_argument(
        "--consensus_iou_threshold",
        type=float,
        default=0.8,
        help="The IOU threshold for the consensus mask generation (Algo. 1 in the paper).",
    )
    return parser.parse_args()

def read_json_file(json_file_path):
    # Load the JSON data from a file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    prompt_dataset = []

    # Parse the JSON data
    for item in data['prompt_dataset']:
        assert item['type'] in ['text', 'NM_embedding']
        prompt_dataset.append(item)

    return prompt_dataset
