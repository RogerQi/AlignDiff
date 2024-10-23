import os

from aligndiff.mask_generator import mask_generator
from aligndiff.utils import parse_args, read_json_file

def main(args):
    dataset_dir = args.dataset_dir
    dataset_json_file = os.path.join(dataset_dir, "dataset.json")
    dataset = read_json_file(dataset_json_file)

    my_mask_generator = mask_generator(args.fss_ckpt_path)
    for prompt_idx in range(len(dataset)):
        item = dataset[prompt_idx]
        coarse_dir = os.path.join(args.dataset_dir, 'synthesized', 'aligndiff_{}'.format(str(prompt_idx).zfill(6)))
        good_dir = os.path.join(args.dataset_dir, item['D_good_dir'])
        # Gather image-mask pairs
        D_good = []
        for img_fn in os.listdir(good_dir):
            if img_fn.endswith('.jpg'):
                img_path = os.path.join(good_dir, img_fn)
                mask_path = os.path.join(good_dir, img_fn.replace('.jpg', '.png'))
                if os.path.exists(mask_path):
                    D_good.append((img_path, mask_path))
        
        my_mask_generator.generate_mask_one_class(D_good, coarse_dir, args.consensus_iou_threshold)

if __name__ == "__main__":
    args = parse_args()
    main(args)
