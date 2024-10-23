import os
import numpy as np
from tqdm import tqdm, trange
from PIL import Image
from aligndiff.ssp import ssp_segmenter
import matplotlib.pyplot as plt

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

class mask_generator:
    def __init__(self, fss_checkpoint_path, device='cuda'):
        self.fss_segmenter = ssp_segmenter(fss_checkpoint_path, device)        
        
    def recalibrate_fss(self, img_mask_path_list):
        self.fss_segmenter.reset()

        for img_path, mask_path in img_mask_path_list:
            assert os.path.exists(img_path)
            assert os.path.exists(mask_path)

        self.fss_segmenter.set_support_set(img_mask_path_list)
    
    def segment_one(self, img_path, confidence, viz=False):
        fss_pred = self.fss_segmenter.query(img_path)
        fss_pred = (fss_pred > confidence).cpu().numpy().astype(np.uint8)

        if viz:
            plt.imshow(fss_pred)
            plt.show()

        return fss_pred
    
    def process_one(self, img_path, coarse_mask_path, confidence=0.4, att_cutoff=0.4):
        # Evaluate one image
        fss_pred = self.segment_one(img_path, confidence)

        assert os.path.exists(coarse_mask_path)
        attention_mask = np.array(Image.open(coarse_mask_path)).astype(np.uint8)
        attention_mask = (attention_mask > int(255 * att_cutoff)).astype(np.uint8)
        
        iou = compute_iou(fss_pred, attention_mask)
        
        return fss_pred, iou
    
    def generate_mask_one_class(self, D_good, diffusion_generated_dir, iou_threshold):
        print("Initial D_good size: ", len(D_good))
        # D_good and D_noisy are (img_path, mask_path) pairs as decribed in the paper
        D_noisy = []
        self.recalibrate_fss(D_good)

        generated_img_dir = os.path.join(diffusion_generated_dir, 'generated_images')
        attention_mask_dir = os.path.join(diffusion_generated_dir, 'attention_masks')
        final_mask_dir = os.path.join(diffusion_generated_dir, 'aligndiff_masks')
        os.makedirs(final_mask_dir, exist_ok=True)

        generated_img_fn_list = sorted([fn for fn in os.listdir(generated_img_dir) if fn.endswith('.jpg')])

        for fn in generated_img_fn_list:
            img_path = os.path.join(generated_img_dir, fn)
            mask_path = os.path.join(attention_mask_dir, fn.replace('.jpg', '.png'))
            assert os.path.exists(img_path)
            assert os.path.exists(mask_path)
            D_noisy.append((img_path, mask_path))
        
        fss_pred_mask_list, iou_list = [], []
        
        for img_path, mask_path in tqdm(D_noisy, desc='First FSS-Cross attention run'):
            fss_pred, cur_iou = self.process_one(img_path, mask_path)
            fss_pred_mask_list.append(fss_pred)
            iou_list.append(cur_iou)
        
        D_noisy_new = []
        for i in range(len(D_noisy)):
            img_path, mask_path = D_noisy[i]
            if iou_list[i] > iou_threshold:
                target_mask_path = os.path.join(final_mask_dir, os.path.basename(mask_path))
                Image.fromarray(fss_pred_mask_list[i] * 255).convert('L').save(target_mask_path)
                D_good.append((img_path, target_mask_path))
            else:
                D_noisy_new.append((img_path, mask_path))
        
        print("Final D_good size: ", len(D_good))
        
        # Provide pseudo-labels for the remaining noisy images
        self.recalibrate_fss(D_good)

        for img_path, mask_path in tqdm(D_noisy_new, desc='Pseudo-labeling'):
            fss_pred = self.segment_one(img_path, confidence=0.5)
            target_mask_path = os.path.join(final_mask_dir, os.path.basename(mask_path))
            Image.fromarray(fss_pred * 255).convert('L').save(target_mask_path)

if __name__ == '__main__':
    my_mask_generator = mask_generator('coco_3_resnet101_5shot_39.86.pth')

    D_good = [
        ('./example/example_dataset/dog/1.jpg', './example/example_dataset/dog/1.png')
    ]

    iou_list = my_mask_generator.generate_mask_one_class(D_good,
                                            './example/output/generated_dog',
                                            0.8)
