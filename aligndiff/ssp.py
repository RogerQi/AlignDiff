from aligndiff.SSP_efficient.dataset.transform import crop, hflip, normalize
from aligndiff.SSP_efficient.model.SSP_matching import SSP_MatchingNet

import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

class ssp_segmenter:
    def __init__(self, checkpoint_path, device, backbone='resnet101', refine=True):
        self.model = SSP_MatchingNet(backbone, refine, pretrained=False)

        print('Evaluating model:', checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model.to(device)
        self.model.eval()
    
    def set_support_set(self, support_path_list):
        """Provide support image-mask pairs to the mode

        Args:
            support_path_list (list): [(img1_path, mask1_path), ...]
        """
        img_s_list = []
        mask_s_list = []
        for img_path, mask_path in support_path_list:
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            img, mask = normalize(img, mask)
            mask[mask > 0] = 1
            img_s_list.append(img.unsqueeze(0))
            mask_s_list.append(mask.unsqueeze(0))
        self.model.set_support_set(img_s_list, mask_s_list)

    def query(self, img_path):
        """Query the model with the query image

        Args:
            img_path (str): path to the query image

        Returns:
            mask (np.ndarray): the predicted mask
        """
        query_img = Image.open(img_path).convert('RGB')
        query_img = normalize(query_img, None)
        query_img = query_img.unsqueeze(0)
        with torch.no_grad():
            pred = self.model.fast_forward(query_img)[0]
            pred = pred.softmax(dim=1)
            pred = pred[0, 1]  # foreground probability
        return pred

    def reset(self):
        """Reset the model (i.e., clear support prototypes)
        """
        self.model.FP = None
        self.model.BP = None
