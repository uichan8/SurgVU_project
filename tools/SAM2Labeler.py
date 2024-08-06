import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor
from PIL import Image
import threading

class SAM2Labeler:
    def __init__(self,checkpoint = "checkpoint/sam2/sam2_hiera_large.pt",cfg = "sam2_hiera_l.yaml",gpu_num = [0,1]):
        self.sam2_checkpoint = checkpoint
        self.model_cfg = cfg
        self.predictor = []
        for n in gpu_num:
            self.gpu_config(n)
            pred = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint)
            pred.to(f"cuda:{n}")
            self.predictor.append(pred)
    
    @staticmethod
    def gpu_config(gpu_num):
        # use bfloat16 for the entire notebook
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(gpu_num).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    @staticmethod
    def show_mask(mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords, labels, ax, marker_size=200):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    def forward(self,)


if __name__ == "__main__":
    labeler = SAM2Labeler()
    print(1)