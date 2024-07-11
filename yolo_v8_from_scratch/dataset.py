# import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, Dataloader
from utils import (
    iou_width_height_as_iou,
    non_max_suppression_as_nms,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        label_dir: str,
        anchors,
        #image_size: int=416,
        S: list=[13,26,52],
        C: int=20,
        transform=None,
    ) -> None:
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.amount_of_anchors = self.anchors.shape[0]
        self.amount_of_anchors_per_scale = self.amount_of_anchors // 3 # three scales
        self.C = C # channels
        self.ignore_iou_thresh = 0.5 # intersection over union threshold above which you start counting
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index,1])
        # current order is: [class,x,y,w,h], albumentations needs this order: [x,y,w,h,class]
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        
        img_path = os.path.join(self.label_dir, self.annotations.iloc[index,0])
        
        # these augmentations are done using albumentations:
        if self.transforms:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
            
        targets = [torch.zeros((self.num_anchors//3,S,S,6)) for S in self.S] # [p0,x,y,w,h,class] - these are SIX fields
        
        for box in bboxes:
            # each of the 3 scales should do a prediction and we want to assign which anchor is responsible
            iou_anchors = iou(torch.tensor(bboxes[2:4]), self.anchors) # IOU for a particulate box and ALL anchors
            # lets check which anchors were the best:
            anchor_indices_descending_quality = iou_anchors.argsort(descending=True, dim=0)
            # obtain coordinates:
            x, y, width, height, class_label = box
            # need to be sure that for each scale and each bbox there is an anchor:
            has_anchor = [False, False, False] # these need to become true
            
            
            for anchor_idx in anchor_indices_descending_quality:
                scale_idx = anchor_idx // self.amount_of_anchors_per_scale # 0,1,2
                anchor_on_scale = anchor_idx % self.amount_of_anchors_per_scale # 0,1,2
                S = self.S[scale_idx] # selecting the right scale
                i_y, j_x = int(S*y), int(S*x) # if e.g. x=0.5, S=13 --> int(6.5) = 6
                anchor_taken = targets[scale_idx][anchor_on_scale, i_y, j_x, 0] # in rare cases, this anchor can already be not available (has been taken)
                
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i_y, j_x, 0] = 1
                    x_cell, y_cell = S*x - j_x, S*y - i_y # in between [0...1]
                    width_cell, height_cell = (
                        width * S, # getting width of cell using scale
                        height * S,
                    )
                    box_coordinates = torch.tensor( 
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i_y, j_x, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i_y, j_x, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                    
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    # here we want to IGNORE anchors if they are not taken yet and if the iou is over a certain threshold:
                    targets[scale_idx][anchor_on_scale, i_y, j_x, 0] = -1 # == -1: IGNORE this prediction
                    
        return image, tuple(targets)