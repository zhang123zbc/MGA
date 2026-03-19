import os

import numpy as np
from PIL import Image
from torch.utils import data
import pandas as pd
from torchvision import transforms as T
import torch

class ImageNet(data.Dataset):
    def __init__(self, dir, csv_path,size=224):
        self.dir = dir   
        self.csv = pd.read_csv(csv_path)
        self.size=size

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID =img_obj['ImageId']
        Truelabel = img_obj['TrueLabel']
        img_path = os.path.join(self.dir, ImageID)
        pil_img = Image.open(img_path).convert('RGB')
        pil_img = np.array(pil_img).astype(np.float32) / 255
        data = torch.from_numpy(pil_img).permute(2, 0, 1)
        tr=T.Resize(self.size)
        data=tr(data)
        return data, ImageID, Truelabel

    def __len__(self):
        return len(self.csv)

