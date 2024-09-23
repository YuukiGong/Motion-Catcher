import argparse
import contextlib
import gc
import logging
import math
import os
import random
import shutil
import torchvision.transforms.functional as func
from PIL import Image
from pathlib import Path
from torch.utils.data import TensorDataset,Dataset,DataLoader  
from torch.utils.data import RandomSampler,BatchSampler   
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from torchvision import transforms

class VideoDataset(Dataset):#需要继承data.Dataset
    def __init__(self,folder):
        # TODO
        # 1. Initialize file path or list of file names.
        self.folder = folder
        self.optical_name = "optical.pt"
        self.vedio = []
        self.optical = []
        self.image_transforms = transforms.Compose(
        [
            transforms.Resize((576,1024), interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
        for root, dirs, files in os.walk(self.folder):
            if not dirs:  # 如果当前文件夹没有子文件夹，即为最里面的文件夹
                # 在这里可以对最里面的文件夹进行处理
                vedio = [os.path.join(root,frame) for frame in files if not frame.endswith(".pt")]
                vedio = sorted(vedio, key=lambda x: int(''.join(filter(str.isdigit, x))))
                optical = os.path.join(root,self.optical_name)
                self.vedio.append(vedio)
                self.optical.append(optical)
                # files = [np.array(Image.open(os.path.join(root,frame)).resize((width,height))) for frame in files]
            
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        vedio_list = self.vedio[index] # image
        vedio_list = [self.image_transforms(Image.open(frame)).unsqueeze(0) for frame in vedio_list]
        optical_list = torch.load(self.optical[index]) # （14，2，576，1024）
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # vedio_list = [func.resize(func.to_tensor(frame).unsqueeze(0),(576,1024)) for frame in vedio_list]
        # vedio_list = self.image_transforms(vedio_list)
        vedio_list = torch.cat(vedio_list, dim=0)
        # 3. Return a data pair (e.g. image and label).
        return vedio_list, optical_list
        #这里需要注意的是，第一步：read one data，是一个data
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.vedio)
    
if __name__ == "__main__":
    train_dataset = VideoDataset("../datasets/around")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=2,
        num_workers=1,
    )
    features,labels = next(iter(train_dataloader))   
    print("features = ",features )  
    print("labels = ",labels )

