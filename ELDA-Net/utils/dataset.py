import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class LaneDataset(Dataset):
    def __init__(self, image_dir, label_dir, input_size, train=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.input_size = tuple(input_size)
        self.train = train
        self.image_files = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.input_size)
        label = cv2.resize(label, self.input_size)
        image = torch.tensor(image.transpose(2, 0, 1) / 255.0, dtype=torch.float32)
        label = torch.tensor(label / 255.0, dtype=torch.float32).unsqueeze(0)
        return image, label
