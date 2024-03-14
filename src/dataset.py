import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from config.config import Config
import os
import torch
from torchvision import transforms
from PIL import Image

class WaterMask_Dataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
        data_aug: bool,
        mode=None
    ):
        super().__init__()
        self.index = df.index
        self.df = df
        self.cfg = cfg

        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        watermark_path = '../preprocessed/watermark_jpg/mpu_logo_256.png'
        watermark = Image.open(watermark_path)
        watermark = preprocess(watermark)
        self.watermark = watermark
        self.watermark_embedding = watermark

        white = torch.ones([self.cfg.watermark_decoder.out_ch,self.cfg.watermark_decoder.watermark_size[0],self.cfg.watermark_decoder.watermark_size[1]])
        self.white_img = white

        Z=1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        Image_Path = str(row['Image_Path'])
        npy_Path = str(row['npy_Path'])
        npy = np.load(os.path.join(self.cfg.dataset_dir, npy_Path))
        tensor = torch.from_numpy(npy)
        return {
            "image": tensor,
            "watermark_img": self.watermark,
            "white_img": self.white_img,
            'watermark_embedding': self.watermark_embedding,
        }

