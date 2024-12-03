import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import random

from utils.img_group import build_triplet
from utils.misc import get_img_trans
from configs.dataset_config import DatasetConfig
     

class ImageTripletDataset(Dataset):
     """
     Dataset for training the SD-GFA model,
     Return a (img_x1, img_x2, img_y) and two different keys
     """
     def __init__(self, config:DatasetConfig, dataset_type:str):
          '''
          params:
               config: DatasetConfig object
               dataset_type: str, "train" or "val" or "test"
          '''
          self.config = config
          self.image_size = config.img_size
          self.norm_type = config.norm_type

          if dataset_type == "train":
               self.data_df = pd.read_csv(config.train_metadata_path)
          elif dataset_type == "val":
               self.data_df = pd.read_csv(config.val_metadata_path)
          elif dataset_type == "test":
               self.data_df = pd.read_csv(config.test_metadata_path)
          else:
               raise ValueError(f"Invalid dataset_type {dataset_type}")
          self.triplets = build_triplet(self.data_df)
          self.img_trans = get_img_trans(self.image_size, norm_type=self.norm_type)

     def __len__(self):
          return len(self.triplets)

     def __getitem__(self, idx):
          triplet = self.triplets[idx]
          img_x1 = Image.open(triplet[0])
          img_x2 = Image.open(triplet[1])
          img_y = Image.open(triplet[2])
          key1 = torch.randint(low=0, high=2, size=(8,), dtype=torch.float, requires_grad=False)
          key2 = torch.randint(low=0, high=2, size=(8,), dtype=torch.float, requires_grad=False)
          while torch.all(key1==key2):
               key2 = torch.randint(low=0, high=2, size=(8,), dtype=torch.float, requires_grad=False)
          return self.img_trans(img_x1), self.img_trans(img_x2),\
                self.img_trans(img_y), key1, key2


class GenerationDataset(Dataset):
     """
     Dataset used for generation. Using only test data
     Return a image and a random key
     """
     def __init__(self, config:DatasetConfig):
          data_df = pd.read_csv(config.test_metadata_path)
          self.img_paths = data_df['img_path'].values
          self.img_trans = get_img_trans(config.img_size, norm_type=config.norm_type)

     def __len__(self):
          return len(self.img_paths)

     def __getitem__(self, idx):
          img_path = self.img_paths[idx]
          img = Image.open(img_path)
          key = torch.randint(low=0, high=2, size=(8,), dtype=torch.float, requires_grad=False)
          return self.img_trans(img), key


if __name__ == "__main__":
     config = DatasetConfig()
     train_dataset = ImageTripletDataset(config, "train")
     print(len(train_dataset))
     print(train_dataset[0])
     