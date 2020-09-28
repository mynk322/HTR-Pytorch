import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import io

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms

import pandas as pd
import numpy as np

import config

# path of the dataset
path = config.DATASET_PATH

class HandWritingLinesDataset(Dataset):
    def __init__(self, train=True, transform=None):
        self.line_data = np.array(pd.read_csv(path + "text_data.csv").to_numpy())
        # self.line_data = self.line_data[:64]

        self.line_data = self.filter_examples(self.line_data)
        self.line_data = np.array(self.line_data)

        if train == True:
            self.line_data = self.line_data[:-config.TEST_SIZE]
        else:
            self.line_data = self.line_data[-config.TEST_SIZE:]

        self.line_data = self.line_data

        self.transform = transform

    def __len__(self):
        return len(self.line_data)

    def __getitem__(self, index):
        name, text = self.line_data[index, 0], self.line_data[index, 1]

        text = self.clean_text(text)
        
        image_path = path + name + ".png"
        try:
            image = io.imread(image_path)
        except:
            print("############# Error for ", name, text)

        if self.transform is not None:
            image = {
                    "image": self.transform(torch.tensor(image, dtype=torch.float).unsqueeze(0)),
                    "text": text
                }
        else:
            image = {
                "image": torch.tensor(image, dtype=torch.float).view(1, config.IMAGE_H, config.IMAGE_W),
                "text": text
            }

        return image


    def getImagePath(self, name_split):
        return path + "lines/" + name_split[0] + "/" + name_split[0] + "-" + name_split[1] + "/" + name_split[0] + "-" + name_split[1] + "-" + name_split[2] + ".png"

    def show_image(self, data):
        if isinstance(data, int):
            item = self.__getitem__(data)
            image = item["image"].squeeze(0)
        else:
            image = data[0].squeeze(0)

        plt.figure()
        plt.imshow(image)
        plt.pause(0.001)
        plt.show()

    def clean_text(self, text):
        text = text.replace("&quot;", "\"")
        text = text.replace("&amp;", "&")
        return text

    def filter_examples(self, line_data):
        line_data = [
            [name, text] for (name, text) in line_data 
            if len(text) >= config.MIN_LEN_ALLOWED and len(text) <= config.MAX_LEN_ALLOWED
        ]
        return line_data