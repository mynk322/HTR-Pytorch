import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.transforms import ToTensor

from Image2Text import config
from Image2Text.model import Image2TextNet
from Image2Text.preprocessing import Rescale, GreyscaleToBlackAndWhite, TransposeImage, GaussianFiltering, AverageFiltering, MedianPool2d, NumpyToTensor
from Image2Text.engine import one_sample
from Image2Text.utils import bestPathDecoding

v = 36

def evaluate(segments):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Image2TextNet()
    model.load_state_dict(torch.load("Image2Text/weights/model_checkpoint_%s.pth" % v))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        TransposeImage(),
        Rescale((config.IMAGE_H, config.IMAGE_W)),
        transforms.Normalize(mean=(192,), std=(81,)),
    #     AverageFiltering(channels=1, kernel_size=5),
        NumpyToTensor(),
        GaussianFiltering(channels=1, kernel_size=5, sigma=1),
        GreyscaleToBlackAndWhite(),
    #     MedianPool2d(kernel_size=5, same=True),
    ])

    # segments = [transform(torch.tensor(segment, dtype=torch.float).unsqueeze(0)) for segment in segments]
    # plt.imshow(one[0][0])
    segments = [transform(torch.tensor(segment, dtype=torch.float).unsqueeze(0)) for segment in segments]
    outputs = [one_sample(segment, model, device) for segment in segments]
    outputs = [bestPathDecoding(output) for output in outputs]
    outputs = " ".join(outputs)

    return outputs


