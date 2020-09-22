import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from model import Image2TextNet
from dataset import HandWritingLinesDataset
from preprocessing import Rescale, GreyscaleToBlackAndWhite
import engine

# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('runs/summary/')

transform = transforms.Compose([
    Rescale((config.IMAGE_H, config.IMAGE_W)),
])

train_dataset = HandWritingLinesDataset(train=True, transform=transform)
test_dataset = HandWritingLinesDataset(train=False, transform=transform)

data = next(iter(train_dataset))
mean, std = data["image"].mean(), data["image"].std()

transform = transforms.Compose([
    Rescale((config.IMAGE_H, config.IMAGE_W)),
    transforms.Normalize(mean=(mean,), std=(std,)),
    GreyscaleToBlackAndWhite()
])

train_dataset = HandWritingLinesDataset(train=True, transform=transform)
test_dataset = HandWritingLinesDataset(train=False, transform=transform)

train = DataLoader(train_dataset, batch_size=config.BATCH_SIZE)
test = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

print("Training Examples: %d" % (len(train_dataset)))
print("Testing Examples:  %d" % (len(test_dataset)))

print("Training Mean:     %d" % (mean))
print("Training Std:      %d" % (std))

torch.cuda.empty_cache()

net = Image2TextNet()
print(net)

# x = torch.randn(config.BATCH_SIZE, 1, config.IMAGE_H, config.IMAGE_W)
# writer.add_graph(net, x)
# writer.close()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
net.to(device)

# optimizer = optim.SGD(net.parameters(), lr=config.LEARNING_RATE, momentum=0.9)
# optimizer = optim.Adam(params=net.parameters(), lr=config.LEARNING_RATE)
optimizer = optim.RMSprop(params=net.parameters(), lr=config.LEARNING_RATE)
# optimizer = optim.Adadelta(params=net.parameters(), lr=config.LEARNING_RATE)
# optimizer = optim.Adagrad(params=net.parameters(), lr=config.LEARNING_RATE)
# optimizer = optim.AdamW(params=net.parameters(), lr=config.LEARNING_RATE)
# optimizer = optim.SparseAdam(params=net.parameters(), lr=config.LEARNING_RATE)
# optimizer = optim.Adamax(params=net.parameters(), lr=config.LEARNING_RATE)
# optimizer = optim.ASGD(params=net.parameters(), lr=config.LEARNING_RATE)
# optimizer = optim.LBFGS(params=net.parameters(), lr=config.LEARNING_RATE)
# optimizer = optim.Rprop(params=net.parameters(), lr=config.LEARNING_RATE)

for epoch in range(config.N_EPOCHS):
    print("[%3d / %3d] Epoch Started | Learning Rate: %.6f" % (epoch + 1, config.N_EPOCHS, optimizer.param_groups[0]['lr']))
    epoch_loss = engine.train_fn(train, test, net, optimizer, device, epoch)
    print("[%3d / %3d] Epoch Loss: %.6f Test Loss: %.6f" % (epoch + 1, config.N_EPOCHS, epoch_loss, engine.eval_fn(test, net, device)))
    engine.validate(test, net, device)
    torch.save(net.state_dict(), "./weights/model_checkpoint_%s.pth" % (epoch))

    if epoch < 3:
        multiplier = 0.46415888336
    else:
        multiplier = 0.95

    for g in optimizer.param_groups:
        g['lr'] *= multiplier