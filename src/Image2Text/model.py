import torch
import torch.nn as nn
import torchvision

import config

from collections import OrderedDict

class Image2TextConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = OrderedDict()

        # Convolution --> Batch Normalization --> ReLU / LeakyReLU --> Dropout --> MaxPooling
        for i in range(config.NUM_LAYERS):
            in_channels = config.IMAGE_C if i == 0 else config.CONV_CHANNELS[i - 1]
            out_channels = config.CONV_CHANNELS[i]

            layers["conv_%d" % (i)] = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.CONV_KERNEL[i],
                    stride=config.CONV_STRIDE[i],
                    padding=config.CONV_PADDING[i]
                )

            if config.BATCH_NORM[i]:
                layers["batch_norm_%d" % (i)] = nn.BatchNorm2d(num_features=out_channels)
            
            if config.LEAKY_RELU[i]:
                layers["leaky_relu_%d" % (i)] = nn.LeakyReLU(config.LEAKY_RELU)
            else:
                layers["relu_%d" % (i)] = nn.ReLU()

            if config.DROPOUT[i]:
                layers["dropout_%d" % (i)] = nn.Dropout(config.DROPOUT[i])
            
            if len(config.MAX_POOLING[i]) > 0:
                layers["max_pooling_%d" % (i)] = nn.MaxPool2d(*config.MAX_POOLING[i])

            
        # *[...] allows unpacking list elements as parameters to a function
        self.context_net = nn.Sequential(layers)

    def forward(self, x):
        return self.context_net(x)

class Image2TextRecurrentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = nn.LSTM(
            input_size=config.RNN_INPUT_SIZE,
            hidden_size=config.RNN_HIDDEN_SIZE,
            num_layers=config.RNN_LAYERS,
            bidirectional=config.BIDIRECTIONAL,
            dropout=config.RNN_DROPOUT
        )
        self.reduce = nn.Linear(
            in_features=(1 + config.BIDIRECTIONAL) * config.RNN_HIDDEN_SIZE,
            out_features=config.N_CLASSES
        )

    def forward(self, x):
        output, _ = self.recurrent(x)
        t, b, h = output.size()
        output = output.view(t * b, h)

        reduced = self.reduce(output)
        log_probs = nn.LogSoftmax(dim=1)(reduced).view(t, b, config.N_CLASSES)

        return log_probs

class Image2TextNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.RNN = Image2TextRecurrentNet()
        if config.USE_RESNET:
            self.resnet = torchvision.models.resnet50(pretrained=True)
            self.resnet_input = nn.Conv2d(
                        in_channels=1,
                        out_channels=3,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    )
            for param in self.resnet.parameters():
                param.requires_grad = False
            self.fc1 = nn.Linear(1000, config.TIME_STEPS * config.RNN_INPUT_SIZE)
        else:
            self.CNN = Image2TextConvNet()
 
    def forward(self, x):
        if config.USE_RESNET:
            output = self.fc1(nn.ReLU()(self.resnet(self.resnet_input(x)))).view(config.TIME_STEPS, -1, config.RNN_INPUT_SIZE)
        else:
            output = self.CNN(x)
            output = output.squeeze(3)
            output = output.permute(2, 0, 1)

        output = self.RNN(output)
        return output

# print(Image2TextNet())
# net1 = Image2TextConvNet().to(config.DEVICE)
# input1 = torch.rand((config.BATCH_SIZE, 1, config.IMAGE_H, config.IMAGE_W)).to(config.DEVICE)
# net2 = Image2TextNet().to(config.DEVICE)
# input2 = torch.rand((config.BATCH_SIZE, 1, config.IMAGE_H, config.IMAGE_W)).to(config.DEVICE)
# print(input1.size())
# print(net1(input1).size())
# print(net2(input2).size())
# print(torch.rand((1, 256, 8, 32)).size())
# print(nn.MaxPool2d((2, 2), (2, 1), (0, 1))(torch.rand((1, 256, 8, 32))).size())