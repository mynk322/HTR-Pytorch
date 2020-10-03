import torch
import torch.nn as nn
from tqdm import tqdm

import editdistance
from random import randint

from Image2Text import config, utils

def loss_fn(outputs, targets, device):
    outputs = outputs.to(device)
    output_lengths = torch.full(
        size=(outputs.size()[1],), 
        fill_value=config.TIME_STEPS, 
        dtype=torch.long
    ).to(device)

    #+------------------------------

    target_val = torch.ones((1), dtype=torch.long)
    for target in targets:
        target_val = torch.cat((target_val, utils.stringToClasses(target)), 0)

    #+------------------------------ ABOVE OR BELOW------------------------------

    # targets = [line + (" " * (config.TIME_STEPS - len(line))) for line in targets]
    # target_val = torch.ones((1, config.TIME_STEPS), dtype=torch.long)
    # for target in targets:
    #     target_val = torch.cat((target_val, utils.stringToClasses(target).unsqueeze(0)), 0)
    
    #+------------------------------

    target_val = target_val[1:].to(device)
    target_lengths = torch.tensor(
        [len(target) for target in targets],
        dtype=torch.long
    ).to(device)

    # print("Outputs:")
    # print(outputs)
    # print("Outputs Lengths:")
    # print(output_lengths)
    # print("Targets:")
    # print(target_val)
    # print("Target Lengths:")
    # print(target_lengths)

    loss = nn.CTCLoss(zero_infinity=True)(outputs, target_val, output_lengths, target_lengths)
    return loss

def train_fn(data_loader, test_data_loader, model, optimizer, device, epoch):
    model.train()
    running_loss = 0
    batches = 0 

    for bi, d in enumerate(data_loader):
        optimizer.zero_grad()

        inputs, targets = d["image"], d["text"]
        inputs = inputs.view(-1, 1, config.IMAGE_H, config.IMAGE_W).to(device)
        outputs = model(inputs).to(device).requires_grad_(True)
        loss = loss_fn(outputs, targets, device).to(device)
        running_loss += loss.item()
        batches += 1
        loss.backward()
        optimizer.step()

        if (bi + 1) % 1000 == 0:
            print("[%3d / %3d][%4d / %4d] Loss: %10f | Test Loss: %10f | %-16s | %s |" % 
                (
                    epoch + 1, 
                    config.N_EPOCHS, 
                    bi + 1, 
                    len(data_loader), 
                    loss.item(),
                    eval_fn(test_data_loader, model, device),
                    targets[0],
                    one_sample(d["image"][0], model, device)
                )
            )
        elif bi == 0 or (bi + 1) % 10 == 0 or bi == len(data_loader) - 1:
            print("[%3d / %3d][%4d / %4d] Loss: %10f | %-16s | %s |" % 
                (
                    epoch + 1, 
                    config.N_EPOCHS, 
                    bi + 1, 
                    len(data_loader), 
                    loss.item(),
                    targets[0],
                    one_sample(d["image"][0], model, device)
                )
            )


    return running_loss / batches

def one_sample(input, model, device):
    model.eval()
    input = input.view(1, 1, config.IMAGE_H, config.IMAGE_W).to(device)
    output = model(input).to(device).requires_grad_(True).view(config.TIME_STEPS, config.N_CLASSES)
    output = torch.argmax(output, 1)
    s = "".join([config.ID2CHAR[id.item()] for id in output])
    model.train()
    return s


def eval_fn(data_loader, model, device):
    model.eval()
    running_loss = 0
    batches = 0

    for bi, d in enumerate(data_loader):
        inputs, targets = d["image"], d["text"]
        inputs = inputs.view(-1, 1, config.IMAGE_H, config.IMAGE_W).to(device)
        outputs = model(inputs).to(device)

        loss = loss_fn(outputs, targets, device).to(device)
        running_loss += loss.item()
        batches += 1

    model.train()
    return running_loss / batches

def validate(data_loader, model, device):
    model.eval()

    index = randint(0, len(data_loader) - 1)

    for bi, d in enumerate(data_loader):
        if not bi == index:
            continue
        inputs, targets = d["image"], d["text"]
        inputs = inputs.view(-1, 1, config.IMAGE_H, config.IMAGE_W).to(device)

        for i, input in enumerate(inputs):
            output = one_sample(input, model, device)
            output_decoded = utils.bestPathDecoding(output)
            distance = editdistance.eval(output_decoded, targets[i])
            if distance == 0:
                print("[Correct]      %-16s | %s" % (targets[i], output_decoded))
            else:
                print("[Distance: %d] %-16s | %s" % (distance, targets[i], output_decoded))

    model.train()


# print(loss_fn(
#     outputs=torch.rand((64, 1, 80), dtype=torch.float).log_softmax(2),
#     targets=["Hi my name is Kevin"],
#     device=torch.device("cuda")
# ))