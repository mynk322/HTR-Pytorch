import torch
import config

def stringToClasses(x):
    return torch.tensor(
        [config.CHAR2ID[character] for character in x], 
        dtype=torch.long
    )

def bestPathDecoding(x):
    if len(x) == 0:
        return x

    ret = ""
    ret += x[0]

    for i in range(1, len(x)):
        if x[i] != ret[-1]:
            ret += x[i]
    
    ret = ret.replace("~", "")

    return ret
