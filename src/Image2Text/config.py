import torch

# ID #0 is for CTC blank
CLASSES         = [' ', '!', '\"', '#', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
CHAR2ID         = {' ': 1, '!': 2, '"': 3, '#': 4, '&': 5, "'": 6, '(': 7, ')': 8, '*': 9, '+': 10, ',': 11, '-': 12, '.': 13, '/': 14, '0': 15, '1': 16, '2': 17, '3': 18, '4': 19, '5': 20, '6': 21, '7': 22, '8': 23, '9': 24, ':': 25, ';': 26, '?': 27, 'A': 28, 'B': 29, 'C': 30, 'D': 31, 'E': 32, 'F': 33, 'G': 34, 'H': 35, 'I': 36, 'J': 37, 'K': 38, 'L': 39, 'M': 40, 'N': 41, 'O': 42, 'P': 43, 'Q': 44, 'R': 45, 'S': 46, 'T': 47, 'U': 48, 'V': 49, 'W': 50, 'X': 51, 'Y': 52, 'Z': 53, 'a': 54, 'b': 55, 'c': 56, 'd': 57, 'e': 58, 'f': 59, 'g': 60, 'h': 61, 'i': 62, 'j': 63, 'k': 64, 'l': 65, 'm': 66, 'n': 67, 'o': 68, 'p': 69, 'q': 70, 'r': 71, 's': 72, 't': 73, 'u': 74, 'v': 75, 'w': 76, 'x': 77, 'y': 78, 'z': 79}
ID2CHAR         = {0: '~', 1: ' ', 2: '!', 3: '"', 4: '#', 5: '&', 6: "'", 7: '(', 8: ')', 9: '*', 10: '+', 11: ',', 12: '-', 13: '.', 14: '/', 15: '0', 16: '1', 17: '2', 18: '3', 19: '4', 20: '5', 21: '6', 22: '7', 23: '8', 24: '9', 25: ':', 26: ';', 27: '?', 28: 'A', 29: 'B', 30: 'C', 31: 'D', 32: 'E', 33: 'F', 34: 'G', 35: 'H', 36: 'I', 37: 'J', 38: 'K', 39: 'L', 40: 'M', 41: 'N', 42: 'O', 43: 'P', 44: 'Q', 45: 'R', 46: 'S', 47: 'T', 48: 'U', 49: 'V', 50: 'W', 51: 'X', 52: 'Y', 53: 'Z', 54: 'a', 55: 'b', 56: 'c', 57: 'd', 58: 'e', 59: 'f', 60: 'g', 61: 'h', 62: 'i', 63: 'j', 64: 'k', 65: 'l', 66: 'm', 67: 'n', 68: 'o', 69: 'p', 70: 'q', 71: 'r', 72: 's', 73: 't', 74: 'u', 75: 'v', 76: 'w', 77: 'x', 78: 'y', 79: 'z'}
# ~ is the blank character
N_CLASSES       = len(CLASSES) + 1 # 1 for blank for CTC

# Dataset Information
MAX_H           = 342
AVG_H           = 122.32719239122295
MAX_W           = 1581
AVG_W           = 1697.893881524751
MAX_RATIO_H_W   = 0.7052631578947368
MAX_RATIO_W_H   = 40.04
MAX_LEN         = 97
MIN_LEN         = 1
AVG_LEN         = 42.684190818542646

# Model Config
USE_RESNET      = False
RESNET_TRAIN    = False
IMAGE_C         = 1
IMAGE_H         = 128
IMAGE_W         = 32
MAX_LEN_ALLOWED = 32
CONV_CHANNELS   = [32, 64, 128, 128, 256]
CONV_KERNEL     = [5, 5, 3, 3, 3, 21]
CONV_STRIDE     = [1, 1, 1, 1, 1, 1]
CONV_PADDING    = [2, 2, 1, 1, 1, 10]
BATCH_NORM      = [1, 1, 1, 1, 1, 1] # 1 means we will have a Batch Normalization Layer
LEAKY_RELU      = [0, 0, 0, 0, 0, 0]
DROPOUT         = [0, 0, 0, 0, 0, 0] # 0 means we will not have any Dropout
MAX_POOLING     = [ 
        [(2, 2), (2, 2)], 
        [(2, 2), (2, 2)], 
        [(1, 2), (1, 2)], 
        [(1, 2), (1, 2)], 
        [(1, 2), (1, 2)], 
    ]
NUM_LAYERS      = len(CONV_CHANNELS)

TIME_STEPS      = 32
RNN_INPUT_SIZE  = 256
RNN_HIDDEN_SIZE = 256
RNN_LAYERS      = 2
BIDIRECTIONAL   = True
RNN_DROPOUT     = 0
THRESHOLD       = 0

# Train Config
BATCH_SIZE      = 32
N_EPOCHS        = 100
LEARNING_RATE   = 0.01
EARLY_STOPPING  = 5

TEST_SIZE       = 4096
TEST_BATCHES    = TEST_SIZE // BATCH_SIZE
DATASET_PATH    = "../../data/words/"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")