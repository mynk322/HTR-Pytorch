#%%
import torch
import torch.nn as nn

#%%
input = torch.randint(low=1, high=2, size=(2, 3, 4), dtype=torch.float)
print(input)
output = input.softmax(1)
print(output.size())
print(output.detach())

#%%
T = 50      # Input sequence length
C = 20      # Number of classes (including blank)
N = 16      # Batch size
S = 30      # Target sequence length of longest target in batch (padding length)
S_min = 10  # Minimum target length, for demonstration purposes

# Initialize random batch of input vectors, for *size = (T,N,C)
input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()

print(input.size())

#%%
# Initialize random batch of targets (0 = blank, 1:C = classes)
target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
# print(target)
print(target.size())

#%%
input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
# print(input_lengths)
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
# print(target_lengths)

print(input_lengths.size(), target_lengths.size())
ctc_loss = nn.CTCLoss()
loss = ctc_loss(input, target, input_lengths, target_lengths)

loss.backward()

#%%
# Target are to be un-padded
T = 50      # Input sequence length
C = 20      # Number of classes (including blank)
N = 16      # Batch size

# Initialize random batch of input vectors, for *size = (T,N,C)
input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)

# Initialize random batch of targets (0 = blank, 1:C = classes)
target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
target = torch.randint(low=1, high=C, size=(sum(target_lengths),), dtype=torch.long)

ctc_loss = nn.CTCLoss()
loss = ctc_loss(input, target, input_lengths, target_lengths)
loss.backward()

#%%
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[1], [2]])

print(x + y)