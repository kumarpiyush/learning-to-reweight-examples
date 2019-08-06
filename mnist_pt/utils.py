import torch

def to_cuda(x) :
    return x.cuda() if torch.cuda.is_available() else x
