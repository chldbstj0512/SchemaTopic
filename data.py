"""
data: get_batch 등 ETM에서 사용하는 데이터 유틸 (from data import get_batch)
"""
import torch


def get_batch(data_tensor, indices, device):
    """
    data_tensor: (N, V) BOW
    indices: 1-D tensor of doc indices
    returns: (batch_size, V) on device
    """
    if isinstance(indices, torch.Tensor):
        idx = indices
    else:
        idx = torch.tensor(indices, dtype=torch.long)
    return data_tensor[idx].to(device)
