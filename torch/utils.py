import numpy as np
import torch
def get_default_device():
    # return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_t(data, device=get_default_device()) -> torch.Tensor:

    if isinstance(data, np.ndarray):
        return torch.Tensor(data).to(device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    assert False, "used to_tensor on something that was not np.array nor torch.Tensor"
