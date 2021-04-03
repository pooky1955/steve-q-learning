import numpy as np
import torch
import torch.optim as optim
from torch import Tensor
from torch import nn
from memory import ReplayBuffer
from utils import to_t, get_default_device

class DuelingDQN(nn.Module):
    def __init__(self, lr, n_actions, input_dims, fc1_dims, fc2_dims):
        super().__init__()
        print(f"Input dims : {input_dims}")
        self.network = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
        )
        self.A = nn.Linear(fc2_dims,n_actions)
        self.V = nn.Linear(fc2_dims,n_actions)
        self.device = get_default_device()
        self.loss = nn.MSELoss()
        self.optim = optim.Adam(self.parameters(),lr=lr)
        self.network.to(self.device)
        self.A.to(self.device)
        self.V.to(self.device)
    

    def forward(self,inputs : torch.Tensor):
        latent = self.network()
        return latent

    def predict(self,inputs : np.ndarray):
        with torch.no_grad():
            input_tens = to_t(inputs)
            latent = self.forward(inputs)
            advantages = self.A(latent)
            return advantages
    def get_scores(self,inputs : np.ndarray):
        intermediate = self.network(inputs)
        self.A(intermediate)
        pass

    def fit(self,states,actions,rewards,new_states,dones,gamma):
        pass





