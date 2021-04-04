import numpy as np
import torch
import torch.optim as optim
from torch import Tensor
from torch import nn
from memory import ReplayBuffer
from utils import to_t, get_default_device, to_n

class DuelingDQN(nn.Module):
    def __init__(self, lr, n_actions, input_dims, fc1_dims, fc2_dims):
        super().__init__()
        print(f"Input dims : {input_dims}")
        self.network = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims,fc2_dims),
            nn.ReLU()
        )
        self.A = nn.Linear(fc2_dims,n_actions)
        self.V = nn.Linear(fc2_dims,1)
        self.device = get_default_device()
        self.loss = nn.MSELoss()
        self.optim = optim.Adam(self.parameters(),lr=lr)
        self.network.to(self.device)
        self.A.to(self.device)
        self.V.to(self.device)
    

    def forward(self,inputs : torch.Tensor):
        latent = self.network(inputs)
        return latent

    def predict_grad(self,inputs : np.ndarray):
        input_tens = to_t(inputs)
        latent = self.forward(input_tens)
        advantages = self.A(latent)
        values = self.V(latent)
        # trust me, i tested it on ipython cell
        # it takes advantage, and for each row, it substracts the row mean
        qs = values + (advantages - advantages.mean(dim=1,keepdim=True))
        return qs
    def predict(self,inputs):
        with torch.no_grad():
            return self.predict_grad(inputs)
    def predict_numpy(self,inputs : np.ndarray):
        return to_n(self.predict(to_t(inputs)))
    def get_scores(self,inputs : np.ndarray):
        with torch.no_grad():
            intermediate = self.network(to_t(inputs))
            advantages = self.A(intermediate)
            return to_n(advantages)

    def fit(self,states,actions,rewards,new_states,dones,gamma):
        q_eval = self.predict_numpy(states)
        q_next = self.predict_numpy(new_states)
        batch_size = states.shape[0]
        q_target = np.copy(q_eval)
        batch_index = np.arange(batch_size,dtype=np.int32)
        q_target[batch_index, actions] = rewards + \
            gamma * np.max(q_next, axis=1) * dones
        self.training_step(states, q_target)

    def training_step(self, inputs: torch.Tensor, outputs: torch.Tensor):
        '''training step, but accepting inputs and outputs only as torch tensors converted to correct device'''
        inputs = to_t(inputs)
        outputs = to_t(outputs)
        self.optim.zero_grad()
        predicted = self.predict_grad(inputs)
        loss_val = self.loss(outputs, predicted)
        loss_val.backward()
        self.optim.step()
        return loss_val





