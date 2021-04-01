import numpy as np
import torch
import torch.optim as optim
from torch import Tensor
from torch import nn
import keras
from keras.layers import *
from keras.optimizers import Adam
from keras.models import load_model


class ReplayBuffer():
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, input_dims))
        self.new_state_memory = np.zeros((self.mem_size, input_dims))
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]
        return states, actions, rewards, states_, terminal


def get_default_device():
    # return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def to_t(data, device=get_default_device()) -> torch.Tensor:

    if isinstance(data, np.ndarray):
        return torch.Tensor(data).to(device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    assert False, "used to_tensor on something that was not np.array nor torch.Tensor"


class DQNModel(nn.Module):
    def __init__(self, lr, n_actions, input_dims, fc1_dims, fc2_dims):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions)
        )
        self.device = get_default_device()
        self.loss = nn.MSELoss()
        self.optim = optim.Adam(self.parameters(),lr=lr)
        self.network.to(self.device)

    def predict(self, inputs: np.ndarray):
        with torch.no_grad():
            inputs_tens = to_t(inputs)
            return self.forward(inputs_tens)
    def predict_numpy(self,inputs : np.ndarray):
        return self.predict(inputs).detach().numpy()

    def forward(self, inputs: torch.Tensor):
        '''forward is the method called when invoking __call__ (inherited from nn.Module)'''
        outputs = self.network(inputs)
        return outputs

    def training_step(self, inputs: torch.Tensor, outputs: torch.Tensor):
        '''training step, but accepting inputs and outptus only as torch tensors converted to correct device'''
        inputs = to_t(inputs)
        outputs = to_t(outputs)
        self.optim.zero_grad()
        predicted = self(inputs)
        # import ipdb; ipdb.set_trace()
        loss_val = self.loss(outputs, predicted)
        loss_val.backward()
        self.optim.step()
        return loss_val


def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = keras.Sequential([
        Dense(fc1_dims, activation="relu"),
        Dense(fc2_dims, activation="relu"),
        Dense(n_actions, activation=None)
    ])
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="mean_squared_error")
    return model


class Agent:
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=1e-3, epsilon_end=0.01, mem_size=int(1e6), fname="dqn_model.h5"):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.model_file = fname
        self.epsilon = epsilon
        self.epsilon_min = epsilon_end
        self.epsilon_dec = epsilon_dec
        self.q_eval = DQNModel(lr, n_actions, input_dims, 256, 256)
        self.memory = ReplayBuffer(mem_size, input_dims)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict_numpy(state)
            action = np.argmax(actions)
        return action

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return
        states, actions, rewards, states_, dones = self.memory.sample_buffer(
            self.batch_size)
        q_eval = self.q_eval.predict_numpy(states)
        q_next = self.q_eval.predict_numpy(states_)
        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, actions] = rewards + \
            self.gamma * np.max(q_next, axis=1) * dones
        self.q_eval.training_step(states, q_target)
        self.epsilon = self.epsilon - \
            self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def save_model(self):
        print('heres debugger for saving if you want to')
        import ipdb; ipdb.set_trace()
        # self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
