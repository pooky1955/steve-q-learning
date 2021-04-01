import numpy as np
import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class Memory:
    def __init__(self, size, state_dims: tuple):
        self.counter = 0
        isize = int(size)
        self.size = isize
        self.rewards = np.zeros(isize)
        self.has_futures = np.zeros(isize)
        self.states = np.zeros((isize, *state_dims))
        self.actions = np.zeros(isize, dtype=np.int)

    # state is the state after the action
    def remember(self, action, reward, state, is_done):
        i = self.counter % self.size

        self.states[i] = state
        self.rewards[i] = reward
        self.has_futures[i] = 1 - int(is_done)
        self.actions[i] = action

        self.counter += 1

    def sample(self, size: int):
        ''' returns  actions, rewards, states, next_states, has_futures'''
        max_i = min(self.counter, self.size)
        ii = np.random.choice(max_i, size, replace=False)
        actions = self.actions[ii]
        rewards = self.rewards[ii]
        states = self.states[ii]
        next_states = self.states[ii + 1 % self.states.shape[0]]
        has_futures = self.has_futures[ii]

        return actions, rewards, states, next_states, has_futures


def build_dqn(lr, n_actions):
    model = keras.Sequential([
        Dense(256, activation="relu"),
        Dense(256, activation="relu"),
        Dense(n_actions, activation=None)
    ])
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="mean_squared_error")
    return model
