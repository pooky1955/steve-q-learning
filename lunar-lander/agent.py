import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from memory import Memory


def build_dqn(lr, n_actions):
    model = keras.Sequential([
        Dense(256, activation="relu"),
        Dense(256, activation="relu"),
        Dense(n_actions, activation=None)
    ])
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="mean_squared_error")
    return model


class Agent:
    def __init__(self, lr, discount, n_actions, batch_size, state_dims, eps_dec, eps_min, mem_size):
        self.eps = 0.9
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.discount = discount
        self.batch_size = batch_size
        self.model = build_dqn(lr, n_actions)
        self.action_space = np.arange(n_actions)
        self.memory = Memory(mem_size, state_dims)

    def remember(self, action, reward, state, is_done):
        self.memory.remember(action, reward, state, is_done)

    def get_action(self, data):
        if np.random.random() < self.eps:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([data])
            actions = self.model.predict(state)
            action = np.argmax(actions)
        return action

    def learn(self):
        if self.memory.counter < self.batch_size:
            return  # do not do anything until batch size is big enough

        actions, rewards, states, next_states, has_futures = self.memory.sample(
            self.batch_size)

        q_values = np.copy(self.model.predict(states))
        next_q_values = self.model.predict(next_states)

        ii = np.arange(self.batch_size, dtype=int)
        q_values[ii, actions] = rewards +  has_futures * self.discount * \
            np.max(next_q_values, axis=1) 

        self.model.train_on_batch(states, q_values)
        self.eps = self.eps -  self.eps_dec if self.eps > self.eps_min else self.eps_min
