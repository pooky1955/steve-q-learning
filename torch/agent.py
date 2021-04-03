import numpy as np
from memory import ReplayBuffer


class Agent:
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=1e-3, epsilon_end=0.01, mem_size=int(1e6), fname="dqn_model.h5", memory=None, model=None):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.model_file = fname
        self.epsilon = epsilon
        self.epsilon_min = epsilon_end
        self.epsilon_dec = epsilon_dec
        if model is None:
            self.q_eval = DQNModel(lr, n_actions, input_dims, 512, 512)
        else:
            self.q_eval = model
        if memory is None:
            self.memory = ReplayBuffer(mem_size, input_dims)
        else:
            self.memory = memory

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.get_scores(state)
            action = np.argmax(actions)
        return action

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return
        states, actions, rewards, new_states, dones = self.memory.sample_buffer(
            self.batch_size)
        self.q_eval.fit(states, actions, rewards,
                           new_states, dones, gamma=self.gamma)
        self.epsilon = self.epsilon - \
            self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def save_model(self):
        print('heres debugger for saving if you want to')
        import ipdb
        ipdb.set_trace()
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
