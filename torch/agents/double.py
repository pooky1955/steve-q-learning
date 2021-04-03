import numpy as np
from utils import to_t
from agents.simple import DQNModel
class DDQNModel():
    def __init__(self,lr, n_actions, input_dims, fc1_dims, fc2_dims,replace_count):
        self.target_model = DQNModel(lr, n_actions, input_dims, fc1_dims, fc2_dims)
        self.eval_model = DQNModel(lr,n_actions,input_dims,fc2_dims,fc2_dims)
        self.learn_counter = 0
        self.replace_count = replace_count

    def replace_target(self):
        if self.learn_counter % self.replace_count == 0:
            print("currently replacing model!")
            self.target_model.load_state_dict(self.eval_model.state_dict())

    def predict(self,inputs):
        return self.eval_model.predict(inputs)
    def predict_numpy(self,inputs):
        return self.eval_model.predict_numpy(inputs)

    def get_scores(self,inputs : np.ndarray):
        return self.eval_model.predict_numpy(inputs)

    def fit(self,states,actions,rewards,new_states,dones,gamma):
        self.replace_target()
        q_eval = self.eval_model.predict_numpy(states)
        q_next = self.target_model.predict_numpy(new_states)
        batch_size = states.shape[0]
        q_target = np.copy(q_eval)
        batch_index = np.arange(batch_size,dtype=np.int32)
        q_target[batch_index, actions] = rewards + \
            gamma * np.max(q_next, axis=1) * dones
        self.eval_model.training_step(states, q_target)
        self.learn_counter += 1

