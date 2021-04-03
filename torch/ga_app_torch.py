'''
this implements AgentGeneration, a class used to perform genetic neuroevolution on deep q networks, using a shared experience buffers as well as integrating novelty search
'''
from app_torch import ReplayBuffer, Agent

class AgentArchive:
    def __init__(self,max_count):
        self.max_count = max_count
        self.agents = []

    def calculate_difference(agent1 : Agent,agent2 : Agent):
        params1 = agent1.get_params()
        params2 = agent2.get_params()
        diff = np.sum(np.abs(params1 - params2))
        return diff
    def exists(agent,threshold):
        return True
        
        


class AgentGeneration:
    def __init__(self,population_size,max_size,input_dims,lr,gamma,n_actions,epsilon,batch_size):
        self.memory = ReplayBuffer(max_size, input_dims)
        self.archive = AgentArchive()
        self.agents = [Agent(lr, gamma, n_actions, epsilon, batch_size, input_dims,memory=self.memory) for _ in range(population_size)]

    def get_agents():
        return [agent for agent in self.agents]

    def save_archive(agent):


    


