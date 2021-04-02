print("yo")
from app_torch import Agent
from env import Game
import numpy as np
import gym
print("outside of main?")

def preprocess_obs(obs):
    return obs
    return obs.astype('float32') / 255

if __name__ == "__main__":
    print("running main?")
    # env = gym.make('BipedalWalker-v3')
    env = Game()
    lr = 0.01
    n_games = 500
    # input_dims = env.observation_space.shape[0]
    # output_dims = env.action_space.n
    input_dims = 2
    output_dims = 2

    agent = Agent(gamma=0.99,epsilon=1.0,lr=lr,input_dims=input_dims,n_actions=output_dims,batch_size=128,epsilon_end=0.01,fname="custom_dqn_model.h5")
    scores = []
    eps_history = []
    for i in range(n_games):
        done = False
        score = 0
        observation = preprocess_obs(env.reset())
        if observation is None:
            import ipdb; ipdb.set_trace()
        while not done:
            # print(observation)
            action = agent.choose_action(observation)
            # import ipdb; ipdb.set_trace()
            observation_, reward, done,info = env.step(action)
            observation_ = preprocess_obs(observation_)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            env.render()
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print(f"episode : {i}. score : {score:.2f}. average score : {avg_score:.2f}. epsilon : {agent.epsilon:.2f}")
    agent.save_model()
    


