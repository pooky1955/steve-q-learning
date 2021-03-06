from app_old import Agent
import numpy as np
import gym
import tensorflow as tf

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    env = gym.make('LunarLander-v2')
    lr = 0.001
    n_games = 500
    agent = Agent(gamma=0.99,epsilon=1.0,lr=lr,input_dims=env.observation_space.shape,n_actions=env.action_space.n,batch_size=64,epsilon_end=0.01,fname="custom_dqn_model.h5")
    scores = []
    eps_history = []
    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done,info = env.step(action)
            env.render()
            score += reward
            agent.store_transition(observation, action, reward , done)
            observation = observation_
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print(f"episode : {i}. score : {score:.2f}. average score : {avg_score:.2f}. epsilon : {agent.epsilon:.2f}")
    agent.save_model()
    


