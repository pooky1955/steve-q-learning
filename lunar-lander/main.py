import gym
import numpy as np
from agent import Agent
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    n_episodes = 50000
    agent = Agent(1e-3, 0.8, env.action_space.n, 64,
                  env.observation_space.shape, 1e-3, 0.01, 1e7)

    for i in range(n_episodes):
        done = False
        state = env.reset()
        score = 0

        while not done:
            action = agent.get_action(state)
            new_state , reward, done, info = env.step(action)
            score += reward

            # if agent.eps < 0.5:
                # env.render()

            agent.remember(action, reward, state, done)
            state = new_state
            agent.learn()

        print(
                f"episode : {i} | memory : {100 * agent.memory.counter / agent.memory.size:.2f} | score : {score:.2f} | epsilon : {agent.eps:.2f}")
