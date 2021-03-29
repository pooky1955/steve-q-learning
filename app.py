import keras
import numpy as np
import numpy
from typing import Tuple, List, Dict
import traceback
from random import random
from model import create_model, INPUT_SHAPE
import time
from processing_py import *
import keyboard
print("hello?")
# GAME SETTINGS
WIDTH = 600
HEIGHT = 600
FRAME_RATE = 60

# PLAYER SETTINGS
GRAVITY_ACC = 10
JUMP_ACC = -70
PLAYER_WIDTH = 50
PLAYER_HEIGHT = 50
PLAYER_COLOR = (0, 255, 0)

# OBSTACLE SETTINGS
OBSTACLE_COLOR = (255, 255, 0)
OBSTACLE_HEIGHT = 30
OBSTACLE_WIDTH = 30
OBSTACLE_Y = HEIGHT - OBSTACLE_HEIGHT
SCROLL_SPEED = 40

# DQN SETTINGS
GAMMA = 0.7
epsilon = 0.2
MEMORY_SIZE = int(1e2)
BATCH_SIZE = 64


class Obstacle:
    def __init__(self, x):
        self.x = x

    def display(self, app: App):
        app.fill(*OBSTACLE_COLOR)
        app.rect(self.x, OBSTACLE_Y, OBSTACLE_WIDTH, OBSTACLE_HEIGHT)

    def handle_recycle(self):
        global SCROLL_SPEED
        if self.x < -OBSTACLE_WIDTH:
            self.x = int(WIDTH + random() * 200)
            SCROLL_SPEED += (random() - 0.5) * 5

    def update(self):
        self.handle_recycle()
        self.x -= SCROLL_SPEED


class Game:
    def __init__(self, model: keras.Model):
        self.model = model
        self.init_game()

    def display(self, app: App):
        app.background(51, 51, 51)
        self.obstacle.display(app)
        self.player.display(app)

    def init_game(self):
        self.player = Player(self.model)
        self.obstacle = Obstacle(WIDTH)
        self.player.add_obstacle(self.obstacle)

    def update(self):
        self.obstacle.update()
        self.player.update()
        self.player.collide_obstacle()


class ExperienceBuffer:
    def __init__(self, memory_size: int, batch_size: int, gamma: float, input_shape: Tuple[int, ...]):
        self.memory_size = memory_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.mem_count = 0
        self.filled = False
        self.input_shape = input_shape
        state_shape = (int(self.memory_size), *self.input_shape)
        self.buffers = {
            "curr_state": np.zeros(state_shape),
            "next_state": np.zeros(state_shape),
            "reward": np.zeros((self.memory_size)),
            "action": np.zeros((self.memory_size),dtype="int"),
            "done": np.zeros((self.memory_size))
        }

    def add(self, experience_dict: Dict[str, any]):
        if self.mem_count == self.memory_size - 1:
            self.filled = True
        for key, val in experience_dict.items():
            self.buffers[key][self.mem_count] = val
        self.mem_count = (self.mem_count + 1) % self.memory_size

    def can_learn(self):
        '''returns boolean when memory_counter > memory_size'''
        return self.filled

    def learn(self, model):
        sample_inds = np.random.randint(0, self.memory_size, self.batch_size)

        def get_buffer(buffer_name):
            return self.buffers[buffer_name][sample_inds]
        # get the info
        inputs = get_buffer("curr_state")
        # shape of (batch_size, max_actions)
        expected_qs = model.predict(inputs)
        # shape of (batch_size,) e.g. only the optimal value of q for ONE action
        expected_q = get_buffer("reward") + self.gamma * (1 - get_buffer("done")) * \
            np.argmax(model.predict(get_buffer("next_state")), axis=-1)
        qs_range = np.arange(0, expected_qs.shape[0], 1)
        expected_qs[qs_range, get_buffer("action")] = expected_q
        model.train_on_batch(inputs, expected_qs)


class Agent:
    def __init__(self):
        self.model = create_model()
        self.init_agent()
        self.ep_count = 0
        self.epsilon = 0.1

    def display(self, app: App):
        self.game.display(app)

    def init_agent(self):
        self.game = Game(self.model)
        self.player = self.game.player

    def think(self) -> Tuple[numpy.ndarray, int, int, numpy.ndarray, int]:
        '''makes one tick in the game. returns curr_state, curr_action, reward, next_state, done'''
        curr_state = self.player.get_inputs()
        preds = self.model.predict(curr_state)[0]
        curr_action = np.argmax(preds, axis=-1)
        if np.random.random() < self.epsilon:
            # do random action
            selected_ind = np.random.randint(0,preds.shape[0]-1)
            selected_ind = selected_ind if selected_ind != curr_action else preds.shape[0] - 1
            print(f"doing random action {selected_ind} instead of {curr_action}")
            curr_action = selected_ind
        if random() < 0.1:
            print(f"scores : {preds}")
        next_state, reward, done = self.step(curr_action)
        return curr_state, curr_action, reward, next_state, done

    def step(self, action: int) -> Tuple[np.ndarray, int, int]:
        '''returns next state, reward, and done'''
        self.player.execute_action(action)
        self.game.update()
        return self.player.get_inputs(), self.player.get_reward(), int(self.player.died)

    def new_episode(self):
        self.init_agent()
        self.epsilon = 0.1 * 0.99 ** self.ep_count
        self.ep_count += 1


class Player:
    def __init__(self, model: keras.Model):
        self.pos = np.array([20, HEIGHT - PLAYER_HEIGHT])
        self.vel_y = 0
        self.past_qs = None
        self.reward = 1
        self.can_jump = False
        self.died = False
        self.model = model

    def add_obstacle(self, obstacle: Obstacle):
        self.obstacle = obstacle

    def jump(self):
        if self.can_jump:
            self.can_jump = False
            self.vel_y = JUMP_ACC

    def collide_floor(self):
        floor_y = HEIGHT - PLAYER_HEIGHT
        if self.pos[1] > floor_y:
            self.can_jump = True
            self.pos[1] = floor_y
            self.vel_y = 0

    def collide_obstacle(self):
        if self.obstacle.x <= self.pos[0] + PLAYER_WIDTH and self.obstacle.x >= self.pos[0] and self.pos[1] + PLAYER_HEIGHT > HEIGHT - OBSTACLE_HEIGHT:
            self.died = True

    def display(self, app: App):
        app.fill(*PLAYER_COLOR)
        app.rect(self.pos[0], self.pos[1], PLAYER_WIDTH, PLAYER_HEIGHT)

    def apply_gravity(self):
        self.vel_y += GRAVITY_ACC

    def handle_jump(self):
        pass
        # if keyboard.is_pressed("space"):
        #     self.jump()

    def get_inputs(self) -> np.ndarray:
        distance = (self.obstacle.x - self.pos[0]) / WIDTH
        pos_y = self.pos[1] / HEIGHT
        return np.array([distance, pos_y]).reshape((1, 2))

    def get_reward(self) -> int:
        return 1 if not self.died else -100

    def idle(self):
        pass

    def execute_action(self, action: int):
        actions = [self.idle, self.jump]
        assert action >= 0 and action < len(
            actions), f"Action {action} was not within bounds 0 to {len(actions)} exclusively"
        actions[action]()

    def update(self):

        self.pos[1] += self.vel_y
        self.collide_floor()
        self.handle_jump()
        self.apply_gravity()


app = App(WIDTH, HEIGHT)  # create window: width, height


exp_buffer = ExperienceBuffer(MEMORY_SIZE, BATCH_SIZE, GAMMA, INPUT_SHAPE)
agent = Agent()

# def keyPressed(e):
#     print(e)

if __name__ == "__main__":
    try:
        while True:
            app.redraw()
            agent.display(app)
            curr_state, curr_action, reward, next_state, done = agent.think()
            experience_data = {
                "curr_state": curr_state,
                "action": curr_action,
                "reward": reward,
                "next_state": next_state,
                "done": done
            }
            exp_buffer.add(experience_data)
            if exp_buffer.can_learn():
                exp_buffer.learn(agent.model)
            if done:
                agent.new_episode()
    except KeyboardInterrupt:
        print("\nbye bye")
        app.exit()
    except Exception as e:
        print(e)
        traceback.print_exc()
        app.exit()
