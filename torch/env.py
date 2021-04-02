import numpy as np
import numpy
import time
from numpy.random import random
from processing_py import *
# GAME SETTINGS
WIDTH = 600
HEIGHT = 600
FRAME_RATE = 60

# PLAYER SETTINGS
GRAVITY_ACC = 10
JUMP_ACC = -60
PLAYER_WIDTH = 50
PLAYER_HEIGHT = 50
PLAYER_COLOR = (0, 255, 0)

# OBSTACLE SETTINGS
OBSTACLE_COLOR = (255, 255, 0)
OBSTACLE_HEIGHT = 30
OBSTACLE_WIDTH = 30
OBSTACLE_Y = HEIGHT - OBSTACLE_HEIGHT
SCROLL_SPEED = 20

def constrain(min_val,val,max_val):
    if val < min_val: return min_val
    if val > max_val : return max_val
    return val

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
            SCROLL_SPEED = constrain(15,SCROLL_SPEED,25)

    def update(self):
        self.handle_recycle()
        self.x -= SCROLL_SPEED
class Player:
    def __init__(self):
        self.pos = np.array([20, HEIGHT - PLAYER_HEIGHT])
        self.vel_y = 0
        self.past_qs = None
        self.score = 0
        self.reward = 1
        self.can_jump = False
        self.died = False

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
        self.score += self.get_reward()
        self.pos[1] += self.vel_y
        self.collide_floor()
        self.apply_gravity()




class Game:
    def __init__(self):
        self.init_game()
        self.app = App(WIDTH,HEIGHT)

    def display(self, app: App):
        app.background(51, 51, 51)
        self.obstacle.display(app)
        self.player.display(app)
        app.fill(255,255,255)
        app.textSize(20)
        app.text(f"Player score : {self.player.score}",100,100)

    def render(self):
        self.display(self.app)
        self.app.redraw()
        

    def init_game(self):
        self.player = Player()
        self.obstacle = Obstacle(WIDTH)
        self.player.add_obstacle(self.obstacle)

    def step(self,action):
        '''returns observation , reward, done'''
        self.obstacle.update()
        self.player.execute_action(action)
        self.player.update()
        self.player.collide_obstacle()
        return self.player.get_inputs(), self.player.get_reward(), self.player.died, {}

    def reset(self):
        self.init_game()
        return self.player.get_inputs()
