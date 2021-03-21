import numpy as np
import traceback
from random import random
from model import create_model
import time
from processing_py import *
import keyboard

GRAVITY_ACC = 10
WIDTH = 600
HEIGHT = 600
OBSTACLE_COLOR = (255, 255, 0)
OBSTACLE_HEIGHT = 30
OBSTACLE_WIDTH = 30
OBSTACLE_Y = HEIGHT - OBSTACLE_HEIGHT
SCROLL_SPEED = 20
JUMP_ACC = -70
#JUMP_VEC = np.array([0, JUMP_AC])
PLAYER_WIDTH = 50
PLAYER_HEIGHT = 50
PLAYER_COLOR = (0, 255, 0)


class Player:
    def __init__(self, x, y):
        self.pos = np.array([x, y])
        self.vel_y = 0
        self.can_jump = False
        self.died = False
        self.model = create_model()

    def add_obstacle(self,obstacle):
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


    def display(self, app):
        app.fill(*PLAYER_COLOR)
        app.rect(self.pos[0], self.pos[1], PLAYER_WIDTH, PLAYER_HEIGHT)

    def apply_gravity(self):
        self.vel_y += GRAVITY_ACC

    def handle_jump(self):
        if keyboard.is_pressed("space"):
            self.jump()

    def get_inputs(self):
        pass

    def think(self):
        # big brain time
        pass 
    def update(self):
        self.think()


        self.pos[1] += self.vel_y
        self.collide_floor()
        self.handle_jump()
        self.apply_gravity()

class Obstacle:
    def __init__(self, x): 
        self.x = x

    def display(self, app):
        app.fill(*OBSTACLE_COLOR)
        app.rect(self.x, OBSTACLE_Y, OBSTACLE_WIDTH, OBSTACLE_HEIGHT)

    def handle_recycle(self):
        if self.x < -OBSTACLE_WIDTH:
            self.x = int(WIDTH + random() * 100)

    def update(self):
        self.handle_recycle()
        self.x -= SCROLL_SPEED


class Game:
    def __init__(self):
        self.init_game()

    def display(self, app):
        app.background(51, 51, 51)
        self.obstacle.display(app)
        self.player.display(app)
    def init_game(self):
        self.player = Player(20, HEIGHT - PLAYER_HEIGHT)
        self.obstacle = Obstacle(WIDTH)
        self.player.add_obstacle(self.obstacle)

    def update(self):
        self.player.collide_obstacle()
        if self.player.died:
            print("you ded")
            self.player.died = False
            self.init_game()

        self.obstacle.update()
        self.player.update()


app = App(WIDTH, HEIGHT)  # create window: width, height
game = Game()

# def keyPressed(e):
#     print(e)

if __name__ == "__main__":
    try:
        while True:
            game.update()
            game.display(app)
            app.redraw()
            time.sleep(60**-1)
    except KeyboardInterrupt:
        print("\nbye bye")
        app.exit()
    except Exception as e:
        print(e)
        traceback.print_exc() 
        app.exit()
        