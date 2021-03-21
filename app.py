import numpy as np
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
JUMP_ACC = -50
#JUMP_VEC = np.array([0, JUMP_AC])
PLAYER_WIDTH = 50
PLAYER_HEIGHT = 50
PLAYER_COLOR = (0, 255, 0)


class Player:
    def __init__(self, x, y):
        self.pos = np.array([x, y])
        self.vel_y = 0
        self.can_jump = False

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

    def display(self, app):
        app.fill(*PLAYER_COLOR)
        app.rect(self.pos[0], self.pos[1], PLAYER_WIDTH, PLAYER_HEIGHT)

    def apply_gravity(self):
        self.vel_y += GRAVITY_ACC

    def handle_jump(self):
        if keyboard.is_pressed("space"):
            self.jump()

    def update(self):
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

    def update(self):
        self.x -= SCROLL_SPEED


class Game:
    def __init__(self):
        self.player = Player(20, HEIGHT - PLAYER_HEIGHT - 800)
        self.obstacle = Obstacle(WIDTH - 200)

    def display(self, app):
        app.background(51, 51, 51)
        self.obstacle.display(app)
        self.player.display(app)

    def update(self):
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
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nbye bye")
        app.exit()
    except Exception as e:
        print(e)
        app.exit()
        