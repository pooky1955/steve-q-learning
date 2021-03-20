import numpy as np
import time
from processing_py import *
WIDTH = 600
HEIGHT = 600
class Player:
    JUMP_SPEED = 30
    JUMP_VEC = np.array([0,-JUMP_SPEED])
    PLAYER_WIDTH = 30
    PLAYER_HEIGHT = 30
    PLAYER_COLOR = (0,255,0)
    
    def __init__(self,x,y):
        self.pos = np.array([x,y])

    def jump(self):
        self.pos += JUMP_VEC
        
    def display(app):
        app.fill(*PLAYER_COLOR)
        app.rect(self.pos[0],self.pos[1],PLAYER_WIDTH,PLAYER_HEIGHT)

    
    

class Obstacle:
    OBSTACLE_COLOR = (255,255,0)
    OBSTACLE_HEIGHT = 50
    OBSTACLE_WIDTH = 50
    OBSTACLE_Y = HEIGHT - OBSTACLE_HEIGHT
    SCROLL_SPEED = 20
    def __init__(self,x):
        self.x = x
    def display(self,app):
        app.fill(*OBSTACLE_COLOR)
        app.rect(self.x,OBSTACLE_Y,OBSTACLE_WIDTH,OBSTACLE_HEIGHT)
    def update(self):
        self.x -= SCROLL_SPEED


class Game:
    def __init__(self):
        self.player = Player()
        self.obstacle = Obstacle()
        pass
    def display():
        app.background(51,51,51)
        self.obstacle.display()
        self.player.display()
    def update():
        self.obstacle.update()
        self.player.update()


app = App(WIDTH,HEIGHT) # create window: width, height
game = Game()
while True:
    game.update()
    game.display()
    time.sleep(0.1)