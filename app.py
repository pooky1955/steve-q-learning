import numpy as np
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
    OBSTACLE_Y = HEIGHT - 200
    SCROLL_SPEED = 20
    def __init__(self,x):
        self.x = x
    def display(self,app):
        app.fill(*OBSTACLE_COLOR)
        app.rect(self.x,)
    def update(self):
        self.x -= SCROLL_SPEED


class Game:
    def __init__():
        pass

app = App(WIDTH,HEIGHT) # create window: width, height
app.background(0,0,0) # set background:  red, green, blue
app.fill(255,255,0) # set color for objects: red, green, blue
app.rect(100,100,200,100) # draw a rectangle: x0, y0, size_x, size_y
app.fill(0,0,255) # set color for objects: red, green, blue
app.ellipse(300,200,50,50) # draw a circle: center_x, center_y, size_x, size_y
app.redraw() # refresh the window
