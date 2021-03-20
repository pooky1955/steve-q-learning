import numpy as np
from processing_py import *

class Player:
    JUMP_SPEED = 30
    JUMP_VEC = np.array([0,-JUMP_SPEED])
    PLAYER_WIDTH = 30
    PLAYER_HEIGHT = 30
    
    def __init__(self,x,y):
        self.pos = np.array([x,y])

    def jump(self):
        self.pos += JUMP_VEC
        
    def display(app):
        app.rect(self.pos[0],self.pos[1],PLAYER_WIDTH,PLAYER_HEIGHT)

    
    

class Obstacle:
    def __init__():
        pass

class Game:
    def __init__():
        pass

app = App(600,400) # create window: width, height
app.background(0,0,0) # set background:  red, green, blue
app.fill(255,255,0) # set color for objects: red, green, blue
app.rect(100,100,200,100) # draw a rectangle: x0, y0, size_x, size_y
app.fill(0,0,255) # set color for objects: red, green, blue
app.ellipse(300,200,50,50) # draw a circle: center_x, center_y, size_x, size_y
app.redraw() # refresh the window
