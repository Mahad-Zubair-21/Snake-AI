import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple
#---------------------------------------#
pygame.init()
font = pygame.font.SysFont('arial', 25)

#--using constants to represent the direction--#
class Direction(Enum):
    right = 1
    left = 2
    up = 3
    down = 4
    
Point = namedtuple('Point','x,y')

#--Color Schemes for  the Game--#
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20#--size of the snake's body--#
SPEED = 20 #--speed of the game--#
#----------------------------------------------------#
class SnakeGameAI:
    #-- to setup the game window/state--#
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
#----------------------------------------------------#
    def reset(self):
        self.direction = Direction.right
        self.head = (self.w // 2, self.h // 2)
        self.snake = [self.head, 
                      (self.head[0] - BLOCK_SIZE, self.head[1]),
                      (self.head[0] - (2 * BLOCK_SIZE), self.head[1])]
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0
#----------------------------------------------------#
    #--randomly place the food on the screen--#
    def place_food(self):
        self.food = (
            random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE,
            random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        )
        if self.food in self.snake:
            self.place_food()
#----------------------------------------------------#
    #--handles events occurring in the game--#
    def play_step(self,action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        #--handles movement--#
        self._move(action)
        self.snake.insert(0, self.head)
        #--Check if game is over--#
        reward  = 0
        game_over = False
        #--this checks if snake either collides or is continuously moving without any upgrade 
        if self.collision() or self.frame_iteration> 100*len(self.snake):
            game_over = True
            reward = -10
            return reward,game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = +10
            self.place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)
        return reward,game_over, self.score
#----------------------------------------------------#
    #--handles collision of snake--#
    def collision(self,pt=None):
        if pt is None:
            pt = self.head
        if (
            pt.x > self.w - BLOCK_SIZE or pt.x < 0 or
            pt.y > self.h - BLOCK_SIZE or pt.y < 0
        ):
            return True
        if pt in self.snake[1:]:
            return True
        return False
#----------------------------------------------------#
    #-- updates the game display by drawing the snake,food, and score--#
    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt[0] + 4, pt[1] + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
#----------------------------------------------------#
    def _move(self,action):
        #--[straight,right,left]
        clock_wise = [Direction.right,Direction.down,Direction.left,Direction.up]
        index = clock_wise.index(self.direction)
        if np.array_equal(action,[1,0,0]):
            new_dir = clock_wise[index] #--no change
        elif np.array_equal(action,[0,1,0]):
            next_index = (index+1) % 4
            new_dir = clock_wise[next_index] #--right turn r->d->l->u
        else:
            next_index = (index - 1) % 4 #--go counter-clockwise--#
            new_dir = clock_wise[next_index] #--left turn r->u->l->d
        self.direction = new_dir
        x = self.head[0]
        y = self.head[1]
        if self.direction == Direction.right:
            x += BLOCK_SIZE
        elif self.direction == Direction.left:
            x -= BLOCK_SIZE
        elif self.direction == Direction.up:
            y += BLOCK_SIZE
        elif self.direction == Direction.down:
            y -= BLOCK_SIZE
        self.head  = Point(x,y)

#----------------------------------------------------#

