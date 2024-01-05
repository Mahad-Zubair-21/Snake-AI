import torch
import random
import numpy as np
from collections import deque  #-- a data structure to store memory--#
from game import SnakeGameAI, Direction,Point
from model import Linear_QNet, QTrainer
import matplotlib.pyplot as plt
from IPython import display

MAX_MEMORY = 100_000 #--to store items in deque--#
BATCH_SIZE = 1000
LR = 0.001 #--learning rate--#

class Agent:
    def __init__(self):
        #--
        self.n_games = 0
        self.epsilon = 0 #--to control randomness--#
        self.gamma = 0.9 #--discount rate--##--must be smaller than 1--#
        self.memory = deque(maxlen= MAX_MEMORY)#--calls popleft()--#
        self.model = Linear_QNet(11,256,3) #--11 = size of state, 3 = different action--#
        self.trainer = QTrainer(self.model, lr=LR, gamma = self.gamma)
        
    def get_state(self,game):
        head = game.snake[0] #--snake's head--#
        #--to check if its a danger or it hits the boundaries--#
        point_l = Point(head[0] - 20, head[1]) #--20 is block size--#
        point_r = Point(head[0] + 20, head[1])
        point_u = Point(head[0] , head[1] - 20)
        point_d = Point(head[0] , head[1] + 20)
        #--current direction is boolean, if one is 1 others are 0--#
        dir_l = game.direction == Direction.left
        dir_r = game.direction == Direction.right
        dir_u = game.direction == Direction.up
        dir_d = game.direction == Direction.down
        #--depicts all possible states, snake can be in--#
        state = [
            #--if danger is straight ahead--#
            (dir_r and game.collision(point_r)) or
            (dir_l and game.collision(point_l)) or
            (dir_u and game.collision(point_u)) or
            (dir_d and game.collision(point_d)),
            #--if danger is on the right--#
            (dir_u and game.collision(point_r)) or
            (dir_d and game.collision(point_l)) or
            (dir_l and game.collision(point_u)) or
            (dir_r and game.collision(point_d)),
            #--if danger is on the left--#
            (dir_d and game.collision(point_r)) or
            (dir_u and game.collision(point_l)) or
            (dir_r and game.collision(point_u)) or
            (dir_l and game.collision(point_d)),
            #--Movement Direction--#
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            #--Food Location--#
            game.food[0] < game.head[0],
            game.food[0] > game.head[0],
            game.food[1] < game.head[1],
            game.food[1] > game.head[1],

        ]
        return np.array(state, dtype=int )#--helps transform into 0,1--#
        
    def remember(self,state,action,reward,next_state, done):
        #--Popleft if MAX SIZE--#
        self.memory.append((state,action,reward,next_state,done))
            
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            #--returns a list of tuples--#
            mini_sample= random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
            #-- reorganizes the data into separate tuples or lists
        states,actions,rewards,next_states,outcomes = zip(*mini_sample)   
        self.trainer.train_step(states,actions,rewards,next_states, outcomes)
        
    def train_short_memory(self,state,action,reward,next_state, outcome):
        #--train it for only one step--#
        self.trainer.train_step(state,action,reward,next_state, outcome)
        
    def get_action(self, state):
        #--random moves to explore the environment, and exploit the environment to make the best move
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0] #--available moves or actions agent can make--#
         #--if current value of epsilon is greater, the agent explores rather than exploiting
        if random.randint(0,200) < self.epsilon: 
            move = random.randint(0,2)
            final_move[move] = 1 #--updates the list with the chosen move with the corresponding index--#
        #-- the more games we have, lesser the epsilon, hence lesser random moves--#
        else:
            #--to make a move based on model--#
            #-- converts state data into PyTorch tensor of type float--#
            state0 = torch.tensor(state,dtype=torch.float)
            #--uses the neural network model to make predictions based on provided data--#
            prediction = self.model(state0)
            #--finds action with highest predicted value, and then returns index of the maximum value--#
            move = torch.argmax(prediction).item()  #-- the item() extracts index as a Python integer
            final_move[move] = 1
        return final_move
    
def plot(scores, mean_scores):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.pause(0.1)
    
def train():
    plt.ion()
    scores = []
    new_mean_scores=[]
    total_scores =0
    record =0
    agent = Agent()
    game = SnakeGameAI()
    fig = plt.figure()
    while True:
        #--get old state--#
        state_old = agent.get_state(game)
        #--get move --#
        final_move = agent.get_action(state_old)
        #--perform move and get new state--#
        reward,done,score = game.play_step(final_move)
        state_new = agent.get_state(game)
        #--train old memory--#
        agent.train_short_memory(state_old,final_move,reward,state_new,done)
        #--remember--#
        agent.remember(state_old,final_move,reward,state_new,done)
        
        if done:
            #--very important for agent, as it trains on all previous moves--#
            #--helps agent improve very much--# 
            #--train long memory--#
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()
            print("Game", agent.n_games,'Score',score, 'Record:',record)
    for i in range(100):
        scores.append(np.random.randint(1,100))
        new_mean_scores.append(np.mean(scores))
        plot(scores,new_mean_scores)
if __name__ == '__main__':
    train()