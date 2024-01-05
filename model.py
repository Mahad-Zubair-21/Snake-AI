import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os 

class Linear_QNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,output_size)
    
    def forward(self,x):
        #--apply the linear layer, also use an activation function
        x = F.relu(self.linear1(x)) #--directly from functional module--#
        x = self.linear2(x)
        return x
    
    #--helper function=--#
    def save(self, file_name='model.pth'):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self,model,lr,gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self,state,action,reward,next_state,outcome): #--can be a tuple,list or single value--#
        #--for (n,x)--#
        state = torch.tensor(state,dtype=torch.float)
        next_state = torch.tensor(next_state,dtype = torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward,dtype=torch.float)
        
        #--to handle multiple sizes--#
        if len(state.shape) == 1:
            state = torch.unsqueeze(state,0)
            next_state = torch.unsqueeze(next_state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            outcome = (outcome,)
        
        #1: get predicted Q value with current state
        pred = self.model(state)
        
        target = pred.clone()
        for index in range(len(outcome)):
            Q_new = reward[index]
            if not outcome[index]:
                Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))
            
            target[index][torch.argmax(action).item()] = Q_new
        
        #2: Q_new = r+y*max(next_predicted Q value) -> only do if not done
        # pred.clone()
        #pred[argmax(action)] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target,pred)
        loss.backward() #--apply backward propagation--#
        
        self.optimizer.step()