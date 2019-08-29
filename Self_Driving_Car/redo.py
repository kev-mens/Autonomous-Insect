# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 18:45:00 2019

@author: kevin
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import random
import os

class Network(nn.Module):
    
    def __init__(self, input_nb, output_nb):
        super(Network, self).__init__()
        self.input_nb = input_nb
        self.output_nb = output_nb
        self.fc1 = nn.Linear(input_nb, 7)
        self.fc2 = nn.Linear(7, output_nb)
         
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values 
        
        
class ExperienceReplay(object):
    
    def __init__(self, capacity):
        self.capacity = capacity 
        self.memory = []
        
    def push(self, event):
        self.memory.append(event)
        if len(self.memory)> self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory,batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
class Dqn:
    
    def __init__(self, input_nb, output_nb, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.memory = ExperienceReplay(100000)
        self.model = Network(input_nb, output_nb) 
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_nb).unsqueeze(0)
        self.reward = 0
        self.last_action = 0
        
    def action_selection(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*70)
        action = probs.multinomial()
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_output = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_output + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)
        self.optimizer.step()
        
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor(self.last_reward)))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window)> 1000:
            del self.reward.window[0]
        return action
        
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1)
    
    def save(self):
        torch.save({'model' : self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict()
                    }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print('=> loading checkpoint')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['model'])
            self.model.load_state_dict(checkpoint['optimizer'])
            print('Done')
        else:
            print('Brain not found')
            

       
       
        
kev = Network(2,3)
print(kev.output_nb)
        