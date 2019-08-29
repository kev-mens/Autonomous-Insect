import random # used for extracting random samples during experience replay
import os #allows the trained brain to be saved
import torch #since we'll be using pytorch to implement the ai
import torch.nn as nn #contains all the tools to implement a neural network
import torch.nn.functional as F # includes the loss function 
import torch.optim  as optim #needed for stochastic gradient descent
from torch.autograd import Variable #used to convert tensors into variables with gradients
#we convert to variables because it allows us to perform more operations in them
                                         
class Network(nn.Module): #inherits the tools in the module class
    
    def __init__(self, input_size, nb_action): # number of inputs/actions(outputs)
        super(Network, self).__init__() #super allows us to use the functions from nn.module        
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 8) 
        # creates a connection between the input and each hidden layer
        self.fc2 = nn.Linear(8, nb_action)
        # connects each hidden layer to the action(outputs)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))  # from input to hidden neuron
        # F.relu is the rectifier function we're using
        # an activation function is important to represent more complicated models
        # removes linearity
        # x represents the values within the hidden layer
        q_values = self.fc2(x)
        return q_values

# Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity): 
        # capacity is the number of past experiences we want
        self.capacity = capacity
        self.memory = []
        # our experiences are appended to the list for storage 
    
    def push(self, event):
        #ensures that the number of elements in memory are within the capacity
        self.memory.append(event)
        #an event includes input from the three sensors, +/- orientation
        if len(self.memory) > self.capacity:
            del self.memory[0]
        #self.memory is a list of lists
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        #samples the memory according to the batch_size value
        #the zip(*) function takes a list of lists and unpacks it according to index
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
        #map makes a mapping of the sample to pytorch var containing a tensor and a gradient
        #torch.cat aligns the batches, concatenates them to the first dimension

# Implementing Deep Q Learning

class Dqn:
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma #our discounting factor
        self.reward_window = [] 
        # this takes the mean of the last 100 rewards
        #just for us track the rewards
        self.model = Network(input_size, nb_action)
        #uses the earlier defined class to create a network architecture
        self.memory = ReplayMemory(100000)
        #uses the earlier defined class to create a memory
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        #we use the Adam optimizer for stochastic gradient descent
        #works by changing weights to minimize cost function
        #the optimizer accesses the parameters of our model
        #lr or living penalty is an input 
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        #last_state needs to be a torch tensor, torch.Tensor
        #unsqueeze encapsulates a list into other list
        #unsqueeze is used because the network only accepts a list, not a list element
        self.last_action = 0
        #last action could either to 0,1,2 which corresponds to a tilt 0,20,-20 degrees
        self.last_reward = 0
        #the reward is a float between -1 and 1
                
    def select_action(self,state):
        #the action we'll take depends on the current input state
        probs = F.softmax(self.model(Variable(state, volatile = True))*20)
        #assigns a probablity to each q-value in model using a softmax function 
        #we are able to access the q-values because the forward module is inherited 
        #therefore it is accessed automatically just by passing state as an input
        #refer to the network class
        #we use volatile = true to exlude the gradients associated with the variables to save memory
        #7 is the temperature parameter 
        #the higher the temp parameter, the more sure the agent is of taking an action
        #eg. softmax([1,2,3]) = [0.04, 0.11, 0.85] => softmax([1,2,3]*3) = [0, 0.02, 0.98]
        #what if we had 0.5 probablility initially?
        action = probs.multinomial()
        #randomly chooses an action with respect to their probabilities
        return action.data[(0,0)]
        #w want to return the action taken which is 0,1 or 2
        #we use 0,0 because multinomial returns a pytorch variable which acts like a matrix
        #therefore we need to access the actual action within the batch by using index 0,0

    def learn(self, batch_state, batch_next_state, batch_action, batch_reward):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        #collects the chosen actions for each of the input states within batch state 
        #batch_action is unsqueezed to make it have the same dimension as batch_state
        #we finally squeeze the output because we dont need it to be a batch
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        #we need the next output because the target(y hat) is the (next output*gamma) + reward
        #the next output is the highest q-value across all actions
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        #smooth_l1_loss is the huber function (loss function)
        #we use this for temporal difference
        self.optimizer.zero_grad()
        #re-initializes the optimizer for each iteration
        td_loss.backward(retain_variables = True)
        #backpropagation
        #we use retain variables = True to free some memory
        self.optimizer.step()
        #updates the weights
        
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        #because the new_state depends on the signals received
        #We convert it to a torch tensor because that's what the neural network accepts
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        #self is only included in a method when we use a variable declared within the init function  
        #it seems to have more than one arguement, but remember event is a tuple of st, st+1, last action, and the last rewards
        #we use LongTensor to convert the integer representing the action into a tensor
        #our transitions should only include tensors
        #square brackets are part of the syntax for creating tensors
        action = self.select_action(new_state)
        #selects a new action based on the new state
        if len(self.memory.memory)> 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_action, batch_reward)
            #learning begins when we have more than 100 events in memory
        self.last_action = action
        #we've taken a new action therefore the last state becomes the new state variable above
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
            #our reward windows is a sliding window, max of 1000
        return action
        
    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1)
    #returns the mean of the reward window
    #+1 in the denominator to prevent zero division in case the reward window is empty
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(), 
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    #we save the parameters of the model and the optimizer(the weights)
    #the second argument is the name of the save file
    #try using brackets next to the optimizer save later
    
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            #checks for the existence of the save file in the working directory
            print('=> loading checkpoint...')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict']) 
            #loads data from the saved NN 
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            #loads data from the saved optimizer
            print('done!')
        else:
            print('no checkpoint found...')    