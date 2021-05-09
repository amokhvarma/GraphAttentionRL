import torch
from model import GAT,CNN
import random
import numpy as np
from collections import deque
import torch.optim as  optim
import torch.nn.functional as F
from graphgen import make_graph
class DQN():
    def __init__(self,state_dim=50,num_actions=4,type = "GAT"):
        self.input_dim = state_dim
        self.hidden_dim = 128
        self.output_dim = num_actions
        self.memory = deque(maxlen=2000)
        if(type=="GAT"):
            self.model = GAT()
        elif(type=="CNN"):
            self.model = CNN()

        self.learning_rate = 0.001
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate)
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.gamma = 0.95
        self.loss = deque(maxlen=100)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def remember(self,state,action,reward,next_state,done):
        self.memory.append({'state':state,'action':action,'reward':reward,"next":next_state,'done':done})
        return

    def act(self,state):
        if(np.random.uniform(0,1) < self.epsilon):
            return random.sample([0,1,2,3],1)[0]

        self.model.eval()
        with torch.no_grad():
            act_values = self.model(state).detach().cpu().numpy()
        return np.argmax(act_values)

    def replay(self,batch_size=32):
        if(len(self.memory) < batch_size):
            return 0
        minibatch = random.sample(self.memory,batch_size)
        self.model.train()
        self.optimizer.zero_grad()
        loss = torch.tensor(0,dtype=torch.float32).to(self.device)
        for mem in minibatch:
           target = torch.tensor(mem['reward'],dtype=torch.float32).to(self.device)
           if(not mem['done']):
               with torch.no_grad():
                  target = torch.tensor(mem['reward'] + self.gamma*(np.max(self.model(mem['next']).detach().cpu().numpy())),dtype=torch.float32).to(self.device)

           pred = self.model(mem['state'])[mem['action']].to(self.device)
           loss+=F.mse_loss(pred,target)

        loss = loss/batch_size
        loss.backward()
        self.optimizer.step()
        self.loss.append(loss.item())
        if(self.epsilon > self.epsilon_min):
            self.epsilon*=self.epsilon_decay

        return loss.item()

    def save(self,path="Models/GAT"):
        torch.save(self.model,path)

    def load(self,path="Models/GAT"):
        self.model = torch.load(path)

if __name__ == "__main__":
    Agent = DQN(type="CNN")
    for i in range(1,100):
        a = np.random.randn(60,60,3)
        b = np.random.randn(60,60,3)
        t_a = make_graph(a,50)
        t_b = make_graph(b,50)
        r = np.random.uniform(0,1)
        done = False
        action = random.sample([0,1,2,3],1)[0]
        Agent.remember(t_a,action,r,t_b,done)
        if(i%40==0):
            print(Agent.replay())

    #print(Agent.model.attention[1][7],Agent.model.attention[0])