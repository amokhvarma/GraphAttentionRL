import gym
from DQN import DQN
import numpy as np
from graphgen import make_graph

number_of_games = 1
Agent = DQN()
total_episodes = 0
batch_size = 32
for i in range(0,number_of_games):
    env = gym.make("MsPacman-v0")
    state = env.reset()
    done = False
    while(not done):
        old_state = state
        feat1 = make_graph(old_state,50)
        action = Agent.act(feat1)
        (next_state,reward,done,info) = env.step(action)
        feat2 = make_graph(next_state,50)
        Agent.remember(feat1,action,reward,feat2,done)
        total_episodes+=1
        if(total_episodes%batch_size == 0):
            print(total_episodes)
            print(Agent.replay())
