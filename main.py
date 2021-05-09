import gym
from DQN import DQN
import numpy as np
from graphgen import make_graph

# args
from argparse import ArgumentParser
parser = ArgumentParser()
# adding arguments
parser.add_argument('--number_of_games', dest='number_of_games', default=1, type=int,
                    help='game_num?')
parser.add_argument('--batch_size', dest='batch_size', default=32, type=int,
                    help='batch_size?')
args = parser.parse_args()

number_of_games = args.number_of_games
Agent = DQN('GAT')
total_episodes = 0
batch_size = args.batch_size
for i in range(0,number_of_games):
    env = gym.make("MsPacman-v0")
    state = env.reset()
    done = False
    temp = []
    this_game_episodes = 0
    while(not done):
        old_state = state
        feat1 = make_graph(old_state,50)
        action = Agent.act(feat1)
        (next_state,reward,done,info) = env.step(action)
        feat2 = make_graph(next_state,50)
        Agent.remember(feat1,action,reward,feat2,done)
        total_episodes+=1
        this_game_episodes+=1
        temp.append(reward)
        if(total_episodes % 100 == 0):
          arr = temp[len(temp)-100: len(temp)]
          print(sum(arr), ", episode = ", this_game_episodes)  
    print("This Game Episodes - ", this_game_episodes)      
    print("*****GAME OVER****")
        # if(total_episodes%batch_size == 0):
        #     print(total_episodes)
        #     print(Agent.replay())
