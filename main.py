import gym
from DQN import DQN
import numpy as np
from graphgen import make_graph
import matplotlib.pyplot as plt

# args
from argparse import ArgumentParser
parser = ArgumentParser()
# adding arguments
parser.add_argument('--number_of_games', dest='number_of_games', default=10, type=int,
                    help='game_num?')
parser.add_argument('--batch_size', dest='batch_size', default=32, type=int,
                    help='batch_size?')
args = parser.parse_args()

number_of_games = args.number_of_games
Agent = DQN('GAT')
all_episodes = []
all_rewards = []
batch_size = args.batch_size
for i in range(0,number_of_games):
    env = gym.make("MsPacman-v0")
    state = env.reset()
    done = False
    this_game_episodes = 0
    reward_arr = []
    while(not done):
        old_state = state
        feat1 = make_graph(old_state,50)
        action = Agent.act(feat1)
        (next_state,reward,done,info) = env.step(action)
        feat2 = make_graph(next_state,50)
        Agent.remember(feat1,action,reward,feat2,done)
        this_game_episodes+=1
        reward_arr.append(reward)
    print("Game number - ", i+1, " This Game Episodes - ", this_game_episodes, " This game Average reward - ", sum(reward_arr)/ this_game_episodes)
    all_rewards.append(sum(reward_arr)/ this_game_episodes)
    all_episodes.append(this_game_episodes)
    if(i != 0 and i % 5 == 0):
      y_1 = all_rewards
      y_2 = all_episodes

      # plotting reward evolution
      plt.plot(y_1)
      plt.xlabel('Number of games')
      plt.ylabel('Average reward per episode')
      plt.title('Reward Evolution')
      plt.savefig("Reward Evolution.png")

      # plotting num_episode evolution
      plt.plot(y_2)
      plt.xlabel('Number of games')
      plt.ylabel('Number of episodes')
      plt.title('Episode_count Evolution')
      plt.savefig("Episode_count Evolution.png")