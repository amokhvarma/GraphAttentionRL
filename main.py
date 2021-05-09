import gym
from DQN import DQN
import numpy as np
from graphgen import make_graph
import matplotlib.pyplot as plt
import os
# args
from argparse import ArgumentParser
parser = ArgumentParser()
# adding arguments
parser.add_argument('--number_of_games', dest='number_of_games', default=3, type=int,help="game_num")
parser.add_argument('--batch_size', dest='batch_size', default=32, type=int,
                    help='batch_size?')
parser.add_argument('--save_path',dest='save_path',default="Models/GAT",type=str,help="Save Path")
parser.add_argument("--replay",dest='replay',default=10,type=int,help="Save frequency")

args = parser.parse_args()
if(not os.path.isdir("Results")):
    os.mkdir("Results")
if(not os.path.isdir("Models")):
    os.mkdir("Models")

number_of_games = args.number_of_games
Agent = DQN(type='CNN')
all_episodes = []
all_rewards = []
batch_size = args.batch_size
total_episode = 0
loss_total = []
y_1,y_2,y_3=[],[],[]
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
        #print(reward)
        if(reward==10):
            reward = 0.3
        elif(reward==0 and not done):
            reward = 0.1
        else:
            reward = 0
        feat2 = make_graph(next_state,50)
        Agent.remember(feat1,action,reward,feat2,done)
        this_game_episodes+=1
        reward_arr.append(reward)
        total_episode+=1
        if(total_episode % batch_size == 0):
            loss = Agent.replay(batch_size)
            loss_total.append(loss)



    print("Game number - ", i+1, " This Game Episodes - ", this_game_episodes, "  Gamewise Average reward - ", sum(reward_arr)/ this_game_episodes)
    all_rewards.append(sum(reward_arr[-100:]))
    all_episodes.append(this_game_episodes)
    y_1.append(np.mean(all_rewards))
    y_2.append(np.mean(all_episodes))
    y_3.append(np.mean(loss_total))
    if(i != 0 and i % args.replay == 0):
        # y_1 = all_rewards
        # y_2 = all_episodes
        # y_3 = loss_total
        # plotting reward evolution
        plt.plot(y_1)
        plt.xlabel('Number of games')
        plt.ylabel('Average reward per episode')
        plt.title('Reward Evolution')
        plt.savefig("Results/Reward Evolution.png")
        plt.close()

        # plotting num_episode evolution
        plt.plot(y_2)
        plt.xlabel('Number of games')
        plt.ylabel('Number of episodes')
        plt.title('Episode_count Evolution')
        plt.savefig("Results/Episode_count Evolution.png")

        plt.close()

        # plotting loss change
        plt.plot(y_3)
        plt.xlabel('Number of episodes')
        plt.ylabel('Loss')
        plt.title('Loss Evolution')
        plt.savefig("Results/Loss Evolution.png")

        plt.close()
        Agent.save(args.save_path)


