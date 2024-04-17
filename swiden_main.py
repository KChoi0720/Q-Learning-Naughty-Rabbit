"""
Reinforcement learning black white chess board robot occupation game.

Red rectangle:          explorers(two robots).
Black rectangles:       black obstructions         [reward = -10].
yellow rectangles:      yellow obstructions        [reward = -10].
All other states:       ground                     [reward = 8/-10 (robots is encountered?)].
"""

from black_white_env import Maze
from RL_brain import QLearningTable, DeepQNetwork
import argparse
import os
import torch
import numpy as np


def make_parser():
    parser = argparse.ArgumentParser("swiden reinforcement!")
    parser.add_argument(
        "--rl", default="DQN", help="q learning type, eg. q_table or DQN"
    )
    parser.add_argument(
        "-e", "--total_episode", type=int, default=300, help="iter total episode"
    )
    return parser


def transfer(observation):
    observation = np.array(observation)
    new_observation = np.empty(4)
    new_observation[0] = (observation[0][0] + observation[0][2]) / 2
    new_observation[1] = (observation[0][1] + observation[0][3]) / 2
    new_observation[2] = (observation[1][0] + observation[1][2]) / 2
    new_observation[3] = (observation[1][1] + observation[1][3]) / 2
    return new_observation


def update():
    for episode in range(RL.total_episode):
        # initial observation
        observation = env.reset()
        episode_reward = 0
        ebs_list = []
        ebs_list.append(observation)
        # fresh env
        env.render()

        while True:
            # RL choose action based on observation
            # 0:up 1:down, [0] is robot1 and [1] is robot2
            action = RL.choose_action(str(observation), episode)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_
            # fresh env
            env.render()

            # break while loop when end of this episode
            if done:
                break

            # reward sum
            episode_reward = episode_reward + reward
            ebs_list.append(observation)

        print("episode : {} total reward: {}".format(episode, episode_reward))

    # end of game
    print('game over')
    # save dataframe
    RL.q_table.to_json('my_df.json')
    env.destroy()


def run_maze():
    # step = 0
    for episode in range(RL.total_episode):
        total_reward = 0
        observation = env.reset()
        observation = transfer(observation)
        while True:
            # print("step: {}".format(step))
            env.render()
            action = RL.choose_action(observation, episode)
            observation_, reward, done = env.step(action)
            total_reward = total_reward + reward
            if done:
                RL.learn(episode)
                break
            observation_ = transfer(observation_)
            RL.store_transition(observation, action, reward, observation_)
            # if (step > 200) and (step % 5 == 0):
            #     RL.learn()
            observation = observation_

            # step += 1
        print("episode: {}, total_reward : {}".format(episode, total_reward))
    print('game over')
    # save the model
    save_path = os.path.join('maze_dqn_' + str(episode) + '.pth')
    torch.save(RL.q_target.state_dict(), save_path)
    print('Trained model written to', save_path)
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    args = make_parser().parse_args()
    if args.rl == 'q_table':
        RL = QLearningTable(actions=list(range(env.n_actions)), total_episode=args.total_episode, is_resume=True)
        env.after(100, update)
        env.mainloop()
    else:
        RL = DeepQNetwork(env.n_actions, env.n_features,
                          learning_rate=0.01,
                          reward_decay=0.9,
                          e_greedy=0.9,
                          replace_target_iter=200,
                          memory_size=2000,
                          total_episode=args.total_episode,
                          is_resume=True
                          )
        env.after(100, run_maze)
        env.mainloop()
        RL.plot_cost()
