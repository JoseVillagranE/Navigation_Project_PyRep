import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from env import PioneerEnv
from RRT import main
from utils import get_distance


# done = True if get_distance(env.agent.get_position(), pos_m1)  < pos_margin and \
#         env.agent.get_orientation()[-1] - orientation_m1 < orientation_margin else False
# pos_m1 = env.agent.get_position()
# orientation_m1 = env.agent.get_orientation()[-1]

def argparser():

    parser = argparse.ArgumentParser("Navigation_project")
    parser.add_argument('--seed', help='Seed that configure torch and numpy', default=123, type=int)
    parser.add_argument('--episodes', help='Episode of simulation', default=1000, type=int)
    parser.add_argument('--iterations', help='caca', default=10000, type=int)
    parser.add_argument('--load_path', help = "load a path that is already calculate", default=True, type=bool)
    parser.add_argument('--type_of_planning', help="Neural Network or PID", default='PID', type=str)
    return parser.parse_args()

def main(args):
    pos_margin = 0.05
    orientation_margin = 0.01
    env = PioneerEnv(_load_path=args.load_path, type_of_planning=args.type_of_planning)
    for e in range(args.episodes):
        print('Starting episode %d' % e)
        total_reward_episode = 0
        state, sensor_state = env.reset()
        i = 0
        for i in range(args.iterations):
            action = env.agent.predict(state, sensor_state, i)
            next_state, reward, _, done, sensor_state = env.step(action)
            if args.type_of_planning=='PID':
                env.agent.update_local_goal()
            else:
                experience_tuple = (state, action, reward, next_state, done)
                env.agent.trainer.replay_add_memory(experience_tuple)
            state = next_state
            total_reward_episode += reward
            if done:
                break

        print(f"iteration: {i} || episode reward: {total_reward_episode}")
        if args.type_of_planning=='nn':
            # training
            print("training...")
            env.model_update()
    env.shutdown()
    print("Done!!")

if __name__ == "__main__":

    args = argparser()

    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)

    main(args)
