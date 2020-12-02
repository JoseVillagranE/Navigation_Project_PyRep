import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from env import PioneerEnv
from RRT import main

def argparser():

    parser = argparse.ArgumentParser("Navigation_project")
    parser.add_argument('--seed', help='Seed that configure torch and numpy', default=123, type=int)
    parser.add_argument('--episodes', help='Episode of simulatoin', default=5, type=int)
    parser.add_argument('--episode_length', help='caca', default=10000, type=int)

    return parser.parse_args()

def main(args):

    env = PioneerEnv()
    for e in range(args.episodes):
        print('Starting episode %d' % e)
        state = env.reset()
        for i in range(args.episode_length):
            action = env.agent.predict(state)
            next_state, reward, _, done = env.step(action)

            state = next_state


    env.shutdown()
    print("Done!!")

if __name__ == "__main__":

    args = argparser()

    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)

    main(args)
