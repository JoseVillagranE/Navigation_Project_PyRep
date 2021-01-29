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
    parser.add_argument('--type_replay_buffer', help="Type of sample from replay buffer: random or agent_expert", default='random', type=str)
    parser.add_argument('--headless', help='mode headless simulation', default=False, type=bool)
    return parser.parse_args()

def main(args):
    pos_margin = 0.05
    orientation_margin = 0.01
    env = PioneerEnv(_load_path=args.load_path,
                     type_of_planning=args.type_of_planning,
                     headless=args.headless)
    for e in range(args.episodes):
        print('Starting episode %d' % e)
        total_reward_episode = 0
        state, sensor_state = env.reset()
        i = 0
        for i in range(args.iterations):
            action = env.agent.predict(state, sensor_state, i)
            next_state, reward, _, done, sensor_state = env.step(action)
            experience_tuple = (state, action, reward, next_state, done)
            env.agent.trainer.replay_add_memory(experience_tuple)
            state = next_state
            total_reward_episode += reward
            if done:
                break
            if args.type_of_planning=='PID':
                env.agent.update_local_goal()

        print(f"iteration: {i} || episode reward: {total_reward_episode}")
        if args.type_of_planning=='nn':
            # training
            print("training...")
            env.model_update()
    env.shutdown()
    print("Done!!")

def main_2(args):

    env = PioneerEnv(_load_path=args.load_path,
                     type_of_planning=args.type_of_planning,
                     type_replay_buffer=args.type_replay_buffer)
    state, sensor_state = env.reset()
    done = False
    total_reward_episode = 0
    linear_max_action = ang_max_action = 0
    linear_min_action = ang_min_action = 500

    pf_f_list = []
    pf_or_list = []

    # get experience w/ PID
    while not done:
        action, action_b_rm, pf_f, pf_or = env.agent.predict(state, sensor_state, 0)
        next_state, reward, _, done, sensor_state = env.step(action, pf_f)
        if action_b_rm[0] > linear_max_action: linear_max_action = action_b_rm[0]
        if action_b_rm[1] > ang_max_action: ang_max_action = action_b_rm[1]
        if action_b_rm[0] < linear_min_action: linear_min_action = action_b_rm[0]
        if action_b_rm[1] < ang_min_action: ang_min_action = action_b_rm[1]

        experience_tuple = (state, action_b_rm, reward, next_state, done)
        env.agent.trainer.replay_memory.add_expert_memory(experience_tuple)
        state = next_state
        total_reward_episode += reward
        pf_f_list.append(pf_f)
        pf_or_list.append(pf_or)
        if done:
            break
        env.agent.update_local_goal()

    print(f"max_action: [{linear_max_action}, {ang_max_action}]")
    print(f"min_action: [{linear_min_action}, {ang_min_action}]")

    # plt.figure(figsize=(12, 6))
    # plt.subplot(1,2,1)
    # plt.plot(pf_f_list)
    # plt.xlabel("timestep", fontsize=12)
    # plt.ylabel("Reactive Force", fontsize=12)
    # plt.subplot(1,2,2)
    # plt.plot(pf_or_list)
    # plt.xlabel("timestep", fontsize=12)
    # plt.ylabel("Angle correction [Rad]", fontsize=12)
    # plt.savefig("PF_graphs.png")
    # plt.show()

    # RL training
    env.agent.set_type_of_planning("nn")
    env.agent.trainer.set_max_min_action(np.array([linear_max_action, ang_max_action]),
                                        np.array([linear_min_action, ang_min_action]))
    env.set_margin(0.1)

    # expert training
    for j in range(5000):
        _ = env.model_update("CoL", pretraining_loop=True)
    env.agent.trainer.set_lambdas([1, 1, 1])
    try:
        total_reward_list = []
        for k in range(1000):
            state, sensor_state = env.reset()
            loss_mean = 0
            total_reward = 0
            done = False
            for i in range(5000):
                action, action_b_rm, pf_f, _ = env.agent.predict(state, sensor_state, k)
                next_state, reward, _, done, sensor_state = env.step(action, pf_f)
                experience_tuple = (state, action_b_rm, reward, next_state, done)
                env.agent.trainer.replay_memory.add_agent_memory(experience_tuple)
                state = next_state
                total_reward += reward
                if done:
                    break

            print(f"iteration: {k} || episode reward: {total_reward}")
            total_reward_list.append(total_reward)
            for _ in range(50):
                env.model_update("CoL")

    except KeyboardInterrupt:
        pass
    finally:

        models_state_dicts = (env.agent.trainer.actor.state_dict(),
                             env.agent.trainer.critic.state_dict(),
                             env.agent.trainer.actor_target.state_dict(),
                             env.agent.trainer.critic_target.state_dict())

        optimizers_dicts = (env.agent.trainer.actor_optimizer.state_dict(),
                            env.agent.trainer.critic_optimizer.state_dict())

        save_dict = {
                    "episode": k,
                    "models_state_dict": models_state_dicts,
                    "optimizer_dict": optimizers_dicts,
                    "total_reward_list": total_reward_list
                    }


        filename = "1s_experiment_" + str(k) + ".pth.tar"
        torch.save(save_dict, os.path.join("./weights", filename))
        env.shutdown()
        print("Done!!")


def dojo_training(args):

    env = PioneerEnv(scene_image="dojo_training.ttt",
                     _load_path=args.load_path,
                     type_of_planning=args.type_of_planning,
                     type_replay_buffer=args.type_replay_buffer)

    for task in range(10):
        state, sensor_state = env.reset()
        done = False
        total_reward_episode = 0
        linear_max_action = ang_max_action = 0
        linear_min_action = ang_min_action = 500

        pf_f_list = []
        pf_or_list = []

        # get experience w/ PID
        while not done:
            action, action_b_rm, pf_f, pf_or = env.agent.predict(state, sensor_state, 0)
            next_state, reward, _, done, sensor_state = env.step(action, pf_f)
            if action_b_rm[0] > linear_max_action: linear_max_action = action_b_rm[0]
            if action_b_rm[1] > ang_max_action: ang_max_action = action_b_rm[1]
            if action_b_rm[0] < linear_min_action: linear_min_action = action_b_rm[0]
            if action_b_rm[1] < ang_min_action: ang_min_action = action_b_rm[1]

            experience_tuple = (state, action_b_rm, reward, next_state, done)
            env.agent.trainer.replay_memory.add_expert_memory(experience_tuple)
            state = next_state
            total_reward_episode += reward
            pf_f_list.append(pf_f)
            pf_or_list.append(pf_or)
            if done:
                break
            env.agent.update_local_goal()


if __name__ == "__main__":

    args = argparser()

    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)

    main_2(args)
