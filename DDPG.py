import torch
import torch.nn as nn
from torch.autograd import Variable
from networks import *
from utils import *
from replay_memory import SequentialDequeMemory, AE_ReplayB

class DDPG:

    def __init__(self,
                 state_dim,
                 action_max=[.5, 1.],
                 action_min=[0., 0.],
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 batch_size=256,
                 gamma=0.99,
                 tau=1e-2,
                 max_memory_size=100000,
                 iteration_limit=5000,
                 type_replay_buffer="random"):

        self.num_states = state_dim
        self.gamma = gamma
        self.tau = tau
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size
        self.action_max = np.array(action_max)
        self.action_min = np.array(action_min)
        self.iteration_limit = iteration_limit
        self.lambdas = [1, 1, 1]

        # GPU
        self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'

        # Networks
        self.actor = ActorNetwork(self.num_states).to(self.device)
        self.actor_target = ActorNetwork(self.num_states).to(self.device)
        self.critic = CriticNetwork(self.num_states).to(self.device)
        self.critic_target = CriticNetwork(self.num_states).to(self.device)

        # Copy weights
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param)


        # Training
        if type_replay_buffer=="random":
            self.replay_memory = SequentialDequeMemory(queue_capacity=self.max_memory_size)
        elif type_replay_buffer=="agent_expert":
            self.replay_memory = AE_ReplayB(queue_capacity=self.max_memory_size)
        else:
            raise NotImplementedError()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.mse = nn.MSELoss()

    def get_action(self, state, i):
        state = Variable(torch.from_numpy(state).float()).to(self.device)
        if state.ndim <  2:
            state = state.unsqueeze(0)
        action = self.actor(state)
        action = action.detach().cpu().numpy()[0]# + self.gen_random_noise(self.noise_decay(self.iteration_limit, i))
        action = np.clip(action, -1, 1)
        action = (self.action_max - self.action_min)*action/2.0 + (self.action_max + self.action_min)/2.0
        return action

    def update(self):
        if self.replay_memory.get_memory_size() < self.batch_size:
            return
        states, actions, rewards, next_states, done = self.replay_memory.get_random_batch_for_replay(self.batch_size)
        states = torch.from_numpy(states).to(self.device).float()
        actions = torch.from_numpy(actions).to(self.device).float()
        rewards = torch.from_numpy(rewards).to(self.device).float()
        next_states = torch.from_numpy(next_states).to(self.device).float()

        if actions.dim() < 2:
            actions = actions.unsqueeze(1)

        Qvals = self.critic(states, actions)
        next_actions = self.actor_target(next_states)
        next_Q = self.critic_target(next_states, next_actions.detach())

        Q_prime = rewards.unsqueeze(1) + self.gamma*next_Q
        critic_loss = self.mse(Qvals, Q_prime)

        actor_loss = -1*self.critic(states, self.actor(states)).mean()
        # print(f"actor loss: {actor_loss}")

        # updates networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data*self.tau + target_param.data*(1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data*self.tau + target_param.data*(1.0 - self.tau))

    def IL_update(self):

        states, actions, _, _, _ = self.replay_memory.get_random_batch_for_replay(self.batch_size)

        # normalize action
        actions = 2*(actions - self.action_min)/(self.action_max - self.action_min) - 1

        states = Variable(torch.from_numpy(states).to(self.device).float())
        actions = Variable(torch.from_numpy(actions).to(self.device).float())
        pred_actions = self.actor(states)

        # pred_actions = (self.action_max - self.action_min)*pred_actions/2.0 + (self.action_max + self.action_min)/2.0

        loss = self.mse(pred_actions, actions).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return loss.item()

    # Integrating BC and RL (Goecks, 2020)
    def CoL_update(self, pretraining_loop=False):

        if pretraining_loop:
            states, actions, rewards, next_states, dones = self.replay_memory.get_random_batch_for_replay(self.batch_size, type_of_memory="expert")
        else:
            if self.batch_size*0.75 > self.replay_memory.get_agent_memory_size():
                return
            states_e, actions_e, rewards_e, next_states_e, dones_e = self.replay_memory.get_random_batch_for_replay(round(self.batch_size*0.25), type_of_memory="expert")
            states_a, actions_a, rewards_a, next_states_a, dones_a = self.replay_memory.get_random_batch_for_replay(round(self.batch_size*0.75), type_of_memory="agent")
            states, actions, rewards, next_states, dones = self.cat_experience_tuple(states_a,
                                                                                     states_e,
                                                                                     actions_a,
                                                                                     actions_e,
                                                                                     rewards_a,
                                                                                     rewards_e,
                                                                                     next_states_a,
                                                                                     next_states_e,
                                                                                     dones_a,
                                                                                     dones_e)

        # normalize action
        actions = 2*(actions - self.action_min)/(self.action_max - self.action_min) - 1

        states = Variable(torch.from_numpy(states).to(self.device).float())
        actions = Variable(torch.from_numpy(actions).to(self.device).float())
        rewards = Variable(torch.from_numpy(rewards).to(self.device).float())
        next_states = Variable(torch.from_numpy(next_states).to(self.device).float())

        pred_actions = self.actor(states)

        # BC loss
        L_BC = self.mse(pred_actions, actions)

        # 1-step return Q-learning Loss
        R_1 = rewards + self.gamma*self.critic(next_states, self.actor(next_states).detach())
        L_Q1 = self.mse(R_1, self.critic(states, self.actor(states).detach())) # reduction -> mean

        # Actor Q_loss
        L_A = -1*self.critic(states, self.actor(states)).detach().mean()

        L_col_actor = self.lambdas[0]*L_BC + self.lambdas[2]*L_A
        L_col_critic = self.lambdas[1]*L_Q1

        self.actor_optimizer.zero_grad()
        L_col_actor.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        L_col_critic.backward()
        self.critic_optimizer.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data*self.tau + target_param.data*(1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data*self.tau + target_param.data*(1.0 - self.tau))






    def replay_add_memory(self, experience_tuple):
        self.replay_memory.add_to_memory(experience_tuple)

    def set_max_min_action(self, action_max, action_min):
        self.action_max = action_max
        self.action_min = action_min

    def set_lambdas(self, lambdas):
        self.lambdas = lambdas

    @staticmethod
    def gen_random_noise(scale=1):
        k = np.random.normal(scale=scale, size=2)
        k = np.clip(k, -scale, scale)
        return k

    @staticmethod
    def noise_decay(iteration_limit, i):
        return 0.1*(1 - (i/iteration_limit)) + 0.05

    @staticmethod
    def cat_experience_tuple(sa, se, aa, ae, ra, re, nsa, nse, da, de):
        states = np.vstack((sa, se))
        actions = np.vstack((aa, ae))
        rewards = np.vstack((ra, re))
        next_states = np.vstack((nsa, nse))
        dones = np.vstack((da, de))
        return states, actions, rewards, next_states, dones
