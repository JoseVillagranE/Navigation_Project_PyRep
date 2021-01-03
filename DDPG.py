import torch
import torch.nn as nn
from torch.autograd import Variable
from networks import *
from utils import *
from replay_memory import SequentialDequeMemory

class DDPG:

    def __init__(self,
                 state_dim,
                 action_max=1200,
                 action_min=0,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 batch_size=256,
                 gamma=0.99,
                 tau=1e-2,
                 max_memory_size=100000,
                 iteration_limit=500000):

        self.num_states = state_dim
        self.gamma = gamma
        self.tau = tau
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size
        self.action_max = action_max
        self.action_min = action_min
        self.iteration_limit = iteration_limit

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
        self.replay_memory = SequentialDequeMemory(queue_capacity=self.max_memory_size)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_criterion = nn.MSELoss()

    def get_action(self, state, i):
        state = Variable(torch.from_numpy(state).float()).to(self.device)
        if state.ndim <  2:
            state = state.unsqueeze(0)
        action = self.actor(state)
        action = action.detach().cpu().numpy()[0] + self.gen_random_noise(self.noise_decay(self.iteration_limit, i))
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
        critic_loss = self.critic_criterion(Qvals, Q_prime)

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

    def replay_add_memory(self, experience_tuple):
        self.replay_memory.add_to_memory(experience_tuple)

    @staticmethod
    def gen_random_noise(scale=1):
        k = np.random.normal(scale=scale, size=2)
        k = np.clip(k, -scale, scale)
        return k

    @staticmethod
    def noise_decay(iteration_limit, i):
        return 0.95*(1 - (i/iteration_limit)) + 0.05
