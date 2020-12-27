import torch
import torch.nn as nn
from torch.autograd import Variable
from networks import *
from utils import *
from replay_memory import SequentialDequeMemory

class DDPG:

    def __init__(self,
                 state_dim,
                 actor_lr=7e-5,
                 critic_lr=1e-4,
                 batch_size=124,
                 gamma=0.99,
                 tau=1e-2,
                 max_memory_size=200000):

        self.num_states = state_dim
        self.gamma = gamma
        self.tau = tau
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size

        # Networks
        self.actor = ActorNetwork(self.num_states)
        self.actor_target = Actor(self.num_states)
        self.critic = Critic(self.num_states)
        self.critic_target = Critic(self.num_states)

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

    def get_action(self, state):

        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor(state)
        action = action.detach().numpy()[0, 0]
        return action

    def update(self):
        states, actions, rewards, next_states, done = self.replay_memory.get_random_batch_for_replay(self.batch_size)
        states, actions = torch.FloatTensor(states), torch.FloatTensor(actions)
        rewards, next_states = torch.FloatTensor(rewards), torch.FloatTensor(next_states)

        if actions.dim() < 2:
            actions = actions.unsqueeze(1)

        Qvals = self.critic(states, actions)
        next_actions = self.actor_target(next_states)
        next_Q = self.critic_target(next_states, next_actions.detach())

        Q_prime = rewards.unsqueeze(1) + self.gamma*next_Q
        critic_loss = self.critic_criterion(Qvals, Q_prime)

        actor_loss = -1*self.critic(states, self.actor(states)).mean()

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
