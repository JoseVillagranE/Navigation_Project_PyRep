# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:54:50 2020

@author: joser
"""

import random
from collections import deque
import numpy as np


class ExperienceReplayMemory:

    pass

class SequentialDequeMemory(ExperienceReplayMemory):

    def __init__(self, queue_capacity=2000):

        self.queue_capacity = queue_capacity
        self.memory = deque(maxlen=self.queue_capacity)

    def add_to_memory(self, experience_tuple):
        self.memory.append(experience_tuple)

    def get_random_batch_for_replay(self, batch_size=64):
        batch = random.sample(self.memory, batch_size)
        states = np.zeros((batch_size, 20))
        actions = np.zeros((batch_size, 2))
        rewards = np.zeros((batch_size, 1))
        next_states = np.zeros_like(states)
        dones = np.zeros((batch_size, 1))
        for i, experience in enumerate(batch):
            state, action, reward, next_state, done = experience
            states[i, :] = state
            actions[i, :] = action
            rewards[i, :] = reward
            next_states[i, :] = next_state
            dones[i, :] = done
        return states, actions, rewards, next_states, dones

    def get_memory_size(self):
        return len(self.memory)

class AE_ReplayB(ExperienceReplayMemory):

    def __init__(self, queue_capacity=2000):

        self.queue_capacity = queue_capacity
        self.agent_memory = deque(maxlen=self.queue_capacity)
        self.expert_memory = deque(maxlen=self.queue_capacity)

    def add_expert_memory(self, experience_tuple):
        self.expert_memory.append(experience_tuple)

    def add_agent_memory(self, experience_tuple):
        self.agent_memory.append(experience_tuple)

    def get_random_batch_for_replay(self, batch_size, type_of_memory="expert"):

        if type_of_memory=="expert":
            batch = random.sample(self.expert_memory, batch_size)
        else:
            batch = random.sample(self.agent_memory, batch_size)

        states = np.zeros((batch_size, 18))
        actions = np.zeros((batch_size, 2))
        rewards = np.zeros((batch_size, 1))
        next_states = np.zeros_like(states)
        dones = np.zeros((batch_size, 1))
        for i, experience in enumerate(batch):
            state, action, reward, next_state, done = experience
            states[i, :] = state
            actions[i, :] = action
            rewards[i, :] = reward
            next_states[i, :] = next_state
            dones[i, :] = done
        return states, actions, rewards, next_states, dones

    def get_expert_memory_size(self):
        return len(self.expert_memory)

    def get_agent_memory_size(self):
        return len(self.agent_memory)
