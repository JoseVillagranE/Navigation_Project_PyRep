import os
from os.path import join, dirname, abspath
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from RRT import main

from pyrep import PyRep
import pyrep.backend.sim as sim
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pioneer import Pioneer

class PioneerEnv(object):

    def __init__(self, start=[150, 150], goal=[200, 400], rand_area=[100, 450],
                margin=0.3, margin_to_goal=0.5, headless=False):

        SCENE_FILE = join(dirname(abspath(__file__)),
                          'proximity_sensor.ttt')


        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=headless)
        self.pr.start()
        self.agent = Pioneer("Pioneer_p3dx")
        self.agent.set_control_loop_enabled(False)
        self.initial_joint_positions = self.agent.get_joint_positions()

        self.vision_map = VisionSensor("VisionMap")
        self.vision_map.handle_explicitly()
        self.vision_map_handle = self.vision_map.get_handle()

        self.goal = goal
        self.margin = margin
        self.margin_to_goal = margin_to_goal
        self.start = start
        self.rand_area = rand_area

        scene_image = self.vision_map.capture_rgb()*255
        self.Planning(scene_image, self.start, self.goal, self.rand_area)

    def reset(self):

        self.pr.stop()
        self.pr.start()

    def step(self, action):

        action = np.clip(action, 0, self.agent.max_vel)
        self.agent.set_joint_target_velocities(action)
        self.pr.step() # Step the physics simulation

        scene_image = self.vision_map.capture_rgb() # numpy -> [w, h, 3]
        reward = 0

        observations = self._get_state()
        done = False
        if (observations < self.margin).any():
            done = True
        elif np.linalg.norm(self.agent.get_position[:2] - self.goal) < self.margin_to_goal:
            done = True

        return observations, reward, scene_image, done

    def _get_state(self):
        state = [proxy_sensor.read() for proxy_sensor in self.agent.proximity_sensors] # list of distances. -1 if not detect anything
        return np.array(state)

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

    @staticmethod
    def Planning(Map, start, goal, rand_area):
        """
        :parameter Map(ndarray): Image that planning over with
        """
        Map = Image.fromarray(Map.astype(np.uint8)).convert('L')
        path = main(Map, start, goal, rand_area, show_animation=True)
        np.save("PathNodes.npy", path)
