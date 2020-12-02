import os
from os.path import join, dirname, abspath, exists
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

    def __init__(self, start=[161, 361], goal=[208, 102], rand_area=[100, 450],
                path_resolution=5.0, margin=0.3, margin_to_goal=0.5, headless=False):

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

        path = None
        if exists("./paths/PathNodes.npy"):
            path = np.load("paths/PathNodes.npy")
        else:
            path = self.Planning(scene_image,
                          self.start,
                          self.goal,
                          self.rand_area,
                          path_resolution=path_resolution)

        assert path is not None, "path should not be a Nonetype"

        self.path = path[1:] # pull out the initial node
        self.agent.load_path(path)

    def reset(self):

        self.pr.stop()
        self.pr.start()
        return self._get_state()

    def step(self, action):

        action = np.clip(action, 0, self.agent.max_vel)
        self.agent.set_joint_target_velocities(action)
        self.pr.step() # Step the physics simulation

        scene_image = self.vision_map.capture_rgb()*255 # numpy -> [w, h, 3]
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
    def Planning(Map, start, goal, rand_area, path_resolution=5.0):
        """
        :parameter Map(ndarray): Image that planning over with
        """
        Map = Image.fromarray(Map.astype(np.uint8)).convert('L')
        path = main(Map, start, goal, rand_area, path_resolution=path_resolution, show_animation=False)
        np.save("PathNodes.npy", path)
        return path
