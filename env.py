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
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pioneer import Pioneer

class PioneerEnv(object):

    def __init__(self, start=[100, 100], goal=[180, 500], rand_area=[100, 450],
                path_resolution=5.0, margin=0.1, margin_to_goal=0.5, headless=False):

        SCENE_FILE = join(dirname(abspath(__file__)),
                          'proximity_sensor.ttt')


        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=headless)
        self.pr.start()
        self.agent = Pioneer("Pioneer_p3dx")
        self.agent.set_control_loop_enabled(False)
        self.initial_joint_positions = self.agent.get_joint_positions()
        self.initial_position = self.agent.get_position()
        print(f"Agent initial position: {self.initial_position}")

        self.vision_map = VisionSensor("VisionMap")
        self.vision_map.handle_explicitly()
        self.vision_map_handle = self.vision_map.get_handle()

        self.floor = Shape("Floor_respondable")

        self.perspective_angle = self.vision_map.get_perspective_angle # degrees
        self.resolution = self.vision_map.get_resolution # list [resx, resy]
        self.margin = margin
        self.margin_to_goal = margin_to_goal
        self.rand_area = rand_area
        self.start = start
        self.goal = goal
        scene_image = self.vision_map.capture_rgb()*255
        scene_image = np.flipud(scene_image)

        self.image_path = None
        if exists("./paths/PathNodes.npy"):
            print("Load Path..")
            self.image_path = np.load("./paths/PathNodes.npy", allow_pickle=True)
        else:
            print("Planning...")
            self.image_path = self.Planning(scene_image,
                          self.start,
                          self.goal,
                          self.rand_area,
                          path_resolution=path_resolution)

        assert self.image_path is not None, "path should not be a Nonetype"

        self.start_position = Dummy("Start").get_position()
        self.goal_position = Dummy("Goal").get_position() # [x, y, z]

        self.real_path = self.path_image2real(self.image_path,
                                        self.start_position)
        # project in coppelia sim
        sim_drawing_points = 0
        point_size = 10 #[pix]
        duplicate_tolerance = 0
        parent_obj_handle = self.floor.get_handle()
        max_iter_count = 999999
        ambient_diffuse = (255, 0, 0)
        blue_ambient_diffuse = (0, 0, 255)
        point_container = sim.simAddDrawingObject(sim_drawing_points,
                                                 point_size,
                                                 duplicate_tolerance,
                                                 parent_obj_handle,
                                                 max_iter_count,
                                                 ambient_diffuse=ambient_diffuse)

        local_point_container = sim.simAddDrawingObject(sim_drawing_points,
                                                        point_size,
                                                        duplicate_tolerance,
                                                        parent_obj_handle,
                                                        max_iter_count,
                                                        ambient_diffuse=blue_ambient_diffuse)

        # debug
        for point in self.real_path:
            point_data = (point[0], point[1], 0)
            sim.simAddDrawingObjectItem(point_container, point_data)
        # You need to get the real coord in the real world

        assert local_point_container is not None, "point container shouldn't be empty"
        self.agent.load_point_container(local_point_container)
        self.agent.load_path(self.real_path)

    def reset(self):

        self.pr.stop()
        self.agent.reset()
        self.pr.start()
        self.pr.step()
        return self._get_state()

    def step(self, action):

        action = np.clip(action, 0, self.agent.max_vel)
        self.agent.set_joint_target_velocities(action)
        self.pr.step() # Step the physics simulation

        scene_image = self.vision_map.capture_rgb()*255 # numpy -> [w, h, 3]
        reward = 0

        observations = self._get_state()
        done = False
        # collision check
        if observations[observations > 0].min() < self.margin:
            done = True
        # goal achievement
        elif np.linalg.norm(self.agent.get_position()[:2] - self.goal) < self.margin_to_goal:
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
        if path is not None:
            np.save("./paths/PathNodes.npy", path)
            return path
        else:
            return None

    @staticmethod
    def path_image2real(image_path, start):
        """
        image_path: np.array[pix] points of the image path
        start_array: np.array
        """
        scale = 13.0/512 # [m/pix]

        x_init = start[0]
        y_init = start[1]
        deltas = [(image_path[i+1][0] - image_path[i][0], image_path[i+1][1] - image_path[i][1]) for i in range(image_path.shape[0] - 1)]

        path = np.zeros((image_path.shape[0], 3))
        path[0, :] = np.array([x_init, y_init, 0])
        for i in range(1, image_path.shape[0]):
            path[i, :] = np.array([path[i-1, 0] + deltas[i-1][0]*scale, path[i-1, 1] - deltas[i-1][1]*scale, 0])

        rot_mat = np.diagflat([-1, -1, 1])
        tras_mat = np.zeros_like(path)
        tras_mat[:, 1] = np.ones_like(path.shape[0])*4.65
        path = path @ rot_mat + tras_mat
        return np.flip(path, axis=0)
