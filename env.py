import os
from os.path import join, dirname, abspath, exists
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from RRT import main
from logger import get_logger

from pyrep import PyRep
import pyrep.backend.sim as sim
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pioneer import Pioneer
from utils import get_distance, world_to_robot_frame

class PioneerEnv(object):

    def __init__(self, start=[100, 100],
                 goal=[180, 500],
                 rand_area=[100, 450],
                 path_resolution=5.0,
                 margin=0.,
                 margin_to_goal=0.5,
                 action_max=[.5, 1.],
                 action_min=[0., 0.],
                 _load_path=True,
                 path_name="PathNodes",
                 type_of_planning="PID",
                 max_laser_range = 1.0,
                 headless=False):

        SCENE_FILE = join(dirname(abspath(__file__)),
                          'proximity_sensor.ttt')


        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=headless)
        self.pr.start()
        self.agent = Pioneer("Pioneer_p3dx", type_of_planning=type_of_planning)
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
        self.type_of_planning = type_of_planning
        self.max_laser_range = max_laser_range
        self.max_angle_orientation = np.pi
        scene_image = self.vision_map.capture_rgb()*255
        scene_image = np.flipud(scene_image)

        self.rew_weights = [1, 200, 100]

        self.start_position = Dummy("Start").get_position()
        self.goal_position = Dummy("Goal").get_position() # [x, y, z]

        self.max_distance = get_distance([-7.5, -4.1, 0.], self.goal_position)
        self.distance_to_goal_m1 = get_distance(self.start_position, self.goal_position)

        self.c_lin_vel = self.c_ang_vel = self.b_lin_vel = self.b_ang_vel = 0.

        self.action_max = action_max
        self.action_min = action_min

        if type_of_planning == 'PID':
            self.planning_info_logger = get_logger("./loggers", "Planning_Info.log")
            self.image_path = None
            if exists("./paths/"+ path_name + ".npy") and _load_path:
                print("Load Path..")
                self.image_path = np.load("./paths/"+path_name+".npy", allow_pickle=True)
            else:
                print("Planning...")
                self.image_path = self.Planning(scene_image,
                              self.start,
                              self.goal,
                              self.rand_area,
                              path_resolution=path_resolution,
                              logger=self.planning_info_logger)

            assert self.image_path is not None, "path should not be a Nonetype"

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
        if self.type_of_planning == 'PID':
            self.agent.local_goal_reset()
        self.pr.start()
        self.pr.step()
        sensor_state, distance_to_goal, orientation_to_goal = self._get_state()
        sensor_state_n, distance_to_goal, orientation_to_goal = self.normalize_states(sensor_state, distance_to_goal, orientation_to_goal)
        return np.hstack((sensor_state_n, distance_to_goal, orientation_to_goal)), sensor_state

    def step(self, action):
        self.agent.set_joint_target_velocities(action)
        self.pr.step() # Step the physics simulation
        scene_image = self.vision_map.capture_rgb()*255 # numpy -> [w, h, 3]
        sensor_state, distance_to_goal, orientation_to_goal = self._get_state()
        self.b_lin_vel, self.b_ang_vel = self.c_lin_vel, self.c_ang_vel
        self.c_lin_vel, self.c_ang_vel = self.agent.get_velocity() # 3d coordinates
        self.c_lin_vel = np.linalg.norm(self.c_lin_vel)
        self.c_ang_vel = self.c_ang_vel[-1] # w/r z
        reward, done = self._get_reward(sensor_state, distance_to_goal, orientation_to_goal)
        sensor_state_n, distance_to_goal, orientation_to_goal = self.normalize_states(sensor_state, distance_to_goal, orientation_to_goal)
        state = np.hstack((sensor_state_n, distance_to_goal, orientation_to_goal))
        return state, reward, scene_image, done, sensor_state

    def _get_state(self):
        sensor_state = np.array([proxy_sensor.read() for proxy_sensor in self.agent.proximity_sensors]) # list of distances. -1 if not detect anything
        goal_transformed = world_to_robot_frame(self.agent.get_position(), self.goal_position, self.agent.get_orientation()[-1])
        distance_to_goal = get_distance(goal_transformed[:-1], np.array([0,0])) # robot frame
        orientation_to_goal = np.arctan2(goal_transformed[1], goal_transformed[0])
        return sensor_state, distance_to_goal, orientation_to_goal

    def _get_reward(self, sensor_state, distance_to_goal, orientation_to_goal):
        done = False
        r_target = (self.c_lin_vel/self.action_max[0])*np.cos(orientation_to_goal) + 5*((distance_to_goal - self.distance_to_goal_m1) <= 0) - 6
        K = 1/(2*self.action_max[1]) * max(abs(2*self.c_ang_vel), abs(self.c_ang_vel - self.b_ang_vel))
        r_ang = -2*K*(K>0.5)
        s_st_aux = sensor_state[sensor_state>-1]
        r_danger = (60*max(s_st_aux.min()-0.35, 0) - 5)*(s_st_aux.min()<0.4) if s_st_aux.shape[0] > 0 else 0
        reward = self.rew_weights[0]*(r_target + r_ang + r_danger)
        # distance_to_goal = get_distance(agent_position[:-1], goal[:-1])
        self.distance_to_goal_m1 = distance_to_goal
        # collision check
        if self.collision_check(sensor_state, self.margin):
            reward = -1*self.rew_weights[1]
            done = True
        # goal achievement
        if get_distance(self.agent.get_position()[:-1], self.goal_position[:-1]) < self.margin_to_goal:
            print("GOAL!!")
            reward = self.rew_weights[2]
            done = True
        return reward, done

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

    def model_update(self, method="DDPG"):

        if method=="DDPG": self.agent.trainer.update()
        elif method=="IL": loss = self.agent.trainer.IL_update(); return loss
        else: raise NotImplementedError()

    def normalize_states(self, sensor_state, distance_to_goal, orientation_to_goal):
        sensor_state = self.normalize_laser(sensor_state, self.norm_func)
        distance_to_goal = self.norm_func(distance_to_goal, self.max_distance)
        orientation_to_goal = self.norm_func(orientation_to_goal, self.max_angle_orientation)
        return sensor_state, distance_to_goal, orientation_to_goal

    def normalize_laser(self, obs, norm_func):
        return np.array([norm_func(o, self.max_laser_range) if o>=0 else -1 for o in obs])

    def norm_func(self, x, max_value):
        return 2*(1 - min(x, max_value)/max_value) - 1


    def set_margin(self, margin):
        self.margin = margin

    @staticmethod
    def collision_check(observations, margin):
        if observations.sum() == -1*observations.shape[0]:
            return False
        elif observations[observations>=0].min() < margin:
            return True
        else:
            return False

    @staticmethod
    def Planning(Map, start, goal, rand_area, path_resolution=5.0, logger=None):
        """
        :parameter Map(ndarray): Image that planning over with
        """
        Map = Image.fromarray(Map.astype(np.uint8)).convert('L')
        path, n_paths = main(Map, start, goal, rand_area,
                             path_resolution=path_resolution,
                             logger=logger,
                             show_animation=False)

        if path is not None:
            np.save("./paths/PathNodes_" + str(n_paths) +".npy", path)
            return path
        else:
            logger.info("Not found Path")
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
        tras_mat[:, 0] = np.ones_like(path.shape[0])*0.3
        tras_mat[:, 1] = np.ones_like(path.shape[0])*4.65
        path = path @ rot_mat + tras_mat
        return np.flip(path, axis=0)
