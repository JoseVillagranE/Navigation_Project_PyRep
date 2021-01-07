import numpy as np

from pyrep.robots.robot_component import RobotComponent
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from pyrep.backend import sim

from PID import PID
from utils import get_distance, world_to_robot_frame
from DDPG import DDPG

class Pioneer(RobotComponent):
    def __init__(self, name: str,
                 count: int = 0,
                 base_name: str = None,
                 type_of_planning: str = "PID",
                 use_pot_field: bool = True,
                 Krep: float = 1.1):

        self.pio_joint = ["Pioneer_p3dx_leftMotor",
                          "Pioneer_p3dx_rightMotor"]

        self.proximity_sensors = [ProximitySensor(f"ultrasonic_sensor#{i}") for i in range(16)]
        self.proximity_sensors_handles = [sensor.get_handle() for sensor in self.proximity_sensors]

        super().__init__(count, name, self.pio_joint, base_name)
        self.margin_position = 0.3 # [m]
        self.type_of_planning = type_of_planning

        # self.model = NavigationModel
        self.global_goal = Dummy("Goal")
        self.local_goal = Dummy("Local_goal")
        self.local_goal_idx = 0

        self.d, self.r_w = self.get_wheel_axis_radius() # wheel axis radius
        print(f"Wheel axis radius: {self.d}")
        print(f"Wheel radius: {self.r_w}")

        if self.type_of_planning == "PID":
            self.dist_controller = PID(kp=0.1, ki=0, kd=0.)
            self.ang_controller = PID(kp=0.2, ki=0., kd=0.1)
        elif self.type_of_planning == "nn":
            self.trainer = DDPG(len(self.proximity_sensors)+1+1)
        else:
            NotImplementedError()

        self.use_pot_field = use_pot_field
        self.Krep = Krep

    def predict(self, state, sensor_state, i):
        action = [0, 0]
        if self.type_of_planning=="PID":
            # orientation w/r to world frame
            theta = self.get_orientation()[-1]# [x, y, z] -> z in radians -> [-pi pi]
            point_t = self.local_goal_to_robot_frame(theta)
            distance = get_distance(point_t, np.array([0, 0]))
            orientation = np.arctan2(point_t[1], point_t[0])
            try:
                if self.use_pot_field and sensor_state[sensor_state>-1].min() < 0.15:
                    pf_or = self.get_or_pf(sensor_state, orientation)
                    orientation -= pf_or
            except ValueError:
                pass
            action = self.take_action(orientation, distance)
        elif self.type_of_planning=="nn":
            action = self.trainer.get_action(state, i)
            wl, wr = self.robot_model(*action)
            action = np.array([wl, wr])
        return action


    def take_action(self, orientation, distance):
        v_sp = self.dist_controller.control(distance)
        om_sp = self.ang_controller.control(orientation)
        wr, wl = self.robot_model(v_sp, om_sp)
        return [wl, wr]

    def robot_model(self, v_sp, om_sp):
        om_r = (v_sp + self.d*om_sp)/self.r_w
        om_l = (v_sp - self.d*om_sp)/self.r_w
        return om_r, om_l

    def local_goal_reset(self):
        self.local_goal_idx = 0
        self.local_goal.set_position(self.path[0])

    def local_goal_to_robot_frame(self, theta):
        return world_to_robot_frame(self.get_position(), self.local_goal.get_position(), theta)

    def update_local_goal(self, debug=True):
        if self.margin_position > np.linalg.norm(self.get_position()[:-1] - self.local_goal.get_position()[:-1]):
            self.local_goal_idx += 1
            if self.local_goal_idx == self.path.shape[0]:
                return
            self.local_goal.set_position(self.path[self.local_goal_idx])
            if debug:
                self.draw_local_goal()


    def get_or_pf(self, states, orientation):
        gamma = 2*np.pi/states.shape[0]
        angles = np.linspace(0, 2*np.pi - gamma, states.shape[0])
        mask = states > -1
        states = states[mask]
        angles = angles[mask]
        F = (np.cos(angles)/states**2).sum()
        S =  (np.sin(angles)/states**2).sum()
        Xrep = F*np.cos(orientation) - S*np.sin(orientation)
        Yrep = F*np.sin(orientation) + S*np.cos(orientation)
        Px = -Xrep
        Py = -Yrep
        or_pf = np.arctan2(Py, Px)
        return self.Krep*or_pf


    def load_path(self, path, debug=True):
        self.path = path[1:]
        self.start_position = path[0]
        # self.global_goal = path[-1]
        self.local_goal.set_position(self.path[0])
        if debug:
            self.draw_local_goal()
        print(f"local_goal_position: {self.local_goal.get_position()}")

    def load_point_container(self, point_container):
        self.local_point_container = point_container


    def draw_local_goal(self):
        x, y, z = self.local_goal.get_position()
        point_data = (x, y, 0)
        sim.simAddDrawingObjectItem(self.local_point_container, point_data)

    @staticmethod
    def get_wheel_axis_radius():
        left_wheel = Shape("Pioneer_p3dx_leftWheel")
        right_wheel = Shape("Pioneer_p3dx_rightWheel")
        distance = np.linalg.norm(left_wheel.get_position() - right_wheel.get_position())
        bbx_list = left_wheel.get_bounding_box()
        radius = (bbx_list[-1] - bbx_list[-2])/2.0
        return distance, radius
