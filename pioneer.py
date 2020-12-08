import numpy as np

from pyrep.robots.robot_component import RobotComponent
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.dummy import Dummy
from pyrep.backend import sim
# from model import NavigationModel



class Pioneer(RobotComponent):
    def __init__(self, name: str, count: int = 0, base_name: str = None):

        self.pio_joint = ["Pioneer_p3dx_leftMotor",
                          "Pioneer_p3dx_rightMotor"]

        self.proximity_sensors = [ProximitySensor(f"ultrasonic_sensor#{i}") for i in range(16)]
        self.proximity_sensors_handles = [sensor.get_handle() for sensor in self.proximity_sensors]

        super().__init__(count, name, self.pio_joint, base_name)

        self.max_vel = 15 # Tune !!
        self.speed = 5
        self.margin_position = 1 # [m]
        # self.model = NavigationModel
        self.global_goal = Dummy("Goal")
        self.local_goal = Dummy("Local_goal")
        self.local_goal_idx = 0

    def predict(self, state, type_planning="straight"):
        action = [0, 0]
        if type_planning=="straight":
            orientation = self.get_orientation(relative_to=self.local_goal) # [x, y, z] in radians
            print(orientation)
            action = self.take_action(orientation)
        return action

    def take_action(self, orientation):
        angle_over_z = orientation[-1] # rad
        if angle_over_z > 0:
            return self.rotate_left()
        elif angle_over_z < 0:
            return self.rotate_right()
        else:
            return self.move_forward()

    def move_forward(self):
        return [self.speed, self.speed]

    def rotate_right(self):
        return [self.speed, -self.speed]

    def rotate_left(self):
        return [-self.speed, self.speed]

    def move_backward(self):
        return [-self.speed, -self.speed]

    def reset(self):
        self.local_goal_idx = 0
        self.local_goal.set_position(self.path[0])

    def update_local_goal(self):
        if self.margin_position > np.linalg.norm(self.get_position() - self.local_goal.get_position()) and \
             self.local_goal.get_position() != self.global_goal.get_position():
            self.local_goal_idx += 1
            self.local_goal.set_position(self.path[self.local_goal_idx])

    def load_path(self, path):
        self.path = path[1:]
        self.start_position = path[0]
        self.global_goal = path[-1]
        self.local_goal.set_position(self.path[0])
        print(f"local_goal_position: {self.local_goal.get_position()}")
