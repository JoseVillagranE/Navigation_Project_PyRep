import numpy as np

from pyrep.robots.robot_component import RobotComponent
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from pyrep.backend import sim

from PID import PID
# from model import NavigationModel



class Pioneer(RobotComponent):
    def __init__(self, name: str, count: int = 0, base_name: str = None):

        self.pio_joint = ["Pioneer_p3dx_leftMotor",
                          "Pioneer_p3dx_rightMotor"]

        self.proximity_sensors = [ProximitySensor(f"ultrasonic_sensor#{i}") for i in range(16)]
        self.proximity_sensors_handles = [sensor.get_handle() for sensor in self.proximity_sensors]

        super().__init__(count, name, self.pio_joint, base_name)

        self.max_vel = 15 # Tune !!
        self.speed = 0.5
        self.margin_position = 0.1 # [m]
        # self.model = NavigationModel
        self.global_goal = Dummy("Goal")
        self.local_goal = Dummy("Local_goal")
        self.local_goal_idx = 0

        self.d, self.r_w = self.get_wheel_axis_radius() # wheel axis radius
        print(f"Wheel axis radius: {self.d}")
        print(f"Wheel radius: {self.r_w}")

        self.wait_reach_local_goal = False
        self.local_point_container = None

        self.dist_controller = PID(kp=0.5, ki=0, kd=0)
        self.ang_controller = PID(kp=0.5, ki=0, kd=0)

    def predict(self, state, type_planning="straight"):
        action = [0, 0]
        if type_planning=="straight":
            theta = self.get_orientation()#relative_to=self.local_goal) # [x, y, z] in radians -> [-pi pi]
            orientation = self.get_local_orientation(theta)
            distance = self.get_distance()
            action = self.take_action(orientation, distance)
        return action


    def take_action(self, orientation, distance):
        v_sp = self.dist_controller.control(distance)
        om_sp = self.ang_controller.control(orientation)
        vr, vl = self.robot_model(v_sp, om_sp)
        return [vl, vr]


        # angle_over_z = orientation # rad
        # if angle_over_z > 0 and not self.wait_reach_local_goal:
        #     return self.rotate_left()
        # elif angle_over_z < 0 and not self.wait_reach_local_goal:
        #     return self.rotate_right()
        # else:
        #     self.wait_reach_local_goal = True
        #     return self.move_forward()

    def robot_model(self, v_sp, om_sp):
        v_r = v_sp + self.d*om_sp
        v_l = v_sp - self.d*om_sp
        om_r = v_r/self.r_w
        om_l = v_l/self.r_w
        return om_r, om_l

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

    def get_distance(self):
        return np.linalg.norm(self.get_position()[:-1] - self.local_goal.get_position()[:-1])

    def get_local_orientation(self, theta):
        T = np.array([[np.cos(theta), -1*np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        trans = self.local_goal.get_position() - self.get_position()
        point_t = np.dot(T, trans)
        orientation = np.arctan2(point_t[1], point_t[0])
        return orientation

    def update_local_goal(self, debug=True):
        if self.margin_position > np.linalg.norm(self.get_position() - self.local_goal.get_position()) and \
             self.local_goal.get_position() != self.global_goal.get_position():
            self.local_goal_idx += 1
            self.local_goal.set_position(self.path[self.local_goal_idx])
            if debug:
                self.draw_local_goal()
            self.action_forward_wait = False

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
