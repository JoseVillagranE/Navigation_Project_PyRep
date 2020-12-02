from pyrep.robots.robot_component import RobotComponent
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.backend import sim



class Pioneer(RobotComponent):
    def __init__(self, name: str, count: int = 0, base_name: str = None):

        self.pio_joint = ["Pioneer_p3dx_leftMotor",
                          "Pioneer_p3dx_rightMotor"]

        self.proximity_sensors = [ProximitySensor(f"ultrasonic_sensor#{i}") for i in range(16)]
        self.proximity_sensors_handles = [sensor.get_handle() for sensor in self.proximity_sensors]

        super().__init__(count, name, self.pio_joint, base_name)

        self.max_vel = 20 # Tune !!

    def predict(self, state, type_planning="straight"):

        if type_planning == "straight":

            


        return [self.max_vel, self.max_vel]

    def load_path(path):
        self.path = path
