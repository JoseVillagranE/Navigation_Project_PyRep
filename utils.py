import numpy as np


def get_distance(point_1, point_2):
    return np.linalg.norm(point_2 - point_1)

def world_to_robot_frame(point_1, point_2, theta):
    T = get_rotation_matrix(theta)# left handed coord_system
    trans = point_2 - point_1
    point_t = np.dot(T, trans[:-1].T)
    return point_t

def get_rotation_matrix(theta, coord_system="left"):
    T = np.array([[np.cos(theta), -1*np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    if coord_system == "right":
        return T
    elif coord_system == "left":
        return T.T
