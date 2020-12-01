import math
import matplotlib.pyplot as plt
import random
import numpy as np

class RRT:


    class Node:
        def __init__(self, x, y):
                self.x = x
                self.y = y
                self.path_x = []
                self.path_y = []
                self.parent = None

    def __init__(self, Map, start, goal, rand_area,
                 expand_dis=30.0, path_resolution=5.0, goal_sample_rate=5, max_iter=500):
        """
        Setting Parameter
        Map: Gray Imagen
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        randArea:Random Sampling Area [min,max]
        """
        self.Map = Map
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = []

    def planning(self, animation=True):
        """
        rrt path planning
        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):

            print("Iteracion: " + str(i))
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, self.Map):
                self.node_list.append(new_node)
            else:
                print("Colision!")

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision(final_node, self.Map):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5:
                self.draw_graph(rnd_node)


        return None  # cannot find path

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(random.randint(self.min_rand, self.max_rand),
                            random.randint(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += int(self.path_resolution * math.cos(theta))
            new_node.y += int(self.path_resolution * math.sin(theta))
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)

        new_node.parent = from_node

        return new_node

    def draw_graph(self, rnd=None):
#        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, linestyle="-", color='g')



        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.pause(0.01)

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y)
                 ** 2 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind
    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta


    def check_collision(self, node, Map):


        if node is None:
            return False

        last_node_x = node.x
        last_node_y = node.y

#        penultimate_node_x = node.path_x[-2]
#        penultimate_node_y = node.path_y[-2]
        penultimate_node_x = node.parent.x
        penultimate_node_y = node.parent.y

        if Map[node.x, node.y] > 100:
            #Nodo en barrera
            return False

        if (len(self.node_list)==1):
            return True

#        m = (last_node_y - penultimate_node_y)/(last_node_x - penultimate_node_x + 1e-8)

#        y = penultimate_node_y
        if last_node_x > penultimate_node_x:

            if last_node_y > penultimate_node_y:
#                for x in range(penultimate_node_x, last_node_x):
#                    for y in range(penultimate_node_y, last_node_y):
                return self.ConfirmCollision(Map, penultimate_node_x, last_node_x, penultimate_node_y, last_node_y)
#
##                        print((x,y))
#                        if Map[x, y] > 100:
#                            return False
            elif last_node_y == penultimate_node_y:
                return self.ConfirmCollision(Map, penultimate_node_x, last_node_x, last_node_y, last_node_y)
#                y = last_node_y
#                for x in range(penultimate_node_x, last_node_x):
#                    if Map[x, y] > 100:
#                        return False
            else:

                return self.ConfirmCollision(Map, penultimate_node_x, last_node_x, last_node_y, penultimate_node_y)
#                for x in range(penultimate_node_x, last_node_x):
#                    for y in range(last_node_y, penultimate_node_y):
#                        if Map[x, y] > 100:
#                            return False





#                y += m*x
#
#
#                if not(y == int(y)):
#                    continue
#
#                y = int(y)
#
#                if Map[x, y] > 100: # Intersecta barrera
#                    return False

        elif last_node_x == penultimate_node_x:

#            x = last_node_x
            if last_node_y > penultimate_node_y:
                return self.ConfirmCollision(Map, last_node_x, last_node_x, penultimate_node_y, last_node_y)
#                for y in range(penultimate_node_y, last_node_y):
#
#                    if Map[x, y] > 100:
#                        return False
            else:
                return self.ConfirmCollision(Map, last_node_x, last_node_x, last_node_y, penultimate_node_y)
#                for y in range(last_node_y, penultimate_node_y):
#                    if Map[x, y] > 100:
#                        return False

        else:


            if last_node_y > penultimate_node_y:
                return self.ConfirmCollision(Map, last_node_x, penultimate_node_x, penultimate_node_y, last_node_y)
#                for x in range(last_node_x, penultimate_node_x):
#                    for y in range(penultimate_node_y, last_node_y):
#
#                        if Map[x, y] > 100:
#                            return False

            elif last_node_y == penultimate_node_y:
                return self.ConfirmCollision(Map, last_node_x, penultimate_node_x, last_node_y, last_node_y)
#                y = last_node_y
#                for x in range(last_node_x, penultimate_node_x):
#                    if Map[x, y] > 100:
#                        return False
            else:
                return self.ConfirmCollision(Map, last_node_x, penultimate_node_x, last_node_y, penultimate_node_y)
#                for x in range(last_node_x, penultimate_node_x):
#                    for y in range(last_node_y, penultimate_node_y):
#                        if Map[x, y] > 100:
#                            return False


#            for x in range(last_node_x, penultimate_node_x):
#
#
#                y+= m*x
#
#                if not(y == int(y)):
#                    continue
#
#                y = int(y)
#
#                if Map[x, y] > 100: # Intersecta barrera
#                    return False


#        for (ox, oy, size) in obstacleList:
#            dx_list = [ox - x for x in node.path_x]
#            dy_list = [oy - y for y in node.path_y]
#            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]
#
#            if min(d_list) <= size ** 2:
#                return False  # collision

#        return True  # safe

    @staticmethod
    def ConfirmCollision(Map, xmin, xmax, ymin, ymax, PixelVLimit=100):

        if np.where(Map[ymin:ymax+1, xmin:xmax+1]>PixelVLimit)[0].shape[0]!=0:
            answer = False
        else:
            answer = True

        return answer

def main(Map, start, goal, rand_area, gx=6.0, gy=10.0, show_animation=False):

    rrt = RRT(Map, start=start, goal=goal, rand_area=rand_area, max_iter=10000)
    plt.figure(figsize=((10,10)))
    plt.imshow(Map)
    path = rrt.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")
        # Draw final path
        if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], linestyle='--', marker='o', color='r')
            plt.pause(0.01)  # Need for Mac
            plt.show()

    plt.savefig("rrt_planning_map.png")
    return np.array(path)
