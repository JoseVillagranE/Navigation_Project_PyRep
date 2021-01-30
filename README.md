# Autonomous Navigation

### Objective

The objective of the project is to implement a first project in the CoppeliaSim[1] and PyREP[2] frameworks for autonomous navigation using RRT, PID and Potential Field Planning. Along with this, Cycle-of-learning (CoL)[3] is also implemented as a learning imitation algorithm.

### Simulation environment

The simulation environment is built in coppeliaSim. This is a maze where the agent must navigate to get out of it. Some pictures are shown below:

![alt text](https://github.com/JoseVillagranE/Navigation_Project_PyRep/tree/master/images/lab.png)
![alt text](https://github.com/JoseVillagranE/Navigation_Project_PyRep/tree/master/images/lab_over.png)
![alt text](https://github.com/JoseVillagranE/Navigation_Project_PyRep/tree/master/images/pioneer.png)

### APPROACH

As a first approach to the maze solver, rrt is used as the planning algorithm, PID as the main controller and potential reactive fields for collision avoidance. In addition, the use of Cycle-of-learning is proposed for planning the robot. This solution is an alternative and is intended to improve the performance of the previous solution through a reinforced learning approach.

### Results

As a first result, an image are shown below of how rrt solves the proposed maze:

![alt text](https://github.com/JoseVillagranE/Navigation_Project_PyRep/tree/master/images/rrt_planning_map.png)

A first problem that arises with the planning of the robot consists of the proximity of the different points to the walls and that they are potential collision points. So, we must implement some correction to these points and that is the reason to use the potential field.

The intervention of the reactive field is shown below:

![alt text](https://github.com/JoseVillagranE/Navigation_Project_PyRep/tree/master/images/PF_graphs.png)

If you pay attention to the maze and the passage of time, you can recognize that the highest peak on the graph coincides with the potential collision point, so the orientation of the potential field is used to correct the error.

A short demoonstration is show in the following [link](https://drive.google.com/file/d/1VMGoldeVQoIdF3q4CmF4PHKdsQIivmui/view?usp=sharing).

TODO: Test CoL in the propose environment.

### Reference

[1] E. Rohmer, S. P. N. Singh, M. Freese, ÇoppeliaSim (formerly V-REP): a Versatile and Scalable Robot Simulation Framework", IEEE/RSJ Int. Conf. on Intelligent Robots and Systems, 2013. www.coppeliarobotics.com

[2] S. James, M. Freese, and A. J. Davison. PyRep: Bringing V-REP to deep robot learning.arXivpreprint arXiv:1906.11176, 2019.

[3] V. G. Goecks, G. M. Gremillion, V. J. Lawhern, J. Valasek, and N. R. Waytowich, “Integrating BehaviorCloning and Reinforcement Learning for Improved Performance in Dense and Sparse Reward Envi-ronments,” inProceedings of the 19th International Conference on Autonomous Agents and MultiagentSystems (AAMS’20), Richland, SC, USA, May 2020, pp. 465–473.
