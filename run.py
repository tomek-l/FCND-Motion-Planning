import argparse
import time
import msgpack
import pickle
from enum import Enum, auto

import numpy as np
import networkx as nx

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local

from planning_utils import create_grid
from udacity_stuff import extract_polygons, load_data

from spejson import prune_graph, create_graph, sample_points, calculate_offset, prune_path, latlon2loc, loc2latlon

class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()

class MP(Drone):

    def __init__(self, connection, environment_graph, kd_tree, points, obstacles, goal_global):
        super().__init__(connection)
        
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}
        self.graph = environment_graph
        self.kd_tree = kd_tree
        self.points = points
        self.obstacles = obstacles
        self.goal_global = goal_global

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        print("position cb")
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                print(-1.0 * self.local_position[2])
                print(0.95 * self.target_position[2])
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 2.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    # if no waypoints left
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        print("velocity cb")
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        print("state cb")
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        print("arming transition")
        self.flight_state = States.ARMING
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        print("takeoff transition")
        self.flight_state = States.TAKEOFF
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        print("waypoint transition")
        self.flight_state = States.WAYPOINT
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(*self.target_position)

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        self.set_home_position(*self.global_position)
        
        self.goal_local = latlon2loc(
            self.goal_global,
            anchor_global=(self.global_home[1], self.global_home[0], self.global_home[2])
            )


        print("My goal is:")
        print(self.goal_local)
        
        print("Calculating path...")
        path = calculate_path(self.graph,
                              self.kd_tree,
                              self.points,
                              start=(0,0,0),
                              goal=self.goal_local,
                              obstacles=self.obstacles,
                              prune=False)
        
        
        print("Sending waypoints...")
        self.waypoints = path
        self.target_position = path[0]
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")
        print("starting connection")
        self.connection.start()
        self.stop_log()


def calculate_path(graph, kd_tree, points, start, obstacles, goal, prune=False):

    # find nodes closest to start and goal
    (d1,d2),(s_idx,g_idx) = kd_tree.query([start, goal])

    start_node = tuple(*points[s_idx])
    goal_node = tuple(*points[g_idx])

    try:
        print("Searching for the path...")
        path = nx.algorithms.shortest_paths.astar.astar_path(graph, start_node, goal_node)
        print("Pruning the path...")
    except nx.exception.NetworkXNoPath:
        print("No valid path found. Try with a more dense graph")
        exit(-1)
        
    if prune:
        path = prune_path(path, obstacles)

    # append path with attitude
    path = [(int(p[0]), int(p[1]), int(p[2]), 0) for p in path]
    
    return path

def dump_config(graph, kd_tree, points, obstacles):

    with open('graph.pkl', 'wb+') as f:
        pickle.dump(graph, f)

    with open('kdtree.pkl', 'wb+') as f:
        pickle.dump(kd_tree, f)

    with open('points.pkl', 'wb+') as f:
        pickle.dump(points, f)
    
    with open('obstacles.pkl', 'wb+') as f:
        pickle.dump(obstacles, f)


def load_config():
    with open('graph.pkl', 'rb') as f:
        graph = pickle.load(f)

    with open('kdtree.pkl', 'rb') as f:
        kd_tree = pickle.load(f)

    with open('points.pkl', 'rb') as f:
        points = pickle.load(f)

    with open('obstacles.pkl', 'rb') as f:
        obstacles = pickle.load(f)

    return graph, kd_tree, points, obstacles

if __name__ == "__main__":

    debug = True
    if not debug:
        print("Loading data...")
        data = load_data('colliders.csv')
        print("Loaded!")

        # get obstacles
        print("Extracting polygons...")
        obstacles = extract_polygons(data)

        # Sample points randomly 
        print("Sampling points...")
        np.random.seed(420)
        points = sample_points(data, obstacles, 1000, z_range=(10,15))

        print("Building graph...")
        # build a graph with a branching factor of 5
        graph, kd_tree = create_graph(points, bf=5, return_tree=True)

        print("Pruning graph...")
        graph = prune_graph(graph, obstacles, copy_graph=True)

        dump_config(graph, kd_tree, points, obstacles)
        
    else:
        graph, kd_tree, points, obstacles = load_config()

    conn = MavlinkConnection('tcp:{0}:{1}'.format('127.0.0.1', 5760), timeout=600)

    # upper_right = loc2latlon((400,400,10))
    upper_right = (37.79606176175873, -122.39287755584702, 10)

    drone = MP(conn, graph, kd_tree, points, obstacles, goal_global=upper_right)


    drone.start()