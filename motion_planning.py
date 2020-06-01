import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local

from udacity_stuff import extract_polygons, load_data
from spejson import prune_graph, create_graph, sample_points, calculate_offset, prune_path


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()

def l2_dist(p1, p2):
    dx = p1[0]-p2[0]
    dy = p1[1]-p2[1]
    dz = p1[2]-p2[2]
    return np.sqrt(dx**2+dy**2+dz**2)

class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

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
                print(f'target{self.target_position[2]}')
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()
            if self.flight_state == States.LANDING:
                if l2_dist(self.local_position, self.target_position) < 0.5:
                    self.disarming_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
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
        print("planning kurwa")
        self.flight_state = States.PLANNING
        self.set_home_position(*self.global_position)
        debug = True

        if not debug:

            TARGET_ALTITUDE = 5
            SAFETY_DISTANCE = 5

            self.target_position[2] = TARGET_ALTITUDE

            print("Loading data...")
            data = load_data('colliders.csv')
            obstacles = extract_polygons(data)

            print("Sampling points...")
            np.random.seed(420)
            points = sample_points(data, obstacles, 1000, 5)

            print("Building graph...")
            graph, kd_tree = create_graph(points, bf=5, return_tree=True)

            print("Pruning graph...")
            pruned_graph = prune_graph(graph, obstacles, copy_graph=True)


            start = self.local_position
            goal = (400,400,5)

            # find nodes closest to start and goal
            (d1,d2),(s_idx,g_idx) = kd_tree.query([start, goal])

            start_node = tuple(*points[s_idx])
            goal_node = tuple(*points[g_idx])

            assert start_node in graph
            assert goal_node in graph
            assert d1 < 50 # m
            assert d2 < 50 # m

            path = nx.algorithms.shortest_paths.astar.astar_path(pruned_graph, start_node, goal_node) 
        
        # global_home = self.global_home
        # global_position = self.global_position
        # local_position = self.local_position


        path = [(4, 14, 6), (2, 24, 7), (19, 43, 5), (40, 64, 7), (51, 85, 6), (84, 88, 6), (91, 120, 6), (110, 131, 7), (131, 154, 8), (147, 173, 9), (177, 192, 6), (211, 196, 6), (231, 221, 5), (246, 251, 8), (264, 262, 7), (272, 267, 7), (288, 280, 8), (298, 294, 6), (325, 309, 9), (350, 323, 5), (356, 330, 7), (365, 343, 5), (373, 369, 6), (389, 388, 7), (395, 402, 8)]
        
        path = [
            [  30,  -18,    5],
            [ 110,  114,    5],
            [ 177,  192,    6],
            [ 156,  216,    8],
            [  89,  232,    7],
            [-134,  241,    9]
            ]
            
        # calculate offsets
        ANCHOR = (-122.397450, 37.792480)
        ox, oy = calculate_offset(ANCHOR, self.global_position)
        ox, oy = int(ox), int(oy)
        path = [(p[0]+ox, p[1]+oy, p[2], 0) for p in path]

        self.waypoints = path
        self.target_position = path[0]
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        self.stop_log()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)

    time.sleep(1)

    drone.start()
