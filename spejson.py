import numpy as np
np.set_printoptions(suppress=True)

import networkx as nx
from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 10)

import copy
from udacity_stuff import extract_polygons, load_data
from shapely.geometry import Polygon, Point, LineString

import utm

def sample_points(data, polygons, n_points=100, z_range=(5,10)):
    centers = data[:, 0:3]
    sizes = data[:, 3:6]
    
    #radius_xyz = np.sqrt(np.sum(sizes**2, axis=1)).max()
    radius_xy = np.sqrt(np.sum(sizes[:,0:2]**2, axis=1)).max()

    max_x, max_y, max_z = np.max(centers + sizes, axis=0)
    min_x, min_y, min_z = np.min(centers - sizes, axis=0)

    min_z = max(z_range[0], min_z)
    max_z = min(z_range[1], max_z)

    # sample n_points
    points = np.random.uniform(
        low=[min_x, min_y, min_z],
        high=[max_x, max_y, max_z],
        size=(n_points,3)
        ).astype(np.int)

    tree = KDTree(centers[:,0:2], metric='euclidean')

    good_points = []
    for p in points:
        bad = False
        for n_idx in tree.query_radius([[p[0], p[1]]], r=radius_xy)[0]:
            if polygons[n_idx].contains(p):
                bad = True
                break
        if not bad:
            good_points.append(p)
            
    return np.array(good_points)

def collision(n1, n2, polygons):
    l = LineString([n1, n2])
    for p in polygons:
        if p.crosses(l) and p.height >= min(n1[2], n2[2]):
            return True

def create_graph(points, bf=4, return_tree=False):
    """
    points: Nx3 array of sampled points 
    bf: branching factor
    """
    
    tree = KDTree(points)
    # get nearest neighbors of each point
    dist, nn_idx = tree.query(points, bf+1, return_distance=True)

    # drop the very nearest points (it's the query point itself)
    dist, nn_idx = dist[:,1:], nn_idx[:,1:] 
    
    g = nx.Graph()

    for point, nbrs_idx, dists in zip(points, nn_idx, dist):
        for idx, d in zip(nbrs_idx, dists):
            nbr = points[idx]
            g.add_edge(tuple(point), tuple(nbr), weight=d)
    
    return (g, tree if return_tree else g)

def prune_graph(graph, obstacles, copy_graph=True):
    if copy_graph:
        graph = copy.deepcopy(graph)
        
    for (p1, p2) in list(graph.edges):
        if collision(p1, p2, polygons=obstacles):
            graph.remove_edge(p1, p2)
            
    return graph

def calculate_offset(anchor, position):

    easting_p, northing_p, _, _ = utm.from_latlon(position[1], position[0])
    easting_a, northing_a, _, _ = utm.from_latlon(anchor[1], anchor[0])

    dx = easting_p - easting_a
    dy = northing_p - northing_a

    return dx, dy
# [(30, -18, 5), (51, 32, 9), (78, 71, 9), (110, 114, 5), (129, 140, 8), (181, 155, 9), (177, 192, 6), (217, 214, 6), (223, 247, 5), (248, 274, 7), (273, 306, 8), (227, 337, 6), (185, 384, 8), (235, 439, 9), (246, 442, 8), (279, 429, 8), (318, 407, 6), (357, 405, 6), (393, 401, 5)]

def prune_path(path, polygons):
    pth = np.array(path, copy=True)
    i = 0
    l = len(path)
    idx_to_keep = list(range(l))
    while i<l-2:
        st = path[i]
        while not collision(st, path[i+2], polygons):
            idx_to_keep.remove(i+1)
            i+=1
        i+=1
    return pth[idx_to_keep]

if __name__ == '__main__':

    print("Loading data...")
    data = load_data('colliders.csv')
    obstacles = extract_polygons(data)


    print("Sampling points...")
    np.random.seed(420)
    points = sample_points(data, obstacles, 1000, (5,10))
    
    
    print("Building graph...")
    graph, kd_tree = create_graph(points, bf=7, return_tree=True)
    print("Pruning graph...")
    pruned_graph = prune_graph(graph, obstacles, copy_graph=True)
    # graph = prune_graph(graph, obstacles, copy_graph=False)


    start = (0,0,5)
    goal = (-141,255,10)

    # find nodes closest to start and goal
    (d1,d2),(s_idx,g_idx) = kd_tree.query([start, goal])

    start_node = tuple(*points[s_idx])
    goal_node = tuple(*points[g_idx])

    assert start_node in graph
    assert goal_node in graph
    assert d1 < 50
    assert d2 < 50

    path = nx.algorithms.shortest_paths.astar.astar_path(pruned_graph, start_node, goal_node)
    pruned_path = prune_path(path, obstacles)


    from planning_utils import create_grid
    grid, offset_x, offset_y = create_grid(data, 5, 2)

    plt.figure(figsize=(10,10))

    plt.imshow(grid, cmap='gray_r')

    for (n1, n2) in graph.edges:
        
        color = 'blue' if (n1, n2) in pruned_graph.edges else 'red'
        plt.plot([n1[1]-offset_y, n2[1]-offset_y],
                [n1[0]-offset_x, n2[0]-offset_x],
                color,
                linewidth=1,
                alpha=0.5)
        
    for i in range(0, len(path)-1):
        plt.plot(
            [path[i+1][1]-offset_y, path[i][1]-offset_y],
            [path[i+1][0]-offset_x, path[i][0]-offset_x],
            'green',
            linewidth=3,
            alpha=1
        )
        
    for i in range(0, len(pruned_path)-1):
        plt.plot(
            [pruned_path[i+1][1]-offset_y, pruned_path[i][1]-offset_y],
            [pruned_path[i+1][0]-offset_x, pruned_path[i][0]-offset_x],
            'yellow',
            linewidth=3,
            alpha=1
        )