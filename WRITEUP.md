# 1. Explain the Starter Code

Tested ```motion_planning.py```. It is indeed similar to ```backyard_flyer_solution.py```. 

The main differences between these scripts:
- argument parser added
- ```plan_path(self)``` method, which will contain the path planning logic

A major complaint I have about ```motion_planning.py``` is that it starts inside the building and there is no way to change it. 

![https://img.youtube.com/vi/Iqj9-I4N8vE/maxresdefault.jpg](https://www.youtube.com/watch?v=Iqj9-I4N8vE)


# 2. Implementing Path Planning Algorithm

## Setting home position
- In the first step ```ANCHOR``` variable is set to ```(lat0, lon0)``` value.
- Global home is set by ```self.set_home_position(*self.global_position)``` at the begining of the planning sequence

## Local to global coordinates
Local to global coordinate mapping is calculated by:
```python
ox, oy = calculate_offset(ANCHOR, self.global_position)
```

## Setting start & goal

Start:
```python
start = self.local_position
```

Goal:
```
goal_global = (37.7952908,-122.3948162)
goal = global_to_local(goal_global, self.global_home)
```

## A* search 
Since I'm using a graph representation, the diagonal motions are enabled (or arbitrary angle/length motions for that matter).

## Path prunning (ray tracing + collinearity check)

I'm using ray tracing to determine wheteher the edges are free of obstacles.

In the image below, the blue edges correspond to the feasible vertices and the red edges to infeasible ones.
The green edges correspond to the shortest path.

![](docs/graph.png)

As an additional step, I check whether there exists an obstacle free edge between vertices of the graph (graph version of collinearity check).

The yellow edges in the image below illustrate the algorithm finding a shorter (pruned) path.

![](docs/graph_pruned.png)