"""
grid.py - Grid Map Representation
Defines the environment: safe zones, no-fly zones, obstacles
"""

import numpy as np
import random

# Cell types
FREE       = 0   # Safe to fly
OBSTACLE   = 1   # Physical obstacle (building, mountain, tree)
NO_FLY     = 2   # Restricted / no-fly zone
START      = 3   # Drone start position
VISITED    = 4   # Already covered by drone

class Grid:
    def __init__(self, rows=20, cols=20):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)
        self.start = (0, 0)
        self.dynamic_obstacles = []  # Discovered mid-flight

    def set_start(self, row, col):
        self.start = (row, col)
        self.grid[row][col] = START

    def add_obstacle(self, row, col):
        if self.grid[row][col] == FREE:
            self.grid[row][col] = OBSTACLE

    def add_no_fly_zone(self, row_start, row_end, col_start, col_end):
        for r in range(row_start, row_end):
            for c in range(col_start, col_end):
                if self.grid[r][c] == FREE:
                    self.grid[r][c] = NO_FLY

    def is_free(self, row, col):
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        return self.grid[row][col] in [FREE, START, VISITED]

    def mark_visited(self, row, col):
        if self.grid[row][col] == FREE:
            self.grid[row][col] = VISITED

    def add_dynamic_obstacle(self, row, col):
        """Obstacle discovered mid-flight (not on initial map)"""
        if self.grid[row][col] == FREE:
            self.grid[row][col] = OBSTACLE
            self.dynamic_obstacles.append((row, col))
            return True
        return False

    def coverage_percentage(self):
        total_free = np.sum(self.grid == FREE) + np.sum(self.grid == VISITED) + np.sum(self.grid == START)
        visited = np.sum(self.grid == VISITED) + np.sum(self.grid == START)
        if total_free == 0:
            return 0
        return round((visited / total_free) * 100, 2)

    def print_grid(self):
        symbols = {FREE: '.', OBSTACLE: 'X', NO_FLY: 'N', START: 'S', VISITED: '*'}
        print("\n  " + " ".join(str(i % 10) for i in range(self.cols)))
        for r in range(self.rows):
            row_str = " ".join(symbols[self.grid[r][c]] for c in range(self.cols))
            print(f"{r:2d} {row_str}")
        print()


def create_sample_map(rows=20, cols=20):
    """Creates a realistic sample surveillance map"""
    g = Grid(rows, cols)
    g.set_start(0, 0)

    # Add no-fly zones (restricted areas)
    g.add_no_fly_zone(5, 9, 10, 14)   # Zone A
    g.add_no_fly_zone(14, 18, 3, 7)   # Zone B

    # Add scattered obstacles (buildings, trees)
    obstacles = [
        (2,5),(2,6),(3,5),(3,6),         # Building cluster 1
        (8,2),(8,3),(9,2),(9,3),         # Building cluster 2
        (12,15),(13,15),(12,16),         # Mountain
        (1,18),(2,18),(3,18),(4,18),     # Wall-like structure
        (17,12),(17,13),(18,12),         # Far obstacle
        (6,18),(7,18),(7,17),            # Corner obstacle
    ]
    for r, c in obstacles:
        g.add_obstacle(r, c)

    return g
