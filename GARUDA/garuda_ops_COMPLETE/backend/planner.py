"""
planner.py - Path Planning Algorithm
Stage 2: Boustrophedon Coverage Path + A* with energy cost function
"""

import heapq
from grid import Grid, FREE, OBSTACLE, NO_FLY, START, VISITED

# Movement directions: Up, Down, Left, Right, and Diagonals
STRAIGHT_MOVES = [(-1,0),(1,0),(0,-1),(0,1)]
ALL_MOVES      = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

# Energy costs
COST_STRAIGHT  = 1.0   # Moving in a straight line
COST_TURN      = 1.4   # Changing direction (more battery)
COST_DIAGONAL  = 1.6   # Diagonal move
COST_PROXIMITY = 5.0   # Penalty for being near no-fly zone

def energy_cost(prev_dir, new_dir, is_diagonal=False):
    """Calculate energy cost of a move based on direction change"""
    if is_diagonal:
        return COST_DIAGONAL
    if prev_dir is None or prev_dir == new_dir:
        return COST_STRAIGHT
    return COST_TURN

def proximity_penalty(grid, row, col):
    """Penalize cells that are adjacent to no-fly zones or obstacles"""
    penalty = 0
    for dr, dc in ALL_MOVES:
        nr, nc = row + dr, col + dc
        if 0 <= nr < grid.rows and 0 <= nc < grid.cols:
            if grid.grid[nr][nc] == NO_FLY:
                penalty += COST_PROXIMITY
    return penalty


# ─────────────────────────────────────────────────────────────
# STAGE 2A: BOUSTROPHEDON PATH (Lawnmower Pattern)
# ─────────────────────────────────────────────────────────────

def boustrophedon_path(grid: Grid):
    """
    Generates a back-and-forth (lawnmower) sweep path over the grid.
    Skips obstacles and no-fly zones.
    Returns ordered list of (row, col) waypoints.
    """
    path = []
    for r in range(grid.rows):
        if r % 2 == 0:
            cols = range(grid.cols)        # Left to right
        else:
            cols = range(grid.cols - 1, -1, -1)  # Right to left

        for c in cols:
            if grid.is_free(r, c):
                path.append((r, c))

    return path


# ─────────────────────────────────────────────────────────────
# STAGE 2B: A* WITH ENERGY COST FUNCTION
# ─────────────────────────────────────────────────────────────

def heuristic(a, b):
    """Manhattan distance heuristic"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid: Grid, start, goal, prev_direction=None):
    """
    A* pathfinding with custom energy cost.
    Returns (path as list of (row,col), total_energy_cost)
    """
    open_set = []
    heapq.heappush(open_set, (0, start, prev_direction))

    came_from = {}
    g_score = {start: 0}
    direction_map = {start: prev_direction}

    while open_set:
        current_f, current, cur_dir = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, g_score[goal]

        r, c = current
        for dr, dc in STRAIGHT_MOVES:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)

            if not grid.is_free(nr, nc):
                continue

            new_dir = (dr, dc)
            move_cost = energy_cost(cur_dir, new_dir)
            move_cost += proximity_penalty(grid, nr, nc)

            tentative_g = g_score[current] + move_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                came_from[neighbor] = current
                direction_map[neighbor] = new_dir
                heapq.heappush(open_set, (f_score, neighbor, new_dir))

    return [], float('inf')  # No path found


# ─────────────────────────────────────────────────────────────
# STAGE 2C: OPTIMISED COVERAGE PLANNER
# ─────────────────────────────────────────────────────────────

class CoveragePlanner:
    def __init__(self, grid: Grid, battery: float = 500.0):
        self.grid = grid
        self.battery = battery
        self.max_battery = battery
        self.current_pos = grid.start
        self.full_path = [grid.start]
        self.total_energy = 0.0
        self.prev_dir = None
        self.waypoints_visited = 0

    def plan(self):
        """
        Main planning function:
        1. Generate boustrophedon waypoints
        2. Use A* to connect each waypoint with energy-aware routing
        3. Stop if battery runs out
        """
        waypoints = boustrophedon_path(self.grid)

        print(f"📍 Total waypoints to cover: {len(waypoints)}")
        print(f"🔋 Starting battery: {self.battery}")
        print(f"🚁 Starting position: {self.current_pos}")
        print("-" * 50)

        for target in waypoints:
            if target == self.current_pos:
                self.grid.mark_visited(*target)
                self.waypoints_visited += 1
                continue

            if not self.grid.is_free(*target):
                continue

            # Find energy-optimal path to next waypoint
            path, cost = astar(self.grid, self.current_pos, target, self.prev_dir)

            if not path or cost == float('inf'):
                continue  # Can't reach this cell, skip

            if self.battery - cost < 0:
                print(f"⚠️  Battery critical! Stopping at {self.current_pos}")
                print(f"   Remaining battery: {self.battery:.2f}")
                break

            # Execute the path
            for step in path[1:]:
                self.full_path.append(step)
                self.grid.mark_visited(*step)

            self.battery -= cost
            self.total_energy += cost
            self.current_pos = target
            self.waypoints_visited += 1

            # Update previous direction
            if len(path) >= 2:
                dr = path[-1][0] - path[-2][0]
                dc = path[-1][1] - path[-2][1]
                self.prev_dir = (dr, dc)

        return self.full_path

    def stats(self):
        coverage = self.grid.coverage_percentage()
        battery_used = self.max_battery - self.battery
        battery_pct = (battery_used / self.max_battery) * 100
        print("\n" + "=" * 50)
        print("📊 MISSION STATISTICS")
        print("=" * 50)
        print(f"  ✅ Area Coverage     : {coverage}%")
        print(f"  🔋 Battery Used      : {battery_used:.1f} / {self.max_battery} ({battery_pct:.1f}%)")
        print(f"  🔋 Battery Remaining : {self.battery:.1f}")
        print(f"  📍 Total Steps       : {len(self.full_path)}")
        print(f"  ⚡ Total Energy Cost : {self.total_energy:.2f}")
        print(f"  🎯 Waypoints Visited : {self.waypoints_visited}")
        print("=" * 50)
        return {
            "coverage_pct": coverage,
            "battery_used": battery_used,
            "battery_remaining": self.battery,
            "total_steps": len(self.full_path),
            "total_energy": self.total_energy,
        }
