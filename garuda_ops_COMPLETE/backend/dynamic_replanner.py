"""
dynamic_replanner.py - D* Lite Dynamic Replanning
Stage 3: When a new obstacle is detected mid-flight, replan efficiently
without recomputing the entire path from scratch.
"""

import heapq
import math
from grid import Grid, FREE, OBSTACLE, NO_FLY, VISITED

INF = float('inf')

class DStarLite:
    """
    D* Lite algorithm for dynamic replanning.
    More efficient than re-running A* from scratch every time
    an obstacle is discovered — only updates affected nodes.
    """

    def __init__(self, grid: Grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.k_m = 0  # Key modifier for accumulated heuristic shifts

        self.g = {}      # Cost from node to goal
        self.rhs = {}    # One-step lookahead cost

        self.open_set = []
        self._counter = 0  # Tiebreaker for heap

        self._init_all_nodes()
        self.rhs[self.goal] = 0
        key = self._calc_key(self.goal)
        heapq.heappush(self.open_set, (*key, self.goal))

    def _init_all_nodes(self):
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                self.g[(r,c)]   = INF
                self.rhs[(r,c)] = INF

    def _heuristic(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def _calc_key(self, node):
        g_rhs = min(self.g[node], self.rhs[node])
        return (
            g_rhs + self._heuristic(self.start, node) + self.k_m,
            g_rhs
        )

    def _neighbors(self, node):
        r, c = node
        neighbors = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.grid.rows and 0 <= nc < self.grid.cols:
                neighbors.append((nr, nc))
        return neighbors

    def _cost(self, a, b):
        """Cost between adjacent nodes — INF if b is blocked"""
        r, c = b
        if not self.grid.is_free(r, c):
            return INF
        return 1.0

    def _update_vertex(self, node):
        if node != self.goal:
            # rhs = min cost through best neighbor
            min_cost = INF
            for nb in self._neighbors(node):
                c = self._cost(node, nb) + self.g[nb]
                if c < min_cost:
                    min_cost = c
            self.rhs[node] = min_cost

        # Remove existing entry for this node in open_set (lazy deletion)
        # Add back if inconsistent
        if self.g[node] != self.rhs[node]:
            key = self._calc_key(node)
            self._counter += 1
            heapq.heappush(self.open_set, (*key, node))

    def compute_shortest_path(self):
        while self.open_set:
            k_old = self.open_set[0][:2]
            k_start = self._calc_key(self.start)

            if k_old >= k_start and self.rhs[self.start] == self.g[self.start]:
                break

            node = heapq.heappop(self.open_set)[2]
            k_new = self._calc_key(node)

            if k_old < k_new:
                heapq.heappush(self.open_set, (*k_new, node))
            elif self.g[node] > self.rhs[node]:
                self.g[node] = self.rhs[node]
                for nb in self._neighbors(node):
                    self._update_vertex(nb)
            else:
                self.g[node] = INF
                self._update_vertex(node)
                for nb in self._neighbors(node):
                    self._update_vertex(nb)

    def extract_path(self):
        """Extract best path from start to goal"""
        path = [self.start]
        current = self.start
        visited = set()

        while current != self.goal:
            if current in visited:
                return []  # Loop detected, no valid path
            visited.add(current)

            best = None
            best_cost = INF
            for nb in self._neighbors(current):
                c = self._cost(current, nb) + self.g.get(nb, INF)
                if c < best_cost:
                    best_cost = c
                    best = nb

            if best is None or best_cost == INF:
                return []  # No path to goal

            path.append(best)
            current = best

        return path

    def notify_obstacle(self, obstacle_pos):
        """
        Called when a new obstacle is discovered mid-flight.
        Updates affected nodes and replans — much faster than full A*.
        """
        print(f"\n⚠️  NEW OBSTACLE DETECTED at {obstacle_pos}!")
        r, c = obstacle_pos
        self.grid.grid[r][c] = OBSTACLE

        self.k_m += self._heuristic(self.start, obstacle_pos)

        # Update all neighbors of the new obstacle
        for nb in self._neighbors(obstacle_pos):
            self._update_vertex(nb)

        self._update_vertex(obstacle_pos)
        self.compute_shortest_path()
        print(f"   ✅ Path replanned from current position {self.start}")

    def update_start(self, new_start):
        """Move the start position as drone moves"""
        self.start = new_start


# ─────────────────────────────────────────────────────────────
# Wrapper: Dynamic Mission Executor
# ─────────────────────────────────────────────────────────────

class DynamicMission:
    """
    Simulates a drone mission with mid-flight obstacle discovery.
    Uses D* Lite to replan when unknown obstacles are found.
    """

    def __init__(self, grid: Grid, planned_path: list, battery: float, 
                 dynamic_obstacle_schedule: dict = None):
        """
        dynamic_obstacle_schedule: {step_number: (row, col)} 
        — simulates discovering obstacles at specific steps
        """
        self.grid = grid
        self.planned_path = planned_path
        self.battery = battery
        self.dynamic_obstacle_schedule = dynamic_obstacle_schedule or {}
        self.executed_path = []
        self.replanning_events = []

    def run(self):
        print("\n🚁 STARTING DYNAMIC MISSION")
        print("=" * 50)

        i = 0
        current_path = list(self.planned_path)

        while i < len(current_path):
            pos = current_path[i]
            self.executed_path.append(pos)
            self.battery -= 1.0

            # Check if a dynamic obstacle appears at this step
            if i in self.dynamic_obstacle_schedule:
                obs_pos = self.dynamic_obstacle_schedule[i]

                # Only trigger if the obstacle is ahead in our path
                if obs_pos in current_path[i:]:
                    replanner = DStarLite(self.grid, pos, current_path[-1])
                    replanner.compute_shortest_path()
                    replanner.notify_obstacle(obs_pos)

                    new_path = replanner.extract_path()
                    if new_path:
                        current_path = self.executed_path + new_path[1:]
                        self.replanning_events.append({
                            "step": i,
                            "position": pos,
                            "obstacle": obs_pos,
                            "new_path_length": len(new_path)
                        })
                    else:
                        print(f"   ❌ No alternate path found! Mission continues on original route.")

            if self.battery <= 0:
                print(f"🔋 Battery depleted at step {i}")
                break

            i += 1

        print(f"\n✅ Mission complete!")
        print(f"   Steps executed    : {len(self.executed_path)}")
        print(f"   Replanning events : {len(self.replanning_events)}")
        print(f"   Battery remaining : {self.battery:.1f}")

        for event in self.replanning_events:
            print(f"   🔄 Replanned at step {event['step']} | pos {event['position']} | obstacle {event['obstacle']}")

        return self.executed_path, self.replanning_events
