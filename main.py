import matplotlib.pyplot as plt
import numpy as np
import random
from map import create_grid, print_grid
from drone import Drone
from q_learning import QLearningPlanner
from gui import run_simulation

def latlon_to_grid(lat, lon, lat_min, lat_max, lon_min, lon_max, grid_size):
    row = int((lat_max - lat) / (lat_max - lat_min) * (grid_size - 1))
    col = int((lon - lon_min) / (lon_max - lon_min) * (grid_size - 1))
    return (row, col)

def run_drone_simulation(start_coords, end_coords):
    lat_min, lat_max = 28.40, 28.90
    lon_min, lon_max = 76.80, 77.40
    grid_size = 10

    start = latlon_to_grid(*start_coords, lat_min, lat_max, lon_min, lon_max, grid_size)
    end = latlon_to_grid(*end_coords, lat_min, lat_max, lon_min, lon_max, grid_size)

    grid = create_grid()
    drone = Drone(start=start, energy=50)
    planner = QLearningPlanner(grid, epsilon=0.1)
    state = drone.position
    path = [state]
    visited_states = set()

    while state != end:
        if state in visited_states:
            break
        visited_states.add(state)

        action = planner.choose_action(state)
        if action is None:
            break

        reward = -10 if action in visited_states else -1 if grid[action[0]][action[1]] == 0 else -100
        planner.update(state, action, reward, action)
        state = action
        path.append(state)
        if len(path) > 100:
            break

    if path:
        for step in path[1:]:
            dx = abs(step[0] - drone.position[0])
            dy = abs(step[1] - drone.position[1])
            cost = 1.4 if dx == 1 and dy == 1 else 1
            moved = drone.move(step, cost)
            if random.random() < 0.2:
                ox, oy = random.randint(0, 9), random.randint(0, 9)
                if grid[ox][oy] == 0 and (ox, oy) not in drone.visited:
                    grid[ox][oy] = 1
            if not moved:
                break

        planner.save_q_table()

    total_free_cells = grid.size - np.count_nonzero(grid)
    coverage = len(drone.visited) / total_free_cells if total_free_cells > 0 else 0

    # Save path plot
    x, y = zip(*drone.path)
    plt.imshow(grid, cmap='Greys')
    plt.plot(y, x, marker='o', color='blue', linewidth=2)
    plt.scatter(y[0], x[0], color='green', label='Start')
    plt.scatter(y[-1], x[-1], color='red', label='End')
    plt.title("🚁 Drone Path")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.savefig("static/path.png")
    plt.close()

    # Save heatmap
    heatmap = np.zeros_like(grid, dtype=float)
    for (x, y) in drone.visited:
        heatmap[x][y] = 1
    plt.imshow(heatmap, cmap='YlOrRd')
    plt.title("🔥 Coverage Heatmap")
    plt.colorbar(label='Visited Intensity')
    plt.gca().invert_yaxis()
    plt.savefig("static/heatmap.png")
    plt.close()

    return {
        "energy": round(drone.energy, 2),
        "coverage": round(coverage * 100, 2),
        "goal_reached": end in drone.visited
    }
