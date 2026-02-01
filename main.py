from map import create_grid, print_grid
from drone import Drone
from planner import choose_next_move
import matplotlib.pyplot as plt
import numpy as np
from astar_module import astar
import random
from gui import run_simulation
import json


# -------------------------------
# 🌍 Coordinate Mapping Function
# -------------------------------
def latlon_to_grid(lat, lon, lat_min, lat_max, lon_min, lon_max, grid_size):
    row = int((lat_max - lat) / (lat_max - lat_min) * (grid_size - 1))
    col = int((lon - lon_min) / (lon_max - lon_min) * (grid_size - 1))
    return (row, col)

# -------------------------------
# 🌐 Define Bounding Box (Adjust as needed)
# -------------------------------
with open("coords.json") as f: coords = json.load(f) start_lat, start_lon = coords["start"]["lat"], coords["start"]["lng"] end_lat, end_lon = coords["end"]["lat"], coords["end"]["lng"]

# -------------------------------
# 📍 Real-World Coordinates (Replace with dynamic input later)
# -------------------------------
start_lat, start_lon = 28.6129, 77.2295  # Your current location
end_lat, end_lon = 28.5355, 77.3910      # Destination

start = latlon_to_grid(start_lat, start_lon, lat_min, lat_max, lon_min, lon_max, grid_size)
end = latlon_to_grid(end_lat, end_lon, lat_min, lat_max, lon_min, lon_max, grid_size)

# -------------------------------
# 🗺️ Create Grid and Drone
# -------------------------------
grid = create_grid()
print("Initial Grid:")
print_grid(grid)

drone = Drone(start=start, energy=50)
goals = [end]

# -------------------------------
# 🚁 Navigate to Each Goal
# -------------------------------
for goal in goals:
    print(f"\nNavigating to goal: {goal}")
    path = astar(grid, drone.position, goal)

    if path:
        run_simulation(grid, drone, path[1:])  # GUI animation

        for step in path[1:]:
            dx = abs(step[0] - drone.position[0])
            dy = abs(step[1] - drone.position[1])
            cost = 1.4 if dx == 1 and dy == 1 else 1
            moved = drone.move(step, cost)

            # 🧱 Add dynamic obstacle randomly
            if random.random() < 0.2:
                ox, oy = random.randint(0, 9), random.randint(0, 9)
                if grid[ox][oy] == 0 and (ox, oy) not in drone.visited:
                    grid[ox][oy] = 1
                    print(f"⚠️ Dynamic obstacle added at ({ox},{oy})")

            if not moved:
                print("🚨 Drone ran out of energy!")
                break
    else:
        print(f"❌ No path found to goal {goal}.")
        break

# -------------------------------
# 📊 Final Stats and Visualization
# -------------------------------
print("\n✅ Final Drone Status:")
drone.status()

total_free_cells = grid.size - np.count_nonzero(grid)
coverage = len(drone.visited) / total_free_cells
print(f"📊 Coverage Score: {coverage*100:.2f}%")
print(f"🔋 Remaining Energy: {drone.energy:.2f}")

# Path Visualization
x, y = zip(*drone.path)
plt.imshow(grid, cmap='Greys')
plt.plot(y, x, marker='o', color='blue', linewidth=2)
plt.scatter(y[0], x[0], color='green', label='Start')
plt.scatter(y[-1], x[-1], color='red', label='End')
plt.title("🚁 Drone Path Visualization with Dynamic Obstacles")
plt.legend()
plt.gca().invert_yaxis()
plt.show()

# 🔥 Heatmap
heatmap = np.zeros_like(grid, dtype=float)
for (x, y) in drone.visited:
    heatmap[x][y] = 1

plt.figure()
plt.imshow(heatmap, cmap='YlOrRd')
plt.title("🔥 Drone Coverage Heatmap")
plt.colorbar(label='Visited Intensity')
plt.gca().invert_yaxis()
plt.show()

# 📈 Summary
print("\n📈 Summary Report")
print(f"Total Goals Reached: {len([g for g in goals if g in drone.visited])}")
print(f"Total Cells Visited: {len(drone.visited)}")
print(f"Energy Used: {50 - drone.energy:.2f}")
