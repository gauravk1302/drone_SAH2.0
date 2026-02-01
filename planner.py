from map import is_valid_cell
import math

def get_neighbors(pos, grid):
    """
    Return valid neighboring cells (up, down, left, right).
    """
    x, y = pos
    directions = [(0,1), (0,-1), (1,0), (-1,0)]
    neighbors = []

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if is_valid_cell(grid, nx, ny):
            neighbors.append((nx, ny))

    return neighbors

def euclidean_distance(a, b):
    """
    Calculate Euclidean distance between two points.
    """
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def choose_next_move(drone, grid):
    """
    Choose the nearest unvisited neighbor.
    """
    neighbors = get_neighbors(drone.position, grid)
    unvisited = [n for n in neighbors if n not in drone.visited]

    if not unvisited:
        return None

    # Greedy: pick the closest unvisited cell to (0,0) or any target
    unvisited.sort(key=lambda cell: euclidean_distance(cell, (0, 0)))
    return unvisited[0]

if __name__ == "__main__":
    from map import create_grid
    from drone import Drone

    grid = create_grid()
    drone = Drone(start=(0, 0))

    for _ in range(10):
        next_move = choose_next_move(drone, grid)
        if next_move:
            drone.move(next_move)
            print(f"Moved to {next_move}")
        else:
            print("No more unvisited neighbors.")
            break

def get_neighbors(pos, grid):
    """
    Return valid neighboring cells (including diagonals).
    """
    x, y = pos
    directions = [
        (0, 1), (0, -1), (1, 0), (-1, 0),  # straight
        (1, 1), (-1, -1), (1, -1), (-1, 1)  # diagonals
    ]
    neighbors = []

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if is_valid_cell(grid, nx, ny):
            neighbors.append((nx, ny))

    return neighbors
