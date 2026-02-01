import numpy as np

def create_grid(rows=10, cols=10):
    """
    Create a 2D grid with some obstacles.
    0 = free space
    1 = obstacle
    """
    grid = np.zeros((rows, cols), dtype=int)

    # Add some obstacles manually
    obstacles = [(3, 4), (5, 5), (1, 2), (6, 7), (2, 6)]
    for x, y in obstacles:
        grid[x][y] = 1

    return grid

def is_valid_cell(grid, x, y):
    """
    Check if a cell is within bounds and not an obstacle.
    """
    rows, cols = grid.shape
    return 0 <= x < rows and 0 <= y < cols and grid[x][y] == 0

def print_grid(grid):
    """
    Print the grid in a readable format.
    """
    for row in grid:
        print(" ".join(str(cell) for cell in row))

# Test the functions
if __name__ == "__main__":
    grid = create_grid()
    print("Grid Layout (0 = free, 1 = obstacle):")
    print_grid(grid)
