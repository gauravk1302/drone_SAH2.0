import pygame
import time

CELL_SIZE = 50
MARGIN = 2
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 120, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GREY = (200, 200, 200)

def draw_grid(screen, grid, drone):
    screen.fill(WHITE)
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            color = WHITE
            if grid[row][col] == 1:
                color = BLACK
            elif (row, col) in drone.visited:
                color = GREY
            pygame.draw.rect(screen, color,
                [(MARGIN + CELL_SIZE) * col + MARGIN,
                 (MARGIN + CELL_SIZE) * row + MARGIN,
                 CELL_SIZE, CELL_SIZE])
    # Draw drone
    x, y = drone.position
    pygame.draw.rect(screen, BLUE,
        [(MARGIN + CELL_SIZE) * y + MARGIN,
         (MARGIN + CELL_SIZE) * x + MARGIN,
         CELL_SIZE, CELL_SIZE])
    pygame.display.flip()

def run_simulation(grid, drone, path):
    pygame.init()
    width = len(grid[0]) * (CELL_SIZE + MARGIN) + MARGIN
    height = len(grid) * (CELL_SIZE + MARGIN) + MARGIN
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Drone Path Animation")

    running = True
    for step in path:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        dx = abs(step[0] - drone.position[0])
        dy = abs(step[1] - drone.position[1])
        cost = 1.4 if dx == 1 and dy == 1 else 1
        moved = drone.move(step, cost)
        draw_grid(screen, grid, drone)
        time.sleep(0.2)
        if not moved:
            print("Drone ran out of energy!")
            break

    time.sleep(2)
    pygame.quit()
