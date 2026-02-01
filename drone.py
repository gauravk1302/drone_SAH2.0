class Drone:
    def __init__(self, start=(0, 0), energy=100):
        """
        Initialize the drone with a starting position and energy level.
        """
        self.position = start
        self.energy = energy
        self.path = [start]
        self.visited = set([start])

    def move(self, new_pos, cost=1):
        """
        Move the drone to a new position and update energy and path.
        """
        if self.energy < cost:
            print("Not enough energy to move!")
            return False

        self.position = new_pos
        self.energy -= cost
        self.path.append(new_pos)
        self.visited.add(new_pos)
        return True

    def status(self):
        """
        Print current status of the drone.
        """
        print(f"Position: {self.position}, Energy: {self.energy}, Visited: {len(self.visited)} cells")

if __name__ == "__main__":
    drone = Drone(start=(0, 0), energy=10)
    drone.status()

    moves = [(0, 1), (1, 1), (2, 1)]
    for move in moves:
        success = drone.move(move)
        if success:
            print(f"Moved to {move}")
        else:
            print("Move failed.")
        drone.status()
