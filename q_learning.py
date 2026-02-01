import random
import json
import os

class QLearningPlanner:
    def __init__(self, grid, alpha=0.1, gamma=0.9, epsilon=0.2, q_file="q_table.json"):
        self.grid = grid
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_file = q_file
        self.load_q_table()

    def get_actions(self, state):
        actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
        valid = []
        for dx, dy in actions:
            nx, ny = state[0] + dx, state[1] + dy
            if 0 <= nx < len(self.grid) and 0 <= ny < len(self.grid[0]) and self.grid[nx][ny] == 0:
                valid.append((nx, ny))
        return valid

    def choose_action(self, state):
        actions = self.get_actions(state)
        if not actions:
            return None
        if random.random() < self.epsilon or state not in self.q_table:
            return random.choice(actions)
        return max(self.q_table[state], key=self.q_table[state].get)

    def update(self, state, action, reward, next_state):
        self.q_table.setdefault(state, {})
        self.q_table[state].setdefault(action, 0)
        max_future_q = max(self.q_table.get(next_state, {}).values(), default=0)
        self.q_table[state][action] += self.alpha * (reward + self.gamma * max_future_q - self.q_table[state][action])

    def save_q_table(self):
        serializable_q_table = {
            str(k): {str(a): v for a, v in actions.items()}
            for k, actions in self.q_table.items()
        }
        with open(self.q_file, "w") as f:
            json.dump(serializable_q_table, f)

    def load_q_table(self):
        if os.path.exists(self.q_file):
            with open(self.q_file, "r") as f:
                raw = json.load(f)
                self.q_table = {
                    eval(k): {eval(a): v for a, v in actions.items()}
                    for k, actions in raw.items()
                }
