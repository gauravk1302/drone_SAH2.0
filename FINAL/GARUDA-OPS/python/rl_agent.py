"""
rl_agent.py - Q-Learning Agent for Drone Path Planning
The drone learns from discovered obstacles and remembers them across missions.
Each mission it gets smarter — avoiding previously dangerous cells from the start.
"""

import json
import os
import random
import math

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
ROWS, COLS = 20, 20
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]  # Up, Down, Left, Right

# Q-Learning hyperparameters
ALPHA       = 0.3    # Learning rate — how fast we update Q-values
GAMMA       = 0.9    # Discount factor — how much we value future rewards
EPSILON_START = 0.9  # Exploration rate (mission 1 = explore a lot)
EPSILON_DECAY = 0.3  # Reduce exploration each mission
EPSILON_MIN   = 0.05 # Always keep a tiny bit of exploration

# Rewards
R_VISIT_NEW   = +10.0   # Reward for visiting a new cell
R_REVISIT     = -2.0    # Penalty for revisiting already-seen cell
R_OBSTACLE    = -50.0   # Heavy penalty for hitting an obstacle
R_NO_FLY      = -100.0  # Very heavy penalty for entering no-fly zone
R_STEP        = -0.5    # Small step cost (encourages efficiency)
R_GOAL        = +50.0   # Bonus for high coverage

MEMORY_FILE = "drone_memory.json"

# ─────────────────────────────────────────────────────────────
# Q-TABLE (Persistent Memory)
# ─────────────────────────────────────────────────────────────
class DroneMemory:
    """
    Persistent Q-table that survives across missions.
    Stores learned Q-values and known danger zones.
    """

    def __init__(self):
        self.q_table = {}          # {state: {action_idx: q_value}}
        self.danger_map = {}       # {(r,c): penalty_score} — learned danger zones
        self.obstacle_memory = []  # List of discovered obstacles across all missions
        self.mission_count = 0
        self.mission_history = []  # Stats per mission

    def get_q(self, state, action_idx):
        key = str(state)
        if key not in self.q_table:
            self.q_table[key] = [0.0] * len(ACTIONS)
        return self.q_table[key][action_idx]

    def set_q(self, state, action_idx, value):
        key = str(state)
        if key not in self.q_table:
            self.q_table[key] = [0.0] * len(ACTIONS)
        self.q_table[key][action_idx] = value

    def best_action(self, state):
        key = str(state)
        if key not in self.q_table:
            return random.randint(0, len(ACTIONS)-1)
        return max(range(len(ACTIONS)), key=lambda i: self.q_table[key][i])

    def record_obstacle(self, r, c, mission):
        """Remember a discovered obstacle and penalize surrounding cells"""
        if (r, c) not in self.obstacle_memory:
            self.obstacle_memory.append((r, c))

        # Penalize the obstacle cell and nearby cells with distance-based penalty
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r+dr, c+dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    dist = math.sqrt(dr**2 + dc**2)
                    penalty = 50.0 / (dist + 0.5)
                    key = (nr, nc)
                    self.danger_map[key] = self.danger_map.get(key, 0) + penalty

    def get_danger(self, r, c):
        return self.danger_map.get((r, c), 0.0)

    def record_mission(self, coverage, battery_used, steps, replannings, obstacles_found):
        self.mission_count += 1
        self.mission_history.append({
            "mission": self.mission_count,
            "coverage": round(coverage, 2),
            "battery_used": round(battery_used, 1),
            "steps": steps,
            "replannings": replannings,
            "obstacles_found": obstacles_found,
            "known_obstacles": len(self.obstacle_memory)
        })

    def save(self, filepath=MEMORY_FILE):
        data = {
            "q_table": self.q_table,
            "danger_map": {str(k): v for k,v in self.danger_map.items()},
            "obstacle_memory": self.obstacle_memory,
            "mission_count": self.mission_count,
            "mission_history": self.mission_history
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Memory saved: {len(self.q_table)} states, {len(self.obstacle_memory)} known obstacles")

    def load(self, filepath=MEMORY_FILE):
        if not os.path.exists(filepath):
            print("📂 No previous memory found. Starting fresh.")
            return False
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.q_table = data.get("q_table", {})
        self.danger_map = {eval(k): v for k,v in data.get("danger_map", {}).items()}
        self.obstacle_memory = [tuple(o) for o in data.get("obstacle_memory", [])]
        self.mission_count = data.get("mission_count", 0)
        self.mission_history = data.get("mission_history", [])
        print(f"🧠 Memory loaded: {len(self.q_table)} states, {len(self.obstacle_memory)} known obstacles, {self.mission_count} past missions")
        return True


# ─────────────────────────────────────────────────────────────
# Q-LEARNING AGENT
# ─────────────────────────────────────────────────────────────
class QLearningDrone:
    """
    Q-Learning agent that plans paths using learned experience.
    Combines Q-values with A* for hybrid intelligent navigation.
    """

    def __init__(self, grid_data, memory: DroneMemory, mission_num: int):
        self.grid = [row[:] for row in grid_data]  # Deep copy
        self.memory = memory
        self.mission_num = mission_num
        self.pos = (0, 0)
        self.visited = set()
        self.visited.add(self.pos)
        self.path_taken = [self.pos]
        self.total_reward = 0.0
        self.replannings = 0
        self.obstacles_found = []
        self.battery = 500.0

        # Epsilon decays with missions — less random exploration over time
        self.epsilon = max(EPSILON_MIN, EPSILON_START - EPSILON_DECAY * (mission_num - 1))
        print(f"\n🤖 Mission {mission_num} | Epsilon (exploration): {self.epsilon:.2f}")
        print(f"   Known danger zones: {len(memory.danger_map)}")
        print(f"   Remembered obstacles: {len(memory.obstacle_memory)}")

        # Pre-apply memory: mark remembered obstacles on grid
        self._apply_memory_to_grid()

    def _apply_memory_to_grid(self):
        """Before flying, apply all remembered obstacles to grid (key RL feature!)"""
        from grid import OBSTACLE
        applied = 0
        for (r, c) in self.memory.obstacle_memory:
            if 0 <= r < ROWS and 0 <= c < COLS and self.grid[r][c] == 0:  # FREE
                self.grid[r][c] = OBSTACLE
                applied += 1
        if applied:
            print(f"   ✅ Pre-applied {applied} remembered obstacles to map")

    def get_state(self):
        r, c = self.pos
        return (r, c)

    def get_reward(self, new_r, new_c, hit_obstacle=False):
        if hit_obstacle:
            return R_OBSTACLE
        cell = self.grid[new_r][new_c]
        from grid import NO_FLY
        if cell == NO_FLY:
            return R_NO_FLY
        reward = R_STEP
        # Danger zone penalty from memory
        danger = self.memory.get_danger(new_r, new_c)
        reward -= danger * 0.1
        if (new_r, new_c) not in self.visited:
            reward += R_VISIT_NEW
        else:
            reward += R_REVISIT
        return reward

    def choose_action(self):
        """
        Epsilon-greedy with memory bias:
        - With prob epsilon: explore (random)
        - With prob 1-epsilon: exploit best Q-value
        - Always penalize known danger zones
        """
        state = self.get_state()

        if random.random() < self.epsilon:
            # Exploration — but biased away from danger zones
            valid = []
            for i, (dr, dc) in enumerate(ACTIONS):
                nr, nc = self.pos[0]+dr, self.pos[1]+dc
                if self._is_valid(nr, nc):
                    danger = self.memory.get_danger(nr, nc)
                    weight = max(0.1, 1.0 - danger/50.0)
                    valid.append((i, weight))
            if not valid:
                return random.randint(0, len(ACTIONS)-1)
            total = sum(w for _,w in valid)
            r = random.random() * total
            for i, w in valid:
                r -= w
                if r <= 0:
                    return i
            return valid[-1][0]
        else:
            # Exploitation — best Q-value, penalized by danger
            best_i, best_score = None, float('-inf')
            for i, (dr, dc) in enumerate(ACTIONS):
                nr, nc = self.pos[0]+dr, self.pos[1]+dc
                if self._is_valid(nr, nc):
                    q = self.memory.get_q(state, i)
                    danger = self.memory.get_danger(nr, nc)
                    score = q - danger * 0.2
                    if score > best_score:
                        best_score = score
                        best_i = i
            if best_i is None:
                return random.randint(0, len(ACTIONS)-1)
            return best_i

    def _is_valid(self, r, c):
        from grid import OBSTACLE, NO_FLY
        if r < 0 or r >= ROWS or c < 0 or c >= COLS:
            return False
        return self.grid[r][c] not in [OBSTACLE, NO_FLY]

    def step(self, action_idx, dynamic_obstacle_pos=None):
        """
        Execute one step. Returns (new_pos, reward, done, hit_obstacle)
        """
        from grid import OBSTACLE, FREE, VISITED, NO_FLY
        dr, dc = ACTIONS[action_idx]
        nr, nc = self.pos[0]+dr, self.pos[1]+dc
        old_state = self.get_state()

        # Reveal dynamic obstacle when scheduled (simulates incomplete map data)
        hit_obstacle = False
        if dynamic_obstacle_pos:
            obs_r, obs_c = dynamic_obstacle_pos
            if self.grid[obs_r][obs_c] == FREE:
                self.grid[obs_r][obs_c] = OBSTACLE
                self.memory.record_obstacle(obs_r, obs_c, self.mission_num)
                if dynamic_obstacle_pos not in self.obstacles_found:
                    self.obstacles_found.append(dynamic_obstacle_pos)
                    self.replannings += 1
                    hit_obstacle = True

        if not self._is_valid(nr, nc):
            reward = R_OBSTACLE
            new_pos = self.pos
        else:
            new_pos = (nr, nc)
            reward = self.get_reward(nr, nc, hit_obstacle=False)
            if self.grid[nr][nc] == FREE:
                self.grid[nr][nc] = VISITED
            self.visited.add(new_pos)
            self.pos = new_pos
            self.path_taken.append(new_pos)

        self.battery -= 1.0
        self.total_reward += reward

        # Q-Learning update (Bellman equation)
        new_state = self.get_state()
        old_q = self.memory.get_q(old_state, action_idx)
        best_next_q = max(self.memory.get_q(new_state, i) for i in range(len(ACTIONS)))
        new_q = old_q + ALPHA * (reward + GAMMA * best_next_q - old_q)
        self.memory.set_q(old_state, action_idx, new_q)

        done = self.battery <= 0
        return new_pos, reward, done, hit_obstacle

    def coverage(self):
        from grid import VISITED, START
        visited = sum(1 for r in range(ROWS) for c in range(COLS)
                     if self.grid[r][c] in [VISITED, START])
        total = sum(1 for r in range(ROWS) for c in range(COLS)
                   if self.grid[r][c] not in [1, 2])  # not obstacle or no-fly
        return (visited / total * 100) if total > 0 else 0


# ─────────────────────────────────────────────────────────────
# SIMULATION RUNNER
# ─────────────────────────────────────────────────────────────
def run_rl_missions(num_missions=3, steps_per_mission=400):
    from grid import create_sample_map

    memory = DroneMemory()
    memory.load()  # Load previous experience if exists

    # Dynamic obstacles that will be "discovered" mid-flight
    # Same obstacles appear each mission — but drone learns to avoid them!
    dynamic_obstacles = [(3,10), (7,14), (15,8), (11,5)]

    print("\n" + "="*60)
    print("  🤖 Q-LEARNING DRONE — MULTI-MISSION TRAINING")
    print("="*60)

    for mission in range(memory.mission_count + 1, memory.mission_count + num_missions + 1):
        print(f"\n{'='*60}")
        print(f"  🚁 MISSION {mission}")
        print(f"{'='*60}")

        g = create_sample_map()
        agent = QLearningDrone(g.grid, memory, mission)

        # Shuffle which dynamic obstacles appear this mission
        mission_obstacles = random.sample(dynamic_obstacles, k=min(2, len(dynamic_obstacles)))
        obs_schedule = {}
        for i, obs in enumerate(mission_obstacles):
            obs_schedule[random.randint(50,200)] = obs

        step = 0
        obs_triggered = set()
        for step in range(steps_per_mission):
            # Trigger dynamic obstacle at scheduled step
            dyn_obs = obs_schedule.get(step)

            action = agent.choose_action()
            pos, reward, done, hit = agent.step(action, dyn_obs)

            if hit:
                print(f"  ⚠️  Step {step}: Obstacle found at {dyn_obs} → Memory updated!")

            if done:
                break

        cov = agent.coverage()
        battery_used = 500.0 - agent.battery
        print(f"\n  📊 Mission {mission} Results:")
        print(f"     Coverage      : {cov:.1f}%")
        print(f"     Battery Used  : {battery_used:.0f}/500")
        print(f"     Steps         : {step}")
        print(f"     Replannings   : {agent.replannings}")
        print(f"     Obs Discovered: {len(agent.obstacles_found)}")
        print(f"     Total Reward  : {agent.total_reward:.1f}")

        memory.record_mission(cov, battery_used, step, agent.replannings, len(agent.obstacles_found))

    memory.save()

    print("\n" + "="*60)
    print("  📈 LEARNING PROGRESS (Missions over time)")
    print("="*60)
    for m in memory.mission_history[-num_missions:]:
        replan_bar = "▓" * m['replannings'] + "░" * max(0, 5-m['replannings'])
        print(f"  Mission {m['mission']:2d} | Cov:{m['coverage']:5.1f}% | Replan:[{replan_bar}] {m['replannings']} | Obs known:{m['known_obstacles']}")

    print("\n  ✅ RL Training Complete! Memory saved for next run.")
    print("     Next mission will be SMARTER — fewer replannings expected.")


if __name__ == "__main__":
    run_rl_missions(num_missions=3)
