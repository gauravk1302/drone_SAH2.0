"""
main.py - Run the full Autonomous Surveillance Path Optimisation Pipeline
"""

from grid import create_sample_map
from planner import CoveragePlanner
from dynamic_replanner import DynamicMission

def main():
    print("=" * 60)
    print("  🚁 AUTONOMOUS SURVEILLANCE PATH OPTIMISATION")
    print("  Algorithm: Boustrophedon + A* Energy Cost + D* Lite")
    print("=" * 60)

    # ── Stage 1: Create Map ──────────────────────────────────
    print("\n📌 Stage 1: Building Grid Map...")
    grid = create_sample_map(rows=20, cols=20)
    grid.print_grid()

    # ── Stage 2: Coverage Path Planning ─────────────────────
    print("\n📌 Stage 2: Computing Optimised Coverage Path...")
    planner = CoveragePlanner(grid, battery=500.0)
    planned_path = planner.plan()
    planner.stats()

    # ── Stage 3: Dynamic Replanning ──────────────────────────
    print("\n📌 Stage 3: Simulating Mid-Flight Obstacle Discovery...")

    # Reset visited cells for dynamic simulation
    grid2 = create_sample_map(rows=20, cols=20)

    # These obstacles will appear mid-flight (not on initial map!)
    dynamic_obstacles = {
        50:  (3, 10),   # Obstacle discovered at step 50
        120: (7, 14),   # Another one at step 120
        200: (15, 8),   # One more at step 200
    }

    mission = DynamicMission(
        grid=grid2,
        planned_path=planned_path,
        battery=500.0,
        dynamic_obstacle_schedule=dynamic_obstacles
    )
    executed_path, events = mission.run()

    # ── Final Summary ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ ALGORITHM PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total planned waypoints   : {len(planned_path)}")
    print(f"  Total executed steps      : {len(executed_path)}")
    print(f"  Dynamic replanning events : {len(events)}")
    print(f"  Final coverage            : {grid.coverage_percentage()}%")
    print("\n  Ready for visualisation layer! 🎨")
    print("=" * 60)

if __name__ == "__main__":
    main()
