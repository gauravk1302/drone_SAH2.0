# 🦅 GARUDA-OPS — Backend (Python)

## Files
| File | Kya karta hai |
|------|--------------|
| `grid.py` | 20×20 grid map — FREE, OBS, NOFLY, VISITED cells |
| `planner.py` | Boustrophedon path + A* energy-aware routing |
| `dynamic_replanner.py` | D* Lite — mid-flight obstacle replanning |
| `rl_agent.py` | Q-Learning — drone learns across missions |
| `main.py` | Sab stages ko ek saath run karta hai |

## Run karo
```bash
pip install -r requirements.txt

# Full pipeline (Stage 1 + 2 + 3)
python main.py

# Sirf Q-Learning training (3 missions)
python rl_agent.py
```

## Algorithm Flow
```
Stage 1: Grid Map banao (grid.py)
    ↓
Stage 2: Boustrophedon + A* path plan (planner.py)
    ↓
Stage 3: D* Lite dynamic replanning (dynamic_replanner.py)
    ↓
Stage 4: Q-Learning — memory across missions (rl_agent.py)
```
