# 🦅 GARUDA-OPS — Autonomous Drone Surveillance

## Folder Structure
```
GARUDA-OPS/
├── html/
│   ├── login.html       → Login Page
│   └── garuda_ops.html  → Main Simulation
├── css/
│   ├── login.css        → Login Page Styling
│   └── garuda.css       → Simulation Styling
├── js/
│   ├── login.js         → Login Logic
│   └── garuda.js        → A*, Q-Learning, Drone Logic
└── python/
    ├── main.py           → Run karo ye
    ├── grid.py           → Grid Map
    ├── planner.py        → Boustrophedon + A*
    ├── dynamic_replanner.py → D* Lite
    └── rl_agent.py       → Q-Learning
```

## Chalane ka tarika

### Frontend
1. `html/login.html` browser mein open karo
2. Login: Gaurav / Gaurav@4355

### Backend
```bash
cd python
pip install numpy
python main.py
```

## Made by: Gaurav
## Project: GARUDA-OPS Hackathon
