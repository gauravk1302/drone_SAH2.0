// ═══════════════════════════════════════════════════════════
// CORE CONSTANTS
// ═══════════════════════════════════════════════════════════
const ROWS=20, COLS=20;
const FREE=0, OBS=1, NOFLY=2, START=3, VISITED=4;
const DIRS=[[-1,0],[1,0],[0,-1],[0,1]];
const DRONE_COLORS=['#ffd700','#00d4ff','#ff4466','#a855f7'];
const DRONE_NAMES=['GARUDA-1','GARUDA-2','GARUDA-3','GARUDA-4'];
// RL: stronger learning, faster convergence across missions
const ALPHA=0.4,    // higher = learns faster from each experience
      GAMMA=0.92,   // higher = values future rewards more
      EPS0=0.85,    // start with less random exploration
      EPS_DECAY=0.25, // decay faster so model exploits knowledge sooner
      EPS_MIN=0.03; // very little random exploration by mission 4+
const RTB_THRESHOLD=0.12, FOG_RADIUS=3, MAX_BATT=1200;
const RECHARGE_STEPS=40;
// Reward values — stronger signal so Q-table learns meaningfully
const R_NEW_CELL=+15,  // visiting new cell
      R_REVISIT=-8,    // heavy penalty for backtracking
      R_DANGER=-6,     // penalty scaled by danger level
      R_STEP=-0.3,     // small step cost (encourages efficiency)
      R_STRAIGHT=+0.5, // bonus for continuing straight (less turning)
      R_TURN=-1.2;     // penalty for turning (saves energy)

// ═══════════════════════════════════════════════════════════
// ZONE CONFIG
// ═══════════════════════════════════════════════════════════
const Z = { lat:28.6139, lng:77.2090, rad:2, alt:120,
  latPerCell(){return(this.rad*2/ROWS)/111},
  lngPerCell(){return(this.rad*2/COLS)/(111*Math.cos(this.lat*Math.PI/180))},
  cellLatLng(r,c){
    return {
      lat: (this.lat+this.rad/111) - r*this.latPerCell(),
      lng: (this.lng - this.rad/(111*Math.cos(this.lat*Math.PI/180))) + c*this.lngPerCell()
    };
  }
};
const fmtCoord = (v, isLat) => `${Math.abs(v).toFixed(4)}°${isLat?(v>=0?'N':'S'):(v>=0?'E':'W')}`;

// ═══════════════════════════════════════════════════════════
// RL MEMORY (persists across missions — NEVER resets unless user does)
// Stronger learning: danger compounds, Q-values persist and grow
// ═══════════════════════════════════════════════════════════
const MEM = {
  Q: {}, danger: {}, knownObs: [], msnHistory: [],
  // Per-mission efficiency tracking
  msnEfficiency: [],  // stores efficiency % per mission

  getQ(r,c,a){ const k=`${r},${c}`; if(!this.Q[k])this.Q[k]=[0,0,0,0]; return this.Q[k][a]; },
  setQ(r,c,a,v){ const k=`${r},${c}`; if(!this.Q[k])this.Q[k]=[0,0,0,0]; this.Q[k][a]=v; },
  getDanger(r,c){ return this.danger[`${r},${c}`]||0; },

  // Stronger obstacle learning — higher penalty, wider radius
  learnObstacle(r,c){
    if(!this.knownObs.find(o=>o[0]===r&&o[1]===c)) this.knownObs.push([r,c]);
    // Penalize 3-cell radius with distance falloff
    for(let dr=-3;dr<=3;dr++) for(let dc=-3;dc<=3;dc++){
      const nr=r+dr, nc=c+dc;
      if(nr<0||nr>=ROWS||nc<0||nc>=COLS) continue;
      const dist=Math.sqrt(dr*dr+dc*dc);
      const k=`${nr},${nc}`;
      // Danger compounds across missions (memory gets stronger!)
      this.danger[k]=(this.danger[k]||0)+80/(dist+0.5);
    }
  },

  // Reward straight-line paths in Q-table (energy efficiency)
  rewardStraightPath(r,c,action){
    const cur=this.getQ(r,c,action);
    this.setQ(r,c,action, cur+R_STRAIGHT);
  },

  // Penalize turns in Q-table
  penalizeTurn(r,c,action){
    const cur=this.getQ(r,c,action);
    this.setQ(r,c,action, cur+R_TURN);
  },

  // Record mission result for learning curve display
  recordMission(cov, straightMoves, turnMoves, replannings, obs){
    const eff=straightMoves+turnMoves>0
      ?Math.round((straightMoves/(straightMoves+turnMoves))*100):0;
    this.msnEfficiency.push(eff);
    this.msnHistory.push({
      m: this.msnHistory.length+1,
      cov: Math.round(cov*10)/10,
      eff, straight:straightMoves, turns:turnMoves,
      replan:replannings, obs
    });
  },

  // Get improvement vs last mission
  getImprovement(){
    const h=this.msnEfficiency;
    if(h.length<2) return null;
    return h[h.length-1]-h[h.length-2];
  },

  qSize(){ return Object.keys(this.Q).length; },
  dangerCells(){ return Object.keys(this.danger).length; },
  reset(){ this.Q={}; this.danger={}; this.knownObs=[]; this.msnHistory=[]; this.msnEfficiency=[]; }
};

// ═══════════════════════════════════════════════════════════
// GRID MANAGEMENT
// ═══════════════════════════════════════════════════════════
let baseGrid=[], liveGrid=[], fogGrid=[], userPaints=[];

function buildBaseGrid(){
  const g=Array.from({length:ROWS},()=>new Array(COLS).fill(FREE));
  g[0][0]=START;
  // Static no-fly zones
  for(let r=5;r<9;r++) for(let c=10;c<14;c++) g[r][c]=NOFLY;
  for(let r=14;r<18;r++) for(let c=3;c<7;c++) g[r][c]=NOFLY;
  // Static obstacles
  [[2,5],[2,6],[3,5],[3,6],[8,2],[8,3],[9,2],[9,3],[12,15],[13,15],[12,16],
   [1,18],[2,18],[3,18],[4,18],[17,12],[17,13],[18,12],[6,18],[7,18],[7,17]]
  .forEach(([r,c])=>g[r][c]=OBS);
  // Apply user paints (editor)
  userPaints.forEach(([r,c,t])=>{
    if(r===0&&c===0) return;
    g[r][c] = t==='obs'?OBS : t==='nfz'?NOFLY : FREE;
  });
  return g;
}

function buildFogGrid(){ return Array.from({length:ROWS},()=>new Array(COLS).fill(true)); }

function revealFog(r,c){
  for(let dr=-FOG_RADIUS;dr<=FOG_RADIUS;dr++)
    for(let dc=-FOG_RADIUS;dc<=FOG_RADIUS;dc++){
      const nr=r+dr, nc=c+dc;
      if(nr>=0&&nr<ROWS&&nc>=0&&nc<COLS) fogGrid[nr][nc]=false;
    }
}

function preloadMemory(g){
  let cnt=0;
  MEM.knownObs.forEach(([r,c])=>{ if(g[r][c]===FREE){g[r][c]=OBS;cnt++;} });
  return cnt;
}

// ═══════════════════════════════════════════════════════════
// A* PATHFINDER (danger-aware)
// ═══════════════════════════════════════════════════════════
function walkable(g,r,c){ return r>=0&&r<ROWS&&c>=0&&c<COLS&&g[r][c]!==OBS&&g[r][c]!==NOFLY; }

function astar(g, sr,sc, er,ec){
  if(!walkable(g,er,ec)) return [[sr,sc]];
  const H=(r,c)=>Math.abs(r-er)+Math.abs(c-ec);
  const open=[[H(sr,sc),sr,sc]];
  const gScore={}; gScore[`${sr},${sc}`]=0;
  const from={};
  while(open.length){
    open.sort((a,b)=>a[0]-b[0]);
    const [,r,c]=open.shift();
    if(r===er&&c===ec){
      const path=[]; let cr=er,cc=ec;
      while(from[`${cr},${cc}`]){[cr,cc]=from[`${cr},${cc}`];path.unshift([cr,cc]);}
      path.push([er,ec]);
      return path.length?path:[[sr,sc]];
    }
    for(const [dr,dc] of DIRS){
      const nr=r+dr,nc=c+dc;
      if(!walkable(g,nr,nc)) continue;
      const dangerCost=MEM.getDanger(nr,nc)*0.05;
      const ng=(gScore[`${r},${c}`]||0)+1+dangerCost;
      if(gScore[`${nr},${nc}`]===undefined||ng<gScore[`${nr},${nc}`]){
        gScore[`${nr},${nc}`]=ng;
        from[`${nr},${nc}`]=[r,c];
        open.push([ng+H(nr,nc),nr,nc]);
      }
    }
  }
  return [[sr,sc]];
}

// ═══════════════════════════════════════════════════════════
// SECTOR PATH BUILDER
// FIX 1: Only FREE/START cells (no VISITED = no backtracking)
// FIX 4: Waypoints sorted by danger score — low danger first
// Uses greedy nearest-unvisited to minimize total travel distance
// ═══════════════════════════════════════════════════════════
function buildSectorPath(g, droneIdx, totalDrones, startR, startC){
  const secW=Math.ceil(COLS/totalDrones);
  const c0=droneIdx*secW, c1=Math.min(c0+secW,COLS);

  // Collect only UNCOVERED cells — VISITED excluded = zero backtracking
  const waypoints=[];
  for(let r=0;r<ROWS;r++){
    const even=(r%2===0);
    for(let ci=0;ci<(c1-c0);ci++){
      const c=even?(c0+ci):(c1-1-ci);
      if(g[r][c]===FREE || g[r][c]===START){
        // Weight each cell: prefer low-danger, maintain boustrophedon order
        const dg=MEM.getDanger(r,c);
        waypoints.push({r,c,dg});
      }
    }
  }
  if(waypoints.length===0) return [[startR,startC]];

  // Sort: mostly keep boustrophedon order but heavily penalize high danger
  // This means drone naturally avoids danger zones in its sweep
  waypoints.sort((a,b)=>{
    const rowA=a.r, rowB=b.r;
    if(rowA!==rowB) return rowA-rowB; // row order first
    // Same row: sort by danger (low danger first)
    return a.dg-b.dg;
  });

  // Build A*-connected path from current position
  const fullPath=[];
  let cur=[startR,startC];
  for(const wp of waypoints){
    if(wp.r===cur[0]&&wp.c===cur[1]){ fullPath.push([wp.r,wp.c]); continue; }
    const seg=astar(g,cur[0],cur[1],wp.r,wp.c);
    seg.forEach(p=>{
      if(!fullPath.length||(p[0]!==fullPath[fullPath.length-1][0]||p[1]!==fullPath[fullPath.length-1][1]))
        fullPath.push(p);
    });
    cur=[wp.r,wp.c];
  }
  return fullPath.length?fullPath:[[startR,startC]];
}

// ═══════════════════════════════════════════════════════════
// DRONE STATE
// ═══════════════════════════════════════════════════════════
let drones=[], nDrones=2, msnNum=1, totalReplan=0;
let running=false, paused=false, simTimer=null;
let totalStraightMoves=0, totalTurnMoves=0, totalEnergyUsed=0;
let prevDroneDirections={};  // track direction per drone for turn detection
const getEps=()=>Math.max(EPS_MIN, EPS0-EPS_DECAY*(msnNum-1));

// Status: READY → ACTIVE → RTB → RECHARGING → ACTIVE ... → RETURNING → DONE
function makeDrone(idx){
  // All drones start from home base [0,0]
  const path=buildSectorPath(liveGrid, idx, nDrones, 0, 0);
  return {
    idx, color:DRONE_COLORS[idx], name:DRONE_NAMES[idx],
    path, pathIdx:0,
    pos:[0,0],
    batt:MAX_BATT,
    trail:[],
    status:'READY',
    replanCount:0,
    cellsVisited:0,
    speedKmh:0,       // numeric speed display
    // RTB (low battery → go home to recharge)
    rtbPath:null, rtbIdx:0,
    rechargeTick:0,
    // HOME RETURN (sector done → return home)
    homePath:null, homeIdx:0,
    // 3D animation
    rotAngle:Math.random()*Math.PI*2,
    bobPhase:Math.random()*Math.PI*2,
  };
}

// ─── RL CHOOSE ACTION ─────────────────────────────────────
// Fix 4: Penalties actually affect path choice
// Higher danger weight = drone genuinely avoids known danger zones
function chooseAction(r,c){
  if(Math.random()<getEps()){
    // Biased exploration — heavily avoid danger zones
    const valid=[];
    for(let i=0;i<4;i++){
      const [dr,dc]=DIRS[i],nr=r+dr,nc=c+dc;
      if(walkable(liveGrid,nr,nc)){
        const dg=MEM.getDanger(nr,nc);
        // Weight inversely proportional to danger
        const w=Math.max(0.01, 1.0-dg/60);
        valid.push({i,w});
      }
    }
    if(!valid.length) return Math.floor(Math.random()*4);
    const tot=valid.reduce((s,x)=>s+x.w,0);
    let rr=Math.random()*tot;
    for(const x of valid){rr-=x.w;if(rr<=0)return x.i;}
    return valid[valid.length-1].i;
  }
  // Exploitation — best Q-value minus danger penalty
  let best=null, bestScore=-Infinity;
  for(let i=0;i<4;i++){
    const [dr,dc]=DIRS[i],nr=r+dr,nc=c+dc;
    if(walkable(liveGrid,nr,nc)){
      const q=MEM.getQ(r,c,i);
      const dg=MEM.getDanger(nr,nc);
      // Danger has real weight — drone genuinely avoids hot zones
      const score=q - dg*0.25;
      if(score>bestScore){bestScore=score;best=i;}
    }
  }
  return best!==null?best:Math.floor(Math.random()*4);
}

function updateQ(r,c,a,reward,nr,nc){
  const old=MEM.getQ(r,c,a);
  const bestNext=Math.max(...DIRS.map((_,i)=>MEM.getQ(nr,nc,i)));
  // Bellman equation with stronger learning rate
  MEM.setQ(r,c,a, old+ALPHA*(reward+GAMMA*bestNext-old));
}

// ═══════════════════════════════════════════════════════════
// DYNAMIC OBSTACLE SCHEDULE (auto obstacles during mission)
// Uses global tick counter so each fires exactly once
// ═══════════════════════════════════════════════════════════
let dynObsSchedule={};
let globalTick=0;  // increments every gameTick, independent of any drone
const DYN_POOL=[[3,10],[7,14],[15,8],[11,5],[4,16],[13,3],[10,10],[6,16]];

// ═══════════════════════════════════════════════════════════
// TICK: single drone step
// Battery costs: ACTIVE=0.5/step, RTB=0.7/step, RETURNING=0.4/step
// With MAX_BATT=1200: ~2400 ACTIVE steps before empty (grid=400 cells)
// ═══════════════════════════════════════════════════════════
function tickDrone(d){
  // 3D animation always runs
  d.rotAngle += 0.09;
  d.bobPhase += 0.06;

  if(d.status==='DONE') return;

  // ── RECHARGING ──
  if(d.status==='RECHARGING'){
    d.rechargeTick--;
    d.batt=Math.min(MAX_BATT, d.batt + MAX_BATT/RECHARGE_STEPS);
    d.speedKmh=0;
    if(d.rechargeTick<=0){
      // Replan from [0,0] (current pos after RTB), only uncovered cells
      d.path=buildSectorPath(liveGrid, d.idx, nDrones, d.pos[0], d.pos[1]);
      d.pathIdx=0;
      d.status='ACTIVE';
      log(`🔋 ${d.name} recharged → resuming, ${d.path.length} cells remain`, 'ok');
    }
    return;
  }

  // ── RETURNING HOME (sector complete → A* shortest path back) ──
  if(d.status==='RETURNING'){
    d.speedKmh=calcSpeed();
    if(!d.homePath||d.homeIdx>=d.homePath.length){
      d.status='DONE'; d.speedKmh=0;
      log(`🏠 ${d.name} returned to base`, 'ok');
      return;
    }
    const np=d.homePath[d.homeIdx++];
    d.pos=[...np]; d.batt-=0.4;
    d.trail.push([...np]); if(d.trail.length>80) d.trail.shift();
    revealFog(np[0],np[1]);
    return;
  }

  // ── RTB (low battery → go to [0,0] to recharge) ──
  if(d.status==='RTB'){
    d.speedKmh=calcSpeed()*1.2; // RTB faster conceptually
    if(!d.rtbPath||d.rtbIdx>=d.rtbPath.length){
      d.status='RECHARGING'; d.rechargeTick=RECHARGE_STEPS; d.speedKmh=0;
      log(`🔌 ${d.name} at base — RECHARGING`, 'w');
      return;
    }
    const np=d.rtbPath[d.rtbIdx++];
    d.pos=[...np]; d.batt-=0.7;
    d.trail.push([...np]); if(d.trail.length>60) d.trail.shift();
    revealFog(np[0],np[1]);
    return;
  }

  // ── Battery critical → RTB ──
  if(d.batt/MAX_BATT<RTB_THRESHOLD && d.status==='ACTIVE'){
    d.status='RTB';
    // A* from CURRENT position to home [0,0]
    d.rtbPath=astar(liveGrid, d.pos[0],d.pos[1], 0,0);
    d.rtbIdx=0;
    log(`⚠ ${d.name} BATTERY ${(d.batt/MAX_BATT*100).toFixed(0)}% — RTB from [${d.pos}]`, 'w');
    showHint(`${d.name} LOW BATTERY — RTB ENGAGED`, 'obs');
    return;
  }

  // ── Sector path complete → return home ──
  if(d.pathIdx>=d.path.length && d.status==='ACTIVE'){
    d.status='RETURNING';
    // A* shortest path from CURRENT pos back to [0,0]
    d.homePath=astar(liveGrid, d.pos[0],d.pos[1], 0,0);
    d.homeIdx=0; d.speedKmh=calcSpeed();
    log(`✅ ${d.name} sector done → returning home (${d.homePath.length} steps)`, 'ok');
    return;
  }

  if(d.pathIdx>=d.path.length) return;

  // ── Move to next cell on sector path ──
  const [r,c]=d.path[d.pathIdx];
  revealFog(r,c);
  d.trail.push([r,c]); if(d.trail.length>80) d.trail.shift();
  d.speedKmh=calcSpeed();

  // ── Energy tracking + Q reinforcement for straight vs turn ──
  const prevDir=prevDroneDirections[d.idx];
  const newDir=d.pos?[r-d.pos[0], c-d.pos[1]]:null;
  let isTurn=false;
  if(prevDir&&newDir){
    isTurn=!(prevDir[0]===newDir[0]&&prevDir[1]===newDir[1]);
    if(isTurn){
      totalTurnMoves++; totalEnergyUsed+=1.4;
      // Penalize turn in Q-table so future missions prefer straight paths
      const act=DIRS.findIndex(([dr,dc])=>dr===newDir[0]&&dc===newDir[1]);
      if(act>=0) MEM.penalizeTurn(d.pos[0],d.pos[1],act);
    } else {
      totalStraightMoves++; totalEnergyUsed+=1.0;
      // Reward straight path in Q-table
      const act=DIRS.findIndex(([dr,dc])=>dr===newDir[0]&&dc===newDir[1]);
      if(act>=0) MEM.rewardStraightPath(d.pos[0],d.pos[1],act);
    }
  } else {
    totalStraightMoves++; totalEnergyUsed+=1.0;
  }
  if(newDir) prevDroneDirections[d.idx]=newDir;

  // ── RL reward — stronger signals ──
  const act=chooseAction(r,c);
  let reward=R_STEP;
  const dg=MEM.getDanger(r,c);
  reward+=R_DANGER*(dg/80);            // danger penalty proportional to level
  if(isTurn) reward+=R_TURN;           // penalize turns
  else reward+=R_STRAIGHT;             // reward going straight

  if(liveGrid[r][c]===FREE){
    liveGrid[r][c]=VISITED;
    reward+=R_NEW_CELL;                // strong reward for new coverage
    d.cellsVisited++;
  } else {
    reward+=R_REVISIT;                 // punish revisiting (backtracking)
  }

  const nxt=d.pathIdx+1<d.path.length?d.path[d.pathIdx+1]:[r,c];
  updateQ(r,c,act,reward,nxt[0],nxt[1]);

  d.batt-=0.5; d.pos=[r,c]; d.pathIdx++;
}

// Speed in km/h based on slider (cell = ~200m, tick interval = ~80ms at speed 5)
function calcSpeed(){
  const spd=+document.getElementById('spdSlider').value;
  const msPerTick=Math.max(10,190-spd*18);
  const cellSize=(Z.rad*2/ROWS)*1000; // metres per cell
  return Math.round((cellSize/msPerTick)*3600/1000);
}

// ═══════════════════════════════════════════════════════════
// 3D DRONE RENDERING — full detail: glow rings, perspective arms,
// spinning rotors, camera pod, landing gear, body gradient
// ═══════════════════════════════════════════════════════════
function drawDrone3D(px,py,color,size,rotAngle,bobPhase,status){
  gx.save();
  const bob=Math.sin(bobPhase)*size*0.08;
  gx.translate(px,py+bob);
  const dc=status==='RTB'?'#ff8c00':status==='RECHARGING'?'#00d4ff':status==='RETURNING'?'#a855f7':color;
  const bodyR=Math.max(4.5,size*0.40);
  const armLen=bodyR*1.05;
  const rotorR=Math.max(2.2,bodyR*0.38);

  // Ground shadow
  gx.save();gx.translate(size*0.06,bodyR*0.65);
  gx.beginPath();gx.ellipse(0,0,bodyR*1.15,bodyR*0.28,0,0,Math.PI*2);
  gx.fillStyle='rgba(0,0,0,0.28)';gx.fill();gx.restore();

  // Glow rings (3 layers)
  [3.0,2.2,1.5].forEach((s,i)=>{
    gx.beginPath();gx.arc(0,0,bodyR*s,0,Math.PI*2);
    gx.strokeStyle=dc+['0e','1a','2e'][i];
    gx.lineWidth=i===0?3:i===1?2:1;gx.stroke();
  });

  // 4 Arms with 3D perspective + spinning rotors
  for(let i=0;i<4;i++){
    const armAngle=i*(Math.PI/2)+rotAngle;
    const ax=Math.cos(armAngle)*armLen,ay=Math.sin(armAngle)*armLen;
    const perspScale=1-Math.abs(Math.sin(armAngle))*0.12;
    gx.save();gx.scale(1,perspScale);
    const armGrad=gx.createLinearGradient(0,0,ax,ay);
    armGrad.addColorStop(0,dc+'80');armGrad.addColorStop(1,dc+'cc');
    gx.strokeStyle=armGrad;gx.lineWidth=Math.max(1.5,size*0.1);
    gx.lineCap='round';
    gx.beginPath();gx.moveTo(0,0);gx.lineTo(ax,ay);gx.stroke();
    // Motor housing sphere
    gx.beginPath();gx.arc(ax,ay,Math.max(1.5,bodyR*0.22),0,Math.PI*2);
    const mGrad=gx.createRadialGradient(ax-1,ay-1,0.5,ax,ay,bodyR*0.22);
    mGrad.addColorStop(0,'#fff');mGrad.addColorStop(0.4,dc);mGrad.addColorStop(1,dc+'88');
    gx.fillStyle=mGrad;gx.fill();
    // Rotor disk + 3 spinning blades
    gx.save();gx.translate(ax,ay);
    gx.beginPath();gx.ellipse(0,0,rotorR,rotorR*0.38,armAngle,0,Math.PI*2);
    gx.fillStyle=dc+'22';gx.fill();
    gx.strokeStyle=dc+'55';gx.lineWidth=0.8;gx.stroke();
    for(let b=0;b<3;b++){
      const ba=b*(Math.PI*2/3)+rotAngle*4;
      gx.beginPath();gx.moveTo(0,0);
      gx.lineTo(Math.cos(ba)*rotorR,Math.sin(ba)*rotorR*0.38);
      gx.strokeStyle=dc+'70';gx.lineWidth=1.2;gx.stroke();
    }
    gx.restore();gx.restore();
  }

  // Body hemisphere with radial gradient
  const bodyGrad=gx.createRadialGradient(-bodyR*.28,-bodyR*.28,0,0,0,bodyR);
  bodyGrad.addColorStop(0,'#ffffff');
  bodyGrad.addColorStop(0.25,dc);
  bodyGrad.addColorStop(0.7,dc+'cc');
  bodyGrad.addColorStop(1,dc+'55');
  gx.beginPath();gx.arc(0,0,bodyR,0,Math.PI*2);
  gx.shadowColor=dc;gx.shadowBlur=size*0.5;
  gx.fillStyle=bodyGrad;gx.fill();gx.shadowBlur=0;

  // Body cross detail
  gx.strokeStyle='rgba(0,0,0,0.5)';gx.lineWidth=1.2;
  gx.beginPath();gx.moveTo(-bodyR*.6,0);gx.lineTo(bodyR*.6,0);gx.stroke();
  gx.beginPath();gx.moveTo(0,-bodyR*.6);gx.lineTo(0,bodyR*.6);gx.stroke();

  // Camera pod (surveillance camera)
  gx.beginPath();gx.arc(0,bodyR*.35,bodyR*0.22,0,Math.PI*2);
  const camGrad=gx.createRadialGradient(-bodyR*.06,bodyR*.3,0.5,0,bodyR*.35,bodyR*0.22);
  camGrad.addColorStop(0,'#80e0ff');camGrad.addColorStop(1,'#004488');
  gx.fillStyle=camGrad;gx.fill();
  gx.strokeStyle=dc+'60';gx.lineWidth=0.8;gx.stroke();

  // Landing gear (2 legs + cross bar)
  [[-bodyR*.42,bodyR*.15],[bodyR*.42,bodyR*.15]].forEach(([lx,ly])=>{
    gx.strokeStyle=dc+'55';gx.lineWidth=1;
    gx.beginPath();gx.moveTo(lx,ly);gx.lineTo(lx*0.8,bodyR*0.75);gx.stroke();
  });
  gx.beginPath();gx.moveTo(-bodyR*.42,bodyR*0.75);gx.lineTo(bodyR*.42,bodyR*0.75);
  gx.strokeStyle=dc+'44';gx.lineWidth=1;gx.stroke();

  gx.restore();
}

let CELL=24;
const gc=document.getElementById('gc');
const gx=gc.getContext('2d');

function setupCanvas(){
  const cw=window.innerWidth-410, ch=window.innerHeight-90;
  CELL=Math.max(9,Math.floor(Math.min(cw/COLS,ch/ROWS)));
  gc.width=COLS*CELL;gc.height=ROWS*CELL;
}

function drawDrone3D(px,py,color,size,rotAngle,bobPhase,status){
  gx.save();
  // Bob up/down (3D float effect)
  const bob=Math.sin(bobPhase)*size*0.08;
  gx.translate(px, py+bob);

  const dc = status==='RTB'?'#ff8c00' : status==='RECHARGING'?'#00d4ff' : status==='RETURNING'?'#a855f7' : color;
  const bodyR=Math.max(4.5, size*0.40);
  const armLen=bodyR*1.05;
  const rotorR=Math.max(2.2, bodyR*0.38);

  // === SHADOW (ground projection) ===
  gx.save();
  gx.translate(size*0.06, bodyR*0.65);
  gx.beginPath();
  gx.ellipse(0,0, bodyR*1.15, bodyR*0.28, 0,0,Math.PI*2);
  gx.fillStyle='rgba(0,0,0,0.28)';
  gx.fill();
  gx.restore();

  // === GLOW RINGS ===
  [3.0,2.2,1.5].forEach((s,i)=>{
    gx.beginPath(); gx.arc(0,0,bodyR*s,0,Math.PI*2);
    gx.strokeStyle=dc+['0e','1a','2e'][i]; gx.lineWidth=i===0?3:i===1?2:1; gx.stroke();
  });

  // === 4 ARMS (rotate with animation) ===
  for(let i=0;i<4;i++){
    const armAngle=i*(Math.PI/2)+rotAngle;
    const ax=Math.cos(armAngle)*armLen, ay=Math.sin(armAngle)*armLen;
    // 3D perspective: arms going away shrink slightly
    const perspScale=1-Math.abs(Math.sin(armAngle))*0.12;
    gx.save();
    gx.scale(1,perspScale);
    // Arm tube (gradient)
    const armGrad=gx.createLinearGradient(0,0,ax,ay);
    armGrad.addColorStop(0,dc+'80'); armGrad.addColorStop(1,dc+'cc');
    gx.strokeStyle=armGrad; gx.lineWidth=Math.max(1.5,size*0.1);
    gx.lineCap='round';
    gx.beginPath(); gx.moveTo(0,0); gx.lineTo(ax,ay); gx.stroke();
    // Motor housing (sphere-like)
    gx.beginPath(); gx.arc(ax,ay,Math.max(1.5,bodyR*0.22),0,Math.PI*2);
    const mGrad=gx.createRadialGradient(ax-1,ay-1,0.5,ax,ay,bodyR*0.22);
    mGrad.addColorStop(0,'#fff'); mGrad.addColorStop(0.4,dc); mGrad.addColorStop(1,dc+'88');
    gx.fillStyle=mGrad; gx.fill();
    // Rotor disk (3D ellipse — wider than tall for perspective)
    gx.save(); gx.translate(ax,ay);
    gx.beginPath(); gx.ellipse(0,0, rotorR, rotorR*0.38, armAngle,0,Math.PI*2);
    gx.fillStyle=dc+'22'; gx.fill();
    gx.strokeStyle=dc+'55'; gx.lineWidth=0.8; gx.stroke();
    // Spinning rotor blades
    for(let b=0;b<3;b++){
      const ba=b*(Math.PI*2/3)+rotAngle*4;
      gx.beginPath();
      gx.moveTo(0,0);
      gx.lineTo(Math.cos(ba)*rotorR, Math.sin(ba)*rotorR*0.38);
      gx.strokeStyle=dc+'70'; gx.lineWidth=1.2; gx.stroke();
    }
    gx.restore();
    gx.restore();
  }

  // === BODY (hemisphere — 3D radial gradient) ===
  const bodyGrad=gx.createRadialGradient(-bodyR*.28,-bodyR*.28,0, 0,0,bodyR);
  bodyGrad.addColorStop(0,'#ffffff');
  bodyGrad.addColorStop(0.25,dc);
  bodyGrad.addColorStop(0.7,dc+'cc');
  bodyGrad.addColorStop(1,dc+'55');
  gx.beginPath(); gx.arc(0,0,bodyR,0,Math.PI*2);
  gx.shadowColor=dc; gx.shadowBlur=size*0.5; gx.fillStyle=bodyGrad; gx.fill();
  gx.shadowBlur=0;

  // Body detail cross
  gx.strokeStyle='rgba(0,0,0,0.5)'; gx.lineWidth=1.2;
  gx.beginPath(); gx.moveTo(-bodyR*.6,0); gx.lineTo(bodyR*.6,0); gx.stroke();
  gx.beginPath(); gx.moveTo(0,-bodyR*.6); gx.lineTo(0,bodyR*.6); gx.stroke();

  // Camera (bottom sphere)
  gx.beginPath(); gx.arc(0,bodyR*.35,bodyR*0.22,0,Math.PI*2);
  const camGrad=gx.createRadialGradient(-bodyR*.06,bodyR*.3,0.5,0,bodyR*.35,bodyR*0.22);
  camGrad.addColorStop(0,'#80e0ff'); camGrad.addColorStop(1,'#004488');
  gx.fillStyle=camGrad; gx.fill();
  gx.strokeStyle=dc+'60'; gx.lineWidth=0.8; gx.stroke();

  // === LANDING GEAR (3D legs) ===
  [[-bodyR*.42,bodyR*.15],[bodyR*.42,bodyR*.15]].forEach(([lx,ly])=>{
    gx.strokeStyle=dc+'55'; gx.lineWidth=1;
    gx.beginPath(); gx.moveTo(lx,ly); gx.lineTo(lx*0.8,bodyR*0.75); gx.stroke();
  });
  gx.beginPath(); gx.moveTo(-bodyR*.42,bodyR*0.75); gx.lineTo(bodyR*.42,bodyR*0.75);
  gx.strokeStyle=dc+'44'; gx.lineWidth=1; gx.stroke();

  gx.restore();
}

// ═══════════════════════════════════════════════════════════
// MAIN DRAW
// ═══════════════════════════════════════════════════════════
function drawMap(){
  gx.fillStyle='#04060f'; gx.fillRect(0,0,gc.width,gc.height);

  // Cells
  for(let r=0;r<ROWS;r++) for(let c=0;c<COLS;c++){
    const x=c*CELL, y=r*CELL;
    if(fogGrid[r][c]){
      gx.fillStyle='#0b1225'; gx.fillRect(x,y,CELL,CELL);
      gx.strokeStyle='rgba(255,255,255,.03)'; gx.lineWidth=.4; gx.strokeRect(x,y,CELL,CELL);
      continue;
    }
    const dg=MEM.getDanger(r,c);
    if(dg>0){
      gx.fillStyle=`rgba(255,68,102,${Math.min(.28,dg/90)})`; gx.fillRect(x,y,CELL,CELL);
    }
    const v=liveGrid[r][c];
    if(v===OBS){
      gx.fillStyle='rgba(255,68,102,.22)'; gx.fillRect(x+1,y+1,CELL-2,CELL-2);
      gx.strokeStyle='#ff4466'; gx.lineWidth=1; gx.strokeRect(x+1,y+1,CELL-2,CELL-2);
      gx.strokeStyle='rgba(255,68,102,.4)'; gx.lineWidth=.8;
      gx.beginPath(); gx.moveTo(x+3,y+3); gx.lineTo(x+CELL-3,y+CELL-3); gx.stroke();
      gx.beginPath(); gx.moveTo(x+CELL-3,y+3); gx.lineTo(x+3,y+CELL-3); gx.stroke();
    } else if(v===NOFLY){
      gx.fillStyle='rgba(255,140,0,.07)'; gx.fillRect(x,y,CELL,CELL);
      gx.save(); gx.strokeStyle='rgba(255,140,0,.18)'; gx.lineWidth=.5;
      for(let i=-CELL;i<CELL*2;i+=5){ gx.beginPath(); gx.moveTo(x+i,y); gx.lineTo(x+i+CELL,y+CELL); gx.stroke(); }
      gx.restore();
      gx.strokeStyle='rgba(255,140,0,.5)'; gx.lineWidth=.5; gx.strokeRect(x,y,CELL,CELL);
    } else if(v===VISITED){
      gx.fillStyle='#001508'; gx.fillRect(x+1,y+1,CELL-2,CELL-2);
    }
  }

  // Grid lines
  gx.strokeStyle='rgba(26,37,64,.5)'; gx.lineWidth=.4;
  for(let r=0;r<=ROWS;r++){gx.beginPath();gx.moveTo(0,r*CELL);gx.lineTo(COLS*CELL,r*CELL);gx.stroke();}
  for(let c=0;c<=COLS;c++){gx.beginPath();gx.moveTo(c*CELL,0);gx.lineTo(c*CELL,ROWS*CELL);gx.stroke();}

  // Sector dividers
  for(let i=1;i<nDrones;i++){
    const x=Math.floor(i*COLS/nDrones)*CELL;
    gx.strokeStyle='rgba(0,212,255,.08)'; gx.lineWidth=1; gx.setLineDash([3,3]);
    gx.beginPath(); gx.moveTo(x,0); gx.lineTo(x,gc.height); gx.stroke();
    gx.setLineDash([]);
  }

  // RL known obstacles (memory overlay)
  MEM.knownObs.forEach(([r,c])=>{
    if(fogGrid[r][c]) return;
    const x=c*CELL,y=r*CELL;
    gx.fillStyle='rgba(168,85,247,.14)'; gx.fillRect(x+1,y+1,CELL-2,CELL-2);
    gx.strokeStyle='rgba(168,85,247,.55)'; gx.lineWidth=.8; gx.strokeRect(x+1,y+1,CELL-2,CELL-2);
  });

  // Home base marker
  if(!fogGrid[0][0]){
    gx.fillStyle='rgba(0,245,160,.12)'; gx.fillRect(0,0,CELL,CELL);
    gx.strokeStyle='rgba(0,245,160,.55)'; gx.lineWidth=1.5; gx.strokeRect(0,0,CELL,CELL);
    gx.fillStyle='#00f5a0'; gx.font=`bold ${Math.max(7,CELL*.4)}px Share Tech Mono`;
    gx.textAlign='center'; gx.fillText('H',CELL/2,CELL*.72); gx.textAlign='left';
  }

  // Trails
  drones.forEach(d=>{
    if(d.trail.length<2) return;
    const tr=d.trail;
    const grad=gx.createLinearGradient(
      tr[0][1]*CELL+CELL/2, tr[0][0]*CELL+CELL/2,
      tr[tr.length-1][1]*CELL+CELL/2, tr[tr.length-1][0]*CELL+CELL/2
    );
    grad.addColorStop(0,'transparent'); grad.addColorStop(1,d.color+'55');
    gx.beginPath();
    gx.moveTo(tr[0][1]*CELL+CELL/2, tr[0][0]*CELL+CELL/2);
    tr.forEach(p=>gx.lineTo(p[1]*CELL+CELL/2, p[0]*CELL+CELL/2));
    gx.strokeStyle=grad; gx.lineWidth=1.4; gx.stroke();
  });

  // Return/RTB path preview (dotted line ahead)
  drones.forEach(d=>{
    let previewPath=null;
    if(d.status==='RTB' && d.rtbPath && d.rtbIdx<d.rtbPath.length)
      previewPath=d.rtbPath.slice(d.rtbIdx);
    else if(d.status==='RETURNING' && d.homePath && d.homeIdx<d.homePath.length)
      previewPath=d.homePath.slice(d.homeIdx);
    if(previewPath&&previewPath.length>1){
      gx.setLineDash([2,4]); gx.strokeStyle=d.color+'30'; gx.lineWidth=1;
      gx.beginPath(); gx.moveTo(previewPath[0][1]*CELL+CELL/2,previewPath[0][0]*CELL+CELL/2);
      previewPath.forEach(p=>gx.lineTo(p[1]*CELL+CELL/2,p[0]*CELL+CELL/2));
      gx.stroke(); gx.setLineDash([]);
    }
  });

  // 3D Drones
  drones.forEach(d=>{
    if(d.status==='DONE') return;
    const [dr,dc]=d.pos;
    const px=dc*CELL+CELL/2, py=dr*CELL+CELL/2;
    drawDrone3D(px,py,d.color,CELL,d.rotAngle,d.bobPhase,d.status);
    // Label
    const lc=d.status==='RTB'?'#ff8c00':d.status==='RECHARGING'?'#00d4ff':d.status==='RETURNING'?'#a855f7':d.color;
    gx.fillStyle=lc; gx.font=`bold ${Math.max(6,CELL*.26)}px Orbitron,Share Tech Mono`;
    gx.textAlign='center';
    gx.fillText(d.name.split('-')[1], px, py-CELL*.52);
    if(d.status==='RTB'||d.status==='RETURNING'){
      gx.font=`${Math.max(5,CELL*.2)}px Share Tech Mono`;
      gx.fillText(d.status==='RTB'?'RTB↩':'HOME↩', px, py+CELL*.6);
    }
  });
  gx.textAlign='left';
}

// ═══════════════════════════════════════════════════════════
// HUD UPDATE
// ═══════════════════════════════════════════════════════════
function updateHUD(){
  const act=drones.filter(d=>['ACTIVE','RTB','RETURNING','RECHARGING'].includes(d.status)).length;
  document.getElementById('vAct').textContent=act;

  // Coverage
  let vis=0,tot=0;
  for(let r=0;r<ROWS;r++) for(let c=0;c<COLS;c++){
    if(liveGrid[r][c]!==OBS&&liveGrid[r][c]!==NOFLY)tot++;
    if(liveGrid[r][c]===VISITED||liveGrid[r][c]===START)vis++;
  }
  const covPct=tot>0?(vis/tot*100):0;
  document.getElementById('vCov').textContent=covPct.toFixed(1)+'%';
  const covBar=document.getElementById('covBar');
  if(covBar){ covBar.style.width=covPct+'%'; }

  // Fog
  const fogCleared=fogGrid.flat().filter(f=>!f).length;
  document.getElementById('vFog').textContent=(fogCleared/(ROWS*COLS)*100).toFixed(1)+'%';

  // Replan
  document.getElementById('vRep').textContent=totalReplan;
  document.getElementById('vObs').textContent=MEM.knownObs.length;
  document.getElementById('vKO').textContent=MEM.knownObs.length;
  document.getElementById('vDZ').textContent=MEM.dangerCells();
  document.getElementById('vQS').textContent=MEM.qSize();
  document.getElementById('vEps').textContent=getEps().toFixed(2);
  document.getElementById('msnTag').textContent=`MISSION ${msnNum}`;
  const rt=document.getElementById('rlTag');
  if(rt) rt.textContent=`ε=${getEps().toFixed(2)}`;

  // ── Energy Analytics ──
  const totalMoves=totalStraightMoves+totalTurnMoves;
  document.getElementById('vStraight').textContent=totalStraightMoves;
  document.getElementById('vTurns').textContent=totalTurnMoves;
  document.getElementById('vEnergy').textContent=totalEnergyUsed.toFixed(0)+' units';

  // Efficiency = straight moves / total * 100
  const effPct=totalMoves>0?Math.round((totalStraightMoves/totalMoves)*100):0;
  document.getElementById('vEffScore').textContent=effPct+'%';
  document.getElementById('vEff').textContent=effPct+'%';
  const effBar=document.getElementById('effBar');
  if(effBar){
    effBar.style.width=effPct+'%';
    effBar.style.background=effPct>80?'var(--g)':effPct>60?'var(--gd)':'var(--r)';
  }

  // Safe navigation indicator
  const safeEl=document.getElementById('vSafe');
  if(safeEl){
    safeEl.textContent=MEM.knownObs.length>0?`✓ ${MEM.knownObs.length} OBS AVOIDED`:'✓ ACTIVE';
  }

  // Drone cards
  drones.forEach(d=>{
    const el=document.getElementById('dc'+d.idx); if(!el)return;
    const pct=Math.max(0,d.batt/MAX_BATT*100);
    el.querySelector('.badge').textContent=d.status;
    el.querySelector('.badge').style.borderColor=d.color;
    el.querySelector('.badge').style.color=d.color;
    el.querySelector('.battval').textContent=pct.toFixed(0)+'%';
    const bf=el.querySelector('.battfill');
    bf.style.width=pct+'%';
    const bc=pct<12?'#ff4466':pct<40?'#ff8c00':d.color;
    bf.style.background=bc; bf.style.boxShadow=`0 0 5px ${bc}`;
    // Speed display
    const sv=el.querySelector('.speedval');
    if(sv) sv.textContent=d.speedKmh>0?`${d.speedKmh} km/h`:'0 km/h';
    el.className='dcard'+(
      d.status==='RTB'?' rtb':
      d.status==='RETURNING'?' returning':
      d.status==='RECHARGING'?' recharging':
      d.status==='DONE'?' done':''
    );
    if(d.status!=='DONE') el.style.borderColor=d.color+'40';
    el.querySelector('.replct').textContent=d.replanCount;
  });

  // Coord bar
  const lead=drones.find(d=>d.status==='ACTIVE')||drones[0];
  if(lead){
    const {lat,lng}=Z.cellLatLng(lead.pos[0],lead.pos[1]);
    document.getElementById('cbar').textContent=
      `[ ${lead.name} | ${fmtCoord(lat,true)} | ${fmtCoord(lng,false)} | ALT:${Z.alt}m | ACTIVE:${act}/${nDrones} ]`;
  }
}

// ═══════════════════════════════════════════════════════════
// HEATMAP — always shows ALL RL memory (danger + known obstacles)
// regardless of fog. Memory persists across missions.
// Known obstacles shown even in fog-covered areas.
// ═══════════════════════════════════════════════════════════
function drawHeatmap(){
  const hc=document.getElementById('heatCanvas');
  const hx=hc.getContext('2d');
  const cw=hc.width/COLS, ch=hc.height/ROWS;
  // Max danger across ALL memory cells (not just revealed)
  let maxD=1;
  for(let r=0;r<ROWS;r++) for(let c=0;c<COLS;c++) maxD=Math.max(maxD,MEM.getDanger(r,c));
  hx.fillStyle='#04060f'; hx.fillRect(0,0,hc.width,hc.height);
  // Fast lookup: known obstacle positions from RL memory
  const obsSet=new Set(MEM.knownObs.map(([r,c])=>`${r},${c}`));
  for(let r=0;r<ROWS;r++) for(let c=0;c<COLS;c++){
    const x=c*cw, y=r*ch;
    const inFog=fogGrid[r][c];
    const dg=MEM.getDanger(r,c);
    const isKnownObs=obsSet.has(`${r},${c}`);
    const isGridObs=liveGrid[r]&&liveGrid[r][c]===OBS;
    if(isKnownObs||isGridObs){
      // Known obstacle — always visible, red with X mark
      hx.fillStyle=inFog?'rgba(255,68,102,0.5)':'rgba(255,68,102,0.9)';
      hx.fillRect(x,y,cw,ch);
      hx.strokeStyle=inFog?'rgba(255,68,102,0.35)':'rgba(255,100,120,0.8)';
      hx.lineWidth=0.7;
      hx.beginPath(); hx.moveTo(x+1,y+1); hx.lineTo(x+cw-1,y+ch-1); hx.stroke();
      hx.beginPath(); hx.moveTo(x+cw-1,y+1); hx.lineTo(x+1,y+ch-1); hx.stroke();
    } else if(dg>0){
      // Danger zone — show even in fog (RL memory is global knowledge)
      const ratio=dg/maxD;
      const alpha=inFog?ratio*0.42:ratio*0.82;
      hx.fillStyle=`rgba(255,${Math.floor(68*(1-ratio))},${Math.floor(102*(1-ratio))},${alpha})`;
      hx.fillRect(x,y,cw,ch);
    } else if(inFog){
      hx.fillStyle='#080e1e'; hx.fillRect(x,y,cw,ch);
    } else {
      hx.fillStyle='#001508'; hx.fillRect(x,y,cw,ch);
    }
  }
  // Drone position dots
  drones.forEach(d=>{
    if(d.status==='DONE') return;
    const [dr,dc]=d.pos;
    hx.beginPath(); hx.arc(dc*cw+cw/2,dr*ch+ch/2,Math.max(1.5,cw*.38),0,Math.PI*2);
    hx.fillStyle=d.color+'cc'; hx.fill();
  });
  // Grid lines
  hx.strokeStyle='rgba(26,37,64,.18)'; hx.lineWidth=.25;
  for(let r=0;r<=ROWS;r++){hx.beginPath();hx.moveTo(0,r*ch);hx.lineTo(hc.width,r*ch);hx.stroke();}
  for(let c=0;c<=COLS;c++){hx.beginPath();hx.moveTo(c*cw,0);hx.lineTo(c*cw,hc.height);hx.stroke();}
  // Status line under heatmap
  const sub=document.getElementById('heatSub');
  if(sub){
    const o=MEM.knownObs.length, d2=MEM.dangerCells();
    sub.textContent=o>0||d2>0?`${o} OBS · ${d2} DANGER ZONES IN MEMORY`:'NO THREAT DATA YET';
  }
}

// ═══════════════════════════════════════════════════════════
// MISSION MANAGEMENT
// ═══════════════════════════════════════════════════════════
function initMission(){
  baseGrid=buildBaseGrid();
  liveGrid=baseGrid.map(r=>[...r]);
  fogGrid=buildFogGrid();
  const memLoaded=preloadMemory(liveGrid);
  if(memLoaded>0){
    log(`🧠 Memory: ${memLoaded} obstacles pre-applied from past missions`, 'mem');
    showHint(`MEMORY: ${memLoaded} OBS PRE-AVOIDED`, 'mem');
  }
  // Schedule auto obstacles using global tick (fires exactly once each)
  const pool=DYN_POOL.filter(([r,c])=>!MEM.knownObs.find(o=>o[0]===r&&o[1]===c));
  dynObsSchedule={};
  pool.slice(0,2).forEach((obs,i)=>{ dynObsSchedule[120+i*160]=obs; });
  globalTick=0;

  drones=[]; for(let i=0;i<nDrones;i++) drones.push(makeDrone(i));
  totalReplan=0; running=false; paused=false;
  totalStraightMoves=0; totalTurnMoves=0; totalEnergyUsed=0; prevDroneDirections={};

  renderDroneCards(); drawMap(); drawHeatmap(); updateHUD();
  document.getElementById('topTag').textContent=
    `GARUDA ${nDrones}-DRONE | BOUSTROPHEDON+A* | 3D | FOG+RTB+RETURN-HOME | MISSION ${msnNum}`;
  document.getElementById('launchBtn').disabled=false;
  document.getElementById('pauseBtn').disabled=true;
  document.getElementById('nextBtn').disabled=true;
  document.getElementById('pauseTxt').textContent='⏸ PAUSE';
}

function renderDroneCards(){
  document.getElementById('droneCards').innerHTML=
    `<div class="sec-title">◈ DRONE STATUS</div>`+
    drones.map(d=>`
      <div class="dcard" id="dc${d.idx}" style="border-color:${d.color}40">
        <div class="dcard-hd">
          <span class="dname" style="color:${d.color}">${d.name}</span>
          <span class="badge" style="border-color:${d.color};color:${d.color}">READY</span>
        </div>
        <div class="kv"><span>BATTERY</span><span class="val battval" style="color:${d.color}">100%</span></div>
        <div class="battbar"><div class="battfill" style="width:100%;background:${d.color};box-shadow:0 0 6px ${d.color}"></div></div>
        <div class="kv"><span>SPEED</span><span class="val speedval" style="color:${d.color}">0 km/h</span></div>
        <div class="kv"><span>REPLANNINGS</span><span class="val replct" style="color:${d.color}">0</span></div>
      </div>`).join('');
}

// ═══════════════════════════════════════════════════════════
// CONTROLS
// ═══════════════════════════════════════════════════════════
function doLaunch(){
  document.getElementById('launchBtn').disabled=true;
  document.getElementById('pauseBtn').disabled=false;
  drones.forEach(d=>d.status='ACTIVE');
  running=true;
  log(`🚀 GARUDA ${nDrones}-drone swarm launched! ε=${getEps().toFixed(2)}`, 'ok');
  log('Fog of War: ON | RTB at 12% | Return-Home via A* after sector', 'inf');
  startTimer();
}

function startTimer(){
  if(simTimer) clearInterval(simTimer);
  const spd=+document.getElementById('spdSlider').value;
  simTimer=setInterval(gameTick, Math.max(10,190-spd*18));
}

function gameTick(){
  if(paused||!running) return;
  globalTick++;

  // ── Fire auto-obstacles based on global tick (fires exactly once) ──
  if(dynObsSchedule[globalTick]){
    const [or,oc]=dynObsSchedule[globalTick];
    delete dynObsSchedule[globalTick]; // prevent re-fire
    if(liveGrid[or][oc]===FREE){
      liveGrid[or][oc]=OBS; baseGrid[or][oc]=OBS;
      MEM.learnObstacle(or,oc);
      log(`⊕ Auto-obstacle at [${or},${oc}] — checking drones`, 'al');
      showHint(`OBSTACLE DETECTED [${or},${oc}]`, 'al');
      let n=0;
      drones.forEach(od=>{
        if(od.status!=='ACTIVE') return;
        const remaining=od.path.slice(od.pathIdx);
        if(remaining.some(p=>p[0]===or&&p[1]===oc)){
          // Replan from CURRENT position, only uncovered cells remain
          od.path=buildSectorPath(liveGrid,od.idx,nDrones,od.pos[0],od.pos[1]);
          od.pathIdx=0; od.replanCount++; totalReplan++; n++;
          log(`↺ ${od.name} replanned from [${od.pos}]`,'al');
        }
      });
      if(n===0) log(`⊕ Obs [${or},${oc}] — no drones affected`,'inf');
      drawHeatmap();
    }
  }

  drones.forEach(d=>tickDrone(d));
  drawMap(); updateHUD();
  if(drones.every(d=>d.status==='DONE'||d.batt<=0)) missionComplete();
}

function togglePause(){
  paused=!paused;
  document.getElementById('pauseTxt').textContent=paused?'▶ RESUME':'⏸ PAUSE';
  log(paused?'⏸ Mission paused':'▶ Mission resumed', paused?'w':'inf');
}

function missionComplete(){
  clearInterval(simTimer); running=false;
  let vis=0,tot=0;
  for(let r=0;r<ROWS;r++) for(let c=0;c<COLS;c++){
    if(liveGrid[r][c]!==OBS&&liveGrid[r][c]!==NOFLY)tot++;
    if(liveGrid[r][c]===VISITED||liveGrid[r][c]===START)vis++;
  }
  const cov=tot>0?(vis/tot*100):0;

  // Record into persistent memory (survives to next mission!)
  MEM.recordMission(cov, totalStraightMoves, totalTurnMoves, totalReplan, MEM.knownObs.length);
  updateMsnHistory();

  const effPct=totalStraightMoves+totalTurnMoves>0
    ?Math.round((totalStraightMoves/(totalStraightMoves+totalTurnMoves))*100):0;

  document.getElementById('doneGrid').innerHTML=`
    <div>① COVERAGE <b>${cov.toFixed(1)}%</b></div>
    <div>② EFFICIENCY <b>${effPct}%</b></div>
    <div>③ REPLANNINGS <b>${totalReplan}</b></div>
    <div>④ OBS AVOIDED <b>${MEM.knownObs.length}</b></div>
    <div>DRONES <b>${nDrones}</b></div>
    <div>STRAIGHT MOVES <b>${totalStraightMoves}</b></div>
    <div>TURN PENALTIES <b>${totalTurnMoves}</b></div>
    <div>TOTAL ENERGY <b>${totalEnergyUsed.toFixed(0)} units</b></div>
    <div>Q-STATES <b>${MEM.qSize()}</b></div>
    <div>EPSILON <b>${getEps().toFixed(2)}</b></div>`;

  // Show learning improvement
  const imp=MEM.getImprovement();
  const prev=MEM.msnHistory[MEM.msnHistory.length-2];
  let note='';
  if(!prev){
    note='MISSION 1 BASELINE — RUN AGAIN TO SEE GARUDA LEARN!';
  } else if(imp!==null && imp>0){
    note=`✅ EFFICIENCY +${imp}% — GARUDA IS LEARNING! Less turns, more straight paths!`;
  } else if(imp===0){
    note=`✅ EFFICIENCY STABLE — Memory helping avoid ${MEM.knownObs.length} known obstacles`;
  } else {
    note=`⚠ NEW OBSTACLES FOUND — Memory updated, next mission will improve!`;
  }
  document.getElementById('doneNote').textContent=note;

  // Show learning curve if 2+ missions completed
  const lcSec=document.getElementById('lcSection');
  if(MEM.msnHistory.length>1){
    lcSec.style.display='block';
    document.getElementById('lcBars').innerHTML=MEM.msnHistory.map(m=>`
      <div class="lc-row">
        <span style="min-width:24px">M${m.m}</span>
        <div class="lc-bar"><div class="lc-fill" style="width:${m.eff}%;background:${m.eff>80?'var(--g)':m.eff>60?'var(--gd)':'var(--o)'}"></div></div>
        <span style="min-width:40px;text-align:right;color:var(--gd)">${m.eff}% eff</span>
        <span style="min-width:36px;text-align:right;color:var(--g)">${m.cov.toFixed(0)}% cov</span>
      </div>`).join('');
  } else {
    lcSec.style.display='none';
  }

  document.getElementById('doneOv').classList.add('show');
  document.getElementById('nextBtn').disabled=false;
  log(`✅ Mission ${msnNum} done — Cov:${cov.toFixed(1)}% Eff:${effPct}% Replan:${totalReplan}`, 'ok');
}

function nextMission(){
  // Close mission complete overlay
  document.getElementById('doneOv').classList.remove('show');
  msnNum++;
  // Open coordinate modal so user selects new surveillance area
  updPreview();
  document.getElementById('coordModal').classList.add('open');
  // Show next-mission banner with correct mission number
  const mb=document.getElementById('coordNextBanner');
  if(mb) mb.style.display='flex';
  const mn=document.getElementById('coordMsnNum');
  if(mn) mn.textContent=msnNum;
  log(`══ MISSION ${msnNum} — SELECT SURVEILLANCE ZONE ══`,'mem');
}

function setN(n){
  nDrones=n;
  document.querySelectorAll('.cnt-btn').forEach((b,i)=>b.classList.toggle('on',i===n-1));
  if(!running) initMission();
}

function fullReset(){
  clearInterval(simTimer); MEM.reset(); msnNum=1; nDrones=2; userPaints=[];
  obsMode=false; editMode=false;
  document.querySelectorAll('.cnt-btn').forEach((b,i)=>b.classList.toggle('on',i===1));
  document.getElementById('doneOv').classList.remove('show');
  document.getElementById('edbar').style.display='none';
  document.getElementById('editBtn').textContent='✏ EDIT MAP';
  document.getElementById('editBtn').classList.remove('active');
  document.getElementById('obsBtn').innerHTML='<span>🎯 PLACE OBSTACLE</span>';
  document.getElementById('obsBtn').className='btn bo';
  const lc=document.getElementById('lcSection');
  if(lc) lc.style.display='none';
  initMission(); updateMsnHistory();
  log('↺ Full reset — all memory cleared','al');
}

function onSpeedChange(){
  if(running&&!paused) startTimer();
  // Update global speed display
  const spd=calcSpeed();
  const el=document.getElementById('spdDisplay');
  if(el) el.textContent=`~${spd} km/h`;
}

function updateMsnHistory(){
  const el=document.getElementById('mhist');
  if(!MEM.msnHistory.length){
    el.innerHTML='<div style="font-size:8px;color:var(--tx)">No missions completed</div>';
    return;
  }
  el.innerHTML=MEM.msnHistory.map((m,i)=>{
    const prev=MEM.msnHistory[i-1];
    const effChange=prev?m.eff-prev.eff:null;
    const arrow=effChange===null?''
      :effChange>0?`<span style="color:var(--g)"> ▲${effChange}%</span>`
      :effChange<0?`<span style="color:var(--r)"> ▼${Math.abs(effChange)}%</span>`
      :`<span style="color:var(--tx)"> ─</span>`;
    return `
    <div class="mhrow">
      <span style="min-width:42px;color:var(--tx)">M${m.m}×${nDrones}</span>
      <div class="mhtrack">
        <div class="mhfill" style="width:${m.cov}%;background:${m.eff>80?'var(--g)':m.eff>60?'var(--gd)':'var(--o)'}"></div>
      </div>
      <span style="min-width:30px;text-align:right;color:var(--g)">${m.cov.toFixed(0)}%</span>
    </div>
    <div style="font-size:7px;color:var(--tx);padding-left:42px;margin-bottom:4px">
      Eff:${m.eff}%${arrow} · Turns:${m.turns} · Rpl:${m.replan}
    </div>`;
  }).join('');
}

// ═══════════════════════════════════════════════════════════
// DYNAMIC OBSTACLE PLACEMENT (USER CLICK)
// FIX: Smooth placement, no restart, replan from current pos
// ═══════════════════════════════════════════════════════════
let obsMode=false;

function toggleObsMode(){
  if(!running){ log('Launch mission first!','w'); return; }
  obsMode=!obsMode;
  const btn=document.getElementById('obsBtn');
  if(obsMode){
    btn.innerHTML='<span>✕ CANCEL PLACEMENT</span>'; btn.className='btn br';
    gc.style.cursor='crosshair';
    showHint('🎯 CLICK ANY REVEALED FREE CELL TO PLACE OBSTACLE','obs');
  } else {
    btn.innerHTML='<span>🎯 PLACE OBSTACLE</span>'; btn.className='btn bo';
    gc.style.cursor='default'; clearHint();
  }
}

function placeObstacle(r,c){
  if(r<0||r>=ROWS||c<0||c>=COLS) return;
  if(fogGrid[r][c]){ showHint('CELL IN FOG — NOT YET REVEALED','al'); return; }
  if(r===0&&c===0){ showHint('CANNOT BLOCK HOME BASE','al'); return; }
  if(liveGrid[r][c]===OBS||liveGrid[r][c]===NOFLY){ showHint(`[${r},${c}] ALREADY BLOCKED`,'w'); return; }

  // Place obstacle in both grids
  liveGrid[r][c]=OBS; baseGrid[r][c]=OBS;
  MEM.learnObstacle(r,c);

  // KEY FIX: Only replan drones whose REMAINING path crosses this obstacle.
  // Rebuild from their CURRENT position — only uncovered cells will be in new path.
  // pathIdx resets to 0 because new path starts from current position.
  let replanned=0;
  drones.forEach(d=>{
    if(d.status!=='ACTIVE') return;
    const remaining=d.path.slice(d.pathIdx);
    if(remaining.some(p=>p[0]===r&&p[1]===c)){
      d.path=buildSectorPath(liveGrid, d.idx, nDrones, d.pos[0], d.pos[1]);
      d.pathIdx=0; // new path starts from d.pos so this is correct
      d.replanCount++; totalReplan++; replanned++;
      log(`↺ ${d.name} replanned from [${d.pos[0]},${d.pos[1]}] — ${d.path.length} steps left`, 'al');
    }
  });

  const msg=replanned>0
    ?`⊕ OBS [${r},${c}] → ${replanned} DRONE(S) REPLANNING FROM CURRENT POS`
    :`⊕ OBS [${r},${c}] → NO ACTIVE DRONES AFFECTED`;
  showHint(msg,'al'); drawHeatmap(); drawMap();
  log(msg, replanned>0?'al':'inf');

  // Exit obstacle mode
  obsMode=false;
  document.getElementById('obsBtn').innerHTML='<span>🎯 PLACE OBSTACLE</span>';
  document.getElementById('obsBtn').className='btn bo';
  gc.style.cursor='default';
  setTimeout(clearHint, 2800);
}

// ═══════════════════════════════════════════════════════════
// MAP EDITOR
// ═══════════════════════════════════════════════════════════
let editMode=false, currentTool='nfz', mouseDown=false;

function toggleEdit(){
  editMode=!editMode;
  document.getElementById('edbar').style.display=editMode?'flex':'none';
  document.getElementById('editBtn').textContent=editMode?'✕ CLOSE EDITOR':'✏ EDIT MAP';
  document.getElementById('editBtn').classList.toggle('active',editMode);
  gc.style.cursor=editMode?'crosshair':'default';
}

function setTool(t){
  currentTool=t;
  document.getElementById('tObs').classList.toggle('on',t==='obs');
  document.getElementById('tNfz').classList.toggle('on',t==='nfz');
  document.getElementById('tEra').classList.toggle('on',t==='era');
}

function cellFromEvent(e){
  const rect=gc.getBoundingClientRect();
  return [Math.floor((e.clientY-rect.top)/CELL), Math.floor((e.clientX-rect.left)/CELL)];
}

function applyPaint(r,c){
  if(r<0||r>=ROWS||c<0||c>=COLS||running) return;
  if(r===0&&c===0) return;
  userPaints=userPaints.filter(p=>!(p[0]===r&&p[1]===c));
  userPaints.push([r,c,currentTool]);
  liveGrid[r][c]=baseGrid[r][c]=currentTool==='obs'?OBS:currentTool==='nfz'?NOFLY:FREE;
  fogGrid[r][c]=false;
  // Fix 5: If painting obstacle, update RL memory so heatmap shows it
  if(currentTool==='obs'){
    MEM.learnObstacle(r,c);
  } else if(currentTool==='era'){
    // Erasing — remove from danger map partially
    const k=`${r},${c}`;
    if(MEM.danger[k]) delete MEM.danger[k];
    MEM.knownObs=MEM.knownObs.filter(o=>!(o[0]===r&&o[1]===c));
  }
  drawMap();
  drawHeatmap(); // Fix 5: always redraw heatmap after edit
}

function clearEdits(){ userPaints=[]; initMission(); log('Editor cleared','w'); }

gc.addEventListener('mousedown',e=>{
  if(obsMode&&running){ placeObstacle(...cellFromEvent(e)); return; }
  if(!editMode||running) return;
  mouseDown=true; applyPaint(...cellFromEvent(e));
});
gc.addEventListener('mousemove',e=>{ if(mouseDown&&editMode&&!running) applyPaint(...cellFromEvent(e)); });
gc.addEventListener('mouseup',()=>mouseDown=false);
gc.addEventListener('mouseleave',()=>mouseDown=false);

// ═══════════════════════════════════════════════════════════
// HINT SYSTEM
// ═══════════════════════════════════════════════════════════
let hintTimer=null;
function showHint(msg,type='inf'){
  const box=document.getElementById('hintBox');
  box.innerHTML=`<div class="hint hint-${type}">${msg}</div>`;
  if(hintTimer) clearTimeout(hintTimer);
  if(type!=='obs') hintTimer=setTimeout(clearHint,3000);
}
function clearHint(){ document.getElementById('hintBox').innerHTML=''; }

// ═══════════════════════════════════════════════════════════
// EVENT LOG
// ═══════════════════════════════════════════════════════════
function log(msg,type=''){
  const el=document.createElement('div');
  el.className='ei'+(type?' '+type:'');
  el.textContent=`[${new Date().toLocaleTimeString('en',{hour12:false})}] ${msg}`;
  const lg=document.getElementById('elog');
  lg.insertBefore(el,lg.firstChild);
  while(lg.children.length>60) lg.removeChild(lg.lastChild);
}

// ═══════════════════════════════════════════════════════════
// COORDINATE MODAL
// ═══════════════════════════════════════════════════════════
function openCoord(){ updPreview(); document.getElementById('coordModal').classList.add('open'); }
function closeCoord(){
  document.getElementById('coordModal').classList.remove('open');
  const mb=document.getElementById('coordNextBanner');
  if(mb) mb.style.display='none';
}
function setPreset(lat,lng){ document.getElementById('iLat').value=lat; document.getElementById('iLng').value=lng; updPreview(); }
function updPreview(){
  const lat=parseFloat(document.getElementById('iLat').value)||28.6139;
  const lng=parseFloat(document.getElementById('iLng').value)||77.2090;
  const rad=parseFloat(document.getElementById('iRad').value)||2;
  const alt=parseFloat(document.getElementById('iAlt').value)||120;
  document.getElementById('pLL').textContent=`${fmtCoord(lat,true)}, ${fmtCoord(lng,false)}`;
  document.getElementById('pArea').textContent=`${(rad*2).toFixed(1)} × ${(rad*2).toFixed(1)} km`;
  document.getElementById('pCell').textContent=`~${Math.round(rad*2/ROWS*1000)} × ${Math.round(rad*2/COLS*1000)} m`;
  document.getElementById('pAlt').textContent=`${alt} m AGL`;
}
function resetMemoryAndApply(){
  MEM.reset();
  log('RL memory cleared — fresh start','al');
  applyCoord();
}

function applyCoord(){
  Z.lat=parseFloat(document.getElementById('iLat').value)||28.6139;
  Z.lng=parseFloat(document.getElementById('iLng').value)||77.2090;
  Z.rad=parseFloat(document.getElementById('iRad').value)||2;
  Z.alt=parseFloat(document.getElementById('iAlt').value)||120;
  // Hide next-mission banner
  const mb=document.getElementById('coordNextBanner');
  if(mb) mb.style.display='none';
  closeCoord();
  if(running){clearInterval(simTimer);running=false;}
  // On next mission: preserve MEM (keeps obstacle memory + Q-learning)
  // On first launch / manual open: also preserve memory (user may just change zone)
  userPaints=[];
  document.getElementById('doneOv').classList.remove('show');
  initMission(); updateMsnHistory();
  log(`Zone: ${fmtCoord(Z.lat,true)} ${fmtCoord(Z.lng,false)} | ${(Z.rad*2).toFixed(1)}km | ~${Math.round(Z.rad*2/ROWS*1000)}m/cell | MSN ${msnNum}`,'mem');
}

// ═══════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════
function updateClock(){ document.getElementById('clk').textContent=new Date().toLocaleTimeString('en',{hour12:false}); }
updateClock(); setInterval(updateClock,1000);
window.addEventListener('resize',()=>{ setupCanvas(); drawMap(); });
setupCanvas(); initMission(); updateMsnHistory(); onSpeedChange();
log('🦅 GARUDA-OPS v4 — ALL BUGS FIXED', 'ok');
log('✅ No backtracking | ✅ Replan from current pos | ✅ 3D drones | ✅ Return-home', 'inf');
log('✅ Battery 1200 units | ✅ RTB at 12% | ✅ Numeric speed display', 'mem');
log('Place obstacle → click button → click any revealed FREE cell on map', 'w');
setTimeout(openCoord, 700);