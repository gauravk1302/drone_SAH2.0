// ─── SINGLE USER ───────────────────────────────
const CREDS = { id: "Gaurav", pwd: "Gaurav@4355", clr: "delta" };
let failCount = 0;

// ─── LOGIN ──────────────────────────────────────
function doLogin() {
  if (failCount >= 5) { showErr("SYSTEM LOCKED — MAXIMUM ATTEMPTS EXCEEDED"); return; }
  const uid = document.getElementById("uid").value.trim();
  const pwd = document.getElementById("pwd").value;
  const clr = document.getElementById("clr").value;
  if (!uid || !pwd || !clr) { showErr("⚠ ALL FIELDS REQUIRED"); return; }
  if (uid !== CREDS.id || pwd !== CREDS.pwd || clr !== CREDS.clr) {
    failCount++;
    const rem = 5 - failCount;
    showErr(rem > 0 ? `✕ AUTHENTICATION FAILED — ${rem} ATTEMPT${rem>1?"S":""} REMAINING` : "⛔ SYSTEM LOCKED");
    document.getElementById("errMsg").classList.remove("shake");
    requestAnimationFrame(() => document.getElementById("errMsg").classList.add("shake"));
    return;
  }
  document.getElementById("loginBtn").disabled = true;
  document.getElementById("loginBtn").textContent = "⟳  AUTHENTICATING...";
  setTimeout(startAuth, 500);
}

function showErr(msg) {
  const e = document.getElementById("errMsg");
  e.textContent = msg;
  e.classList.remove("shake");
}

document.getElementById("uid").addEventListener("keydown", e => { if(e.key==="Enter") document.getElementById("pwd").focus(); });
document.getElementById("pwd").addEventListener("keydown", e => { if(e.key==="Enter") document.getElementById("clr").focus(); });
document.getElementById("clr").addEventListener("keydown", e => { if(e.key==="Enter") doLogin(); });

// ─── AUTH SEQUENCE ────────────────────────────────
const AUTH_STEPS = [
  { label: "IDENTITY VERIFICATION", ms: 600 },
  { label: "BIOMETRIC SCAN COMPLETE", ms: 750 },
  { label: "CLEARANCE LEVEL CHECK", ms: 620 },
  { label: "ENCRYPTION HANDSHAKE", ms: 900 },
  { label: "MISSION DATABASE SYNC", ms: 520 },
  { label: "DRONE NETWORK ACCESS", ms: 1050 },
  { label: "GARUDA SYSTEM BOOT", ms: 680 }
];

function startAuth() {
  document.querySelector(".main-wrap").style.display = "none";
  const scr = document.getElementById("authScreen"); scr.style.display = "flex";
  document.getElementById("authSteps").innerHTML = AUTH_STEPS.map((s, i) => `
    <div class="auth-step">
      <span class="step-check" id="chk${i}">○</span>
      <div class="step-track"><div class="step-fill" id="fill${i}"></div></div>
      <span style="min-width:210px;font-size:7px;letter-spacing:1px;">${s.label}</span>
    </div>`).join("");

  let delay = 0;
  AUTH_STEPS.forEach((s, i) => {
    setTimeout(() => {
      document.getElementById("authStatus").textContent = "PROCESSING: " + s.label + "...";
      const f = document.getElementById("fill" + i);
      f.style.transition = `width ${s.ms * 0.82}ms ease`; f.style.width = "100%";
      setTimeout(() => {
        const c = document.getElementById("chk" + i); c.textContent = "✓"; c.classList.add("done");
        if (i === AUTH_STEPS.length - 1) {
          setTimeout(() => { document.getElementById("authStatus").textContent = "✓ GAURAV — ACCESS GRANTED — WELCOME COMMANDER"; }, 200);
          setTimeout(startSplash, 1000);
        }
      }, s.ms * 0.88);
    }, delay);
    delay += s.ms;
  });
}

// ─── SPLASH ────────────────────────────────────
const SPLASH_MSGS = [
  "LOADING SURVEILLANCE MODULES...",
  "INITIALISING Q-LEARNING ENGINE...",
  "CONNECTING DRONE NETWORK...",
  "CALIBRATING SENSOR ARRAYS...",
  "LOADING CARTOGRAPHIC DATA...",
  "SYNCING GARUDA SWARM INTELLIGENCE...",
  "SYSTEM READY — WELCOME COMMANDER GAURAV"
];

function startSplash() {
  document.getElementById("authScreen").style.display = "none";
  const scr = document.getElementById("splashScreen"); scr.style.display = "flex";
  setTimeout(() => { document.getElementById("splashBar").style.width = "100%"; }, 80);
  let mi = 0;
  const mt = setInterval(() => {
    document.getElementById("splashMsg").textContent = SPLASH_MSGS[Math.min(mi++, SPLASH_MSGS.length-1)];
  }, 720);
  let cd = 5;
  document.getElementById("splashCd").textContent = cd;
  const ct = setInterval(() => {
    cd--;
    document.getElementById("splashCd").textContent = cd > 0 ? cd : "";
    if (cd <= 0) clearInterval(ct);
  }, 1000);
  setTimeout(() => {
    clearInterval(mt);
    scr.style.transition = "opacity .8s";
    scr.style.opacity = "0";
    setTimeout(() => { window.location.href = "garuda_ops.html"; }, 850);
  }, 5500);
}

// ─── PARTICLES ─────────────────────────────────
const canvas = document.getElementById("bgCanvas");
const ctx = canvas.getContext("2d");
canvas.width = window.innerWidth; canvas.height = window.innerHeight;
const particles = Array.from({ length: 70 }, () => ({
  x: Math.random() * canvas.width, y: Math.random() * canvas.height,
  vx: (Math.random() - .5) * .33, vy: (Math.random() - .5) * .33,
  r: Math.random() * 1.4 + .4, a: Math.random() * .35 + .08,
  c: Math.random() > .5 ? "0,255,140" : "0,212,255"
}));
function animParticles() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  particles.forEach(p => {
    p.x += p.vx; p.y += p.vy;
    if (p.x < 0) p.x = canvas.width; if (p.x > canvas.width) p.x = 0;
    if (p.y < 0) p.y = canvas.height; if (p.y > canvas.height) p.y = 0;
    ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${p.c},${p.a})`; ctx.fill();
  });
  for (let i = 0; i < particles.length; i++) for (let j = i+1; j < particles.length; j++) {
    const dx = particles[i].x - particles[j].x, dy = particles[i].y - particles[j].y;
    const d = Math.sqrt(dx*dx + dy*dy);
    if (d < 90) {
      ctx.beginPath(); ctx.moveTo(particles[i].x, particles[i].y); ctx.lineTo(particles[j].x, particles[j].y);
      ctx.strokeStyle = `rgba(0,212,255,${.07*(1-d/90)})`; ctx.lineWidth = .5; ctx.stroke();
    }
  }
  requestAnimationFrame(animParticles);
}
animParticles();

// ─── CLOCK / TICKER ────────────────────────────
const TICKER_TEXT = "◈ GARUDA AUTONOMOUS SWARM SURVEILLANCE SYSTEM ◈ MINISTRY OF DEFENCE — CLASSIFIED ◈ UNAUTHORISED ACCESS IS PUNISHABLE BY LAW ◈ ALL SESSIONS ARE MONITORED AND RECORDED ◈ Q-LEARNING ACTIVE ◈ FOG OF WAR ENGAGED ◈ BOUSTROPHEDON PATH PLANNING ◈ A* ENERGY-AWARE ROUTING ◈ D* LITE DYNAMIC REPLANNING ◈ 3D DRONE SWARM ONLINE ◈ ";
document.getElementById("tickerText").textContent = TICKER_TEXT + TICKER_TEXT;
function updateTime() {
  const n = new Date();
  document.getElementById("topClock").textContent = n.toLocaleTimeString("en", { hour12: false });
  document.getElementById("botDate").textContent = n.toLocaleDateString("en-IN", { day:"2-digit", month:"short", year:"numeric" }).toUpperCase();
}
updateTime(); setInterval(updateTime, 1000);