import React, { useEffect, useRef, useState } from "react";
import * as BABYLON from "babylonjs";

const WORLD_SIZE = 52;
const HALF = WORLD_SIZE / 2;
const BOT_RADIUS = 0.7;
const ENEMY_RADIUS = 0.8;
const OBSTACLE_HEIGHT = 2;
const FOOD_RADIUS = 0.45;

function clamp(v, min, max) {
  return Math.min(max, Math.max(min, v));
}

function len(x, z) {
  return Math.sqrt(x * x + z * z);
}

function dist2D(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.z - b.z) ** 2);
}

function normalize2D(x, z) {
  const l = len(x, z) || 1;
  return { x: x / l, z: z / l };
}

function resolveCircleAabbCollision(x, z, radius, obstacle) {
  const minX = obstacle.x - obstacle.size;
  const maxX = obstacle.x + obstacle.size;
  const minZ = obstacle.z - obstacle.size;
  const maxZ = obstacle.z + obstacle.size;

  const closestX = clamp(x, minX, maxX);
  const closestZ = clamp(z, minZ, maxZ);
  const dx = x - closestX;
  const dz = z - closestZ;
  const d2 = dx * dx + dz * dz;
  const r2 = radius * radius;

  if (d2 >= r2) return { x, z, collided: false };

  if (d2 > 1e-10) {
    const d = Math.sqrt(d2);
    const push = radius - d;
    const nx = dx / d;
    const nz = dz / d;
    return { x: x + nx * push, z: z + nz * push, collided: true };
  }

  const left = x - minX;
  const right = maxX - x;
  const top = z - minZ;
  const bottom = maxZ - z;
  const minPen = Math.min(left, right, top, bottom);

  if (minPen === left) return { x: minX - radius, z, collided: true };
  if (minPen === right) return { x: maxX + radius, z, collided: true };
  if (minPen === top) return { x, z: minZ - radius, collided: true };
  return { x, z: maxZ + radius, collided: true };
}

function repelFromObstacles(entity, obstacles, radius = 1.2) {
  let rx = 0;
  let rz = 0;
  for (const o of obstacles) {
    const dx = entity.x - o.x;
    const dz = entity.z - o.z;
    const d = Math.sqrt(dx * dx + dz * dz) || 0.0001;
    const minD = o.size + radius;
    if (d < minD + 1.8) {
      const strength = (minD + 1.8 - d) / (minD + 1.8);
      rx += (dx / d) * strength;
      rz += (dz / d) * strength;
    }
  }

  // Boundary wall repel so bots/enemies avoid edges like obstacles.
  const bound = HALF - radius;
  const margin = 3.5;
  if (entity.x > bound - margin) {
    const t = clamp((entity.x - (bound - margin)) / margin, 0, 1);
    rx += -t * 2.2;
  }
  if (entity.x < -bound + margin) {
    const t = clamp(((-bound + margin) - entity.x) / margin, 0, 1);
    rx += t * 2.2;
  }
  if (entity.z > bound - margin) {
    const t = clamp((entity.z - (bound - margin)) / margin, 0, 1);
    rz += -t * 2.2;
  }
  if (entity.z < -bound + margin) {
    const t = clamp(((-bound + margin) - entity.z) / margin, 0, 1);
    rz += t * 2.2;
  }
  return { x: rx, z: rz };
}

function randomFreePosition(obstacles, enemies = [], foods = [], margin = 2) {
  for (let i = 0; i < 200; i++) {
    const x = (Math.random() - 0.5) * (WORLD_SIZE - 2 * margin);
    const z = (Math.random() - 0.5) * (WORLD_SIZE - 2 * margin);
    let ok = true;
    for (const o of obstacles) {
      if (Math.sqrt((x - o.x) ** 2 + (z - o.z) ** 2) < o.size + 1.5) ok = false;
    }
    for (const e of enemies) {
      if (Math.sqrt((x - e.x) ** 2 + (z - e.z) ** 2) < 2.5) ok = false;
    }
    for (const f of foods) {
      if (Math.sqrt((x - f.x) ** 2 + (z - f.z) ** 2) < 1.5) ok = false;
    }
    if (ok) return { x, z };
  }
  return { x: 0, z: 0 };
}

function computeThreat(bot, enemies) {
  let t = 0;
  let nearest = Infinity;
  for (const e of enemies) {
    const d = dist2D(bot, e);
    nearest = Math.min(nearest, d);
    t += 1 / Math.max(d, 0.75);
  }
  return { level: Math.min(1, t / 3.5), nearest: Number.isFinite(nearest) ? nearest : 999 };
}

function useSimulation() {
  const [obstacles, setObstacles] = useState([
    { id: crypto.randomUUID(), x: -4, z: -2, size: 2.2 },
    { id: crypto.randomUUID(), x: 5, z: 3, size: 1.8 },
    { id: crypto.randomUUID(), x: 1, z: -6, size: 2.4 },
  ]);

  const [foods, setFoods] = useState(() => {
    const f = [];
    for (let i = 0; i < 6; i++) {
      const p = randomFreePosition([], [], f);
      f.push({ id: crypto.randomUUID(), x: p.x, z: p.z });
    }
    return f;
  });

  const [enemies, setEnemies] = useState(() => {
    const e = [];
    for (let i = 0; i < 3; i++) {
      const p = randomFreePosition([], e, []);
      e.push({ id: crypto.randomUUID(), x: p.x, z: p.z, vx: 0, vz: 0 });
    }
    return e;
  });

  const [bot, setBot] = useState({ x: 0, z: 0, vx: 0, vz: 0, health: 100, energy: 100, aliveTime: 0 });
  const [paused, setPaused] = useState(false);
  const [running, setRunning] = useState(true);
  const [mode, setMode] = useState("obstacle");
  const [enemySpeed, setEnemySpeed] = useState(2.4);
  const [botSpeed, setBotSpeed] = useState(3.2);
  const [spawnFood, setSpawnFood] = useState(true);
  const [botTrail, setBotTrail] = useState([]);
  const [ticks, setTicks] = useState(0);

  const addObstacle = (x, z) => {
    setObstacles((prev) => [...prev, { id: crypto.randomUUID(), x, z, size: 1.4 + Math.random() * 1.5 }]);
  };

  const addEnemy = (x, z) => {
    setEnemies((prev) => [...prev, { id: crypto.randomUUID(), x, z, vx: 0, vz: 0 }]);
  };

  const addFood = (x, z) => {
    setFoods((prev) => [...prev, { id: crypto.randomUUID(), x, z }]);
  };

  const reset = () => {
    const start = randomFreePosition(obstacles, enemies, foods);
    setBot({ x: start.x, z: start.z, vx: 0, vz: 0, health: 100, energy: 100, aliveTime: 0 });
    setBotTrail([]);
    setTicks(0);
    setRunning(true);
  };

  const clearWorld = () => {
    setObstacles([]);
    setEnemies([]);
    setFoods([]);
    setBot({ x: 0, z: 0, vx: 0, vz: 0, health: 100, energy: 100, aliveTime: 0 });
    setBotTrail([]);
    setTicks(0);
    setRunning(true);
  };

  const randomize = () => {
    const nextObstacles = [];
    for (let i = 0; i < 14; i++) {
      const p = randomFreePosition(nextObstacles, [], [], 3);
      nextObstacles.push({ id: crypto.randomUUID(), x: p.x, z: p.z, size: 1 + Math.random() * 2.2 });
    }
    const nextFoods = [];
    for (let i = 0; i < 9; i++) {
      const p = randomFreePosition(nextObstacles, [], nextFoods);
      nextFoods.push({ id: crypto.randomUUID(), x: p.x, z: p.z });
    }
    const nextEnemies = [];
    for (let i = 0; i < 5; i++) {
      const p = randomFreePosition(nextObstacles, nextEnemies, nextFoods);
      nextEnemies.push({ id: crypto.randomUUID(), x: p.x, z: p.z, vx: 0, vz: 0 });
    }
    const start = randomFreePosition(nextObstacles, nextEnemies, nextFoods);
    setObstacles(nextObstacles);
    setFoods(nextFoods);
    setEnemies(nextEnemies);
    setBot({ x: start.x, z: start.z, vx: 0, vz: 0, health: 100, energy: 100, aliveTime: 0 });
    setBotTrail([]);
    setTicks(0);
    setRunning(true);
  };

  const step = (dt) => {
    if (paused || !running) return;

    setTicks((t) => t + 1);

    setEnemies((prevEnemies) => {
      let latestBot = null;
      setBot((prevBot) => {
        const threat = computeThreat(prevBot, prevEnemies);

        let fleeX = 0;
        let fleeZ = 0;
        let nearestEnemy = null;
        let nearestEnemyDist = Infinity;

        for (const e of prevEnemies) {
          const dx = prevBot.x - e.x;
          const dz = prevBot.z - e.z;
          const d = Math.sqrt(dx * dx + dz * dz) || 0.001;
          if (d < nearestEnemyDist) {
            nearestEnemyDist = d;
            nearestEnemy = e;
          }
          const strength = 1 / Math.max(d * d, 0.5);
          fleeX += (dx / d) * strength;
          fleeZ += (dz / d) * strength;
        }

        let foodSeekX = 0;
        let foodSeekZ = 0;
        let nearestFoodDist = Infinity;
        let bestFood = null;
        for (const f of foods) {
          const d = dist2D(prevBot, f);
          if (d < nearestFoodDist) {
            nearestFoodDist = d;
            bestFood = f;
          }
        }
        if (bestFood) {
          const dx = bestFood.x - prevBot.x;
          const dz = bestFood.z - prevBot.z;
          const n = normalize2D(dx, dz);
          const hungerBoost = prevBot.energy < 45 ? 1.35 : 0.65;
          foodSeekX = n.x * hungerBoost;
          foodSeekZ = n.z * hungerBoost;
        }

        const obstacleRepel = repelFromObstacles(prevBot, obstacles, BOT_RADIUS);

        let wanderX = Math.sin((ticks + 15) * 0.03) * 0.25;
        let wanderZ = Math.cos((ticks + 9) * 0.027) * 0.25;

        let coverX = 0;
        let coverZ = 0;
        if (nearestEnemy) {
          let best = null;
          let bestScore = -Infinity;
          for (const o of obstacles) {
            const botToObs = dist2D(prevBot, o);
            const enemyToObs = dist2D(nearestEnemy, o);
            const score = enemyToObs - botToObs - o.size * 0.5;
            if (score > bestScore) {
              bestScore = score;
              best = o;
            }
          }
          if (best && threat.level > 0.28) {
            const dx = best.x - prevBot.x;
            const dz = best.z - prevBot.z;
            const n = normalize2D(dx, dz);
            coverX = n.x * 1.1;
            coverZ = n.z * 1.1;
          }
        }

        const panic = threat.level;
        const nx = fleeX * (2.7 + panic * 1.8) + foodSeekX * (1 - panic) + obstacleRepel.x * 2.0 + coverX * panic + wanderX;
        const nz = fleeZ * (2.7 + panic * 1.8) + foodSeekZ * (1 - panic) + obstacleRepel.z * 2.0 + coverZ * panic + wanderZ;
        const n = normalize2D(nx, nz);

        const speed = botSpeed * (prevBot.energy > 25 ? 1 : 0.72);
        let nextX = prevBot.x + n.x * speed * dt;
        let nextZ = prevBot.z + n.z * speed * dt;

        nextX = clamp(nextX, -HALF + 1, HALF - 1);
        nextZ = clamp(nextZ, -HALF + 1, HALF - 1);

        for (const o of obstacles) {
          const resolved = resolveCircleAabbCollision(nextX, nextZ, BOT_RADIUS, o);
          if (resolved.collided) {
            nextX = resolved.x;
            nextZ = resolved.z;
          }
        }

        let nextHealth = prevBot.health;
        let nextEnergy = clamp(prevBot.energy - dt * (4.2 + panic * 3.8), 0, 100);
        let nextAlive = prevBot.aliveTime + dt;

        for (const e of prevEnemies) {
          const d = Math.sqrt((nextX - e.x) ** 2 + (nextZ - e.z) ** 2);
          if (d < BOT_RADIUS + ENEMY_RADIUS + 0.1) {
            nextHealth -= 28 * dt;
          }
        }

        let eatenFoodId = null;
        for (const f of foods) {
          const d = Math.sqrt((nextX - f.x) ** 2 + (nextZ - f.z) ** 2);
          if (d < BOT_RADIUS + FOOD_RADIUS + 0.2) {
            eatenFoodId = f.id;
            nextEnergy = clamp(nextEnergy + 35, 0, 100);
            nextHealth = clamp(nextHealth + 8, 0, 100);
            break;
          }
        }
        if (eatenFoodId) {
          setFoods((prev) => prev.filter((f) => f.id !== eatenFoodId));
        }

        if (nextEnergy <= 0) nextHealth -= 8 * dt;
        if (nextHealth <= 0) {
          nextHealth = 0;
          setRunning(false);
        }

        const nextBot = { x: nextX, z: nextZ, vx: n.x * speed, vz: n.z * speed, health: nextHealth, energy: nextEnergy, aliveTime: nextAlive };
        latestBot = nextBot;
        return nextBot;
      });

      const b = latestBot || bot;
      return prevEnemies.map((e, idx) => {
        const pursueBias = idx % 2 === 0 ? 1 : 0.75;
        const dx = b.x - e.x;
        const dz = b.z - e.z;
        const toBot = normalize2D(dx, dz);
        const avoid = repelFromObstacles(e, obstacles, ENEMY_RADIUS);
        const drift = { x: Math.sin((ticks + idx * 17) * 0.02) * 0.18, z: Math.cos((ticks + idx * 13) * 0.023) * 0.18 };
        const dir = normalize2D(toBot.x * pursueBias + avoid.x * 1.8 + drift.x, toBot.z * pursueBias + avoid.z * 1.8 + drift.z);
        let nx = e.x + dir.x * enemySpeed * dt;
        let nz = e.z + dir.z * enemySpeed * dt;
        nx = clamp(nx, -HALF + 1, HALF - 1);
        nz = clamp(nz, -HALF + 1, HALF - 1);
        for (const o of obstacles) {
          const resolved = resolveCircleAabbCollision(nx, nz, ENEMY_RADIUS, o);
          if (resolved.collided) {
            nx = resolved.x;
            nz = resolved.z;
          }
        }
        return { ...e, x: nx, z: nz, vx: dir.x * enemySpeed, vz: dir.z * enemySpeed };
      });
    });

    if (spawnFood && Math.random() < 0.008) {
      const p = randomFreePosition(obstacles, enemies, foods);
      setFoods((prev) => (prev.length > 12 ? prev : [...prev, { id: crypto.randomUUID(), x: p.x, z: p.z }]));
    }

    setBotTrail((prev) => {
      const next = [...prev, [bot.x, 0.1, bot.z]];
      return next.length > 100 ? next.slice(next.length - 100) : next;
    });
  };

  return {
    obstacles,
    foods,
    enemies,
    bot,
    paused,
    setPaused,
    running,
    mode,
    setMode,
    enemySpeed,
    setEnemySpeed,
    botSpeed,
    setBotSpeed,
    spawnFood,
    setSpawnFood,
    botTrail,
    addObstacle,
    addEnemy,
    addFood,
    clearWorld,
    reset,
    randomize,
    step,
  };
}

function BabylonWorld({ sim }) {
  const canvasRef = useRef(null);
  const simRef = useRef(sim);

  useEffect(() => {
    simRef.current = sim;
  }, [sim]);

  useEffect(() => {
    // Babylon pattern: get the canvas element and create an engine from it.
    if (!canvasRef.current) return;

    const engine = new BABYLON.Engine(canvasRef.current, true, { preserveDrawingBuffer: true, stencil: true });
    const scene = new BABYLON.Scene(engine);
    scene.clearColor = BABYLON.Color4.FromHexString("#020617FF");

    const camera = new BABYLON.ArcRotateCamera("camera", -Math.PI / 4, 1.05, 38, BABYLON.Vector3.Zero(), scene);
    camera.attachControl(canvasRef.current, true);
    camera.lowerRadiusLimit = 10;
    camera.upperRadiusLimit = 80;
    camera.wheelDeltaPercentage = 0.01;

    const hemi = new BABYLON.HemisphericLight("ambient", new BABYLON.Vector3(0, 1, 0), scene);
    hemi.intensity = 0.8;
    const dir = new BABYLON.DirectionalLight("sun", new BABYLON.Vector3(-0.5, -1, -0.5), scene);
    dir.position = new BABYLON.Vector3(8, 15, 8);
    dir.intensity = 0.9;

    const ground = BABYLON.MeshBuilder.CreateGround("ground", { width: WORLD_SIZE, height: WORLD_SIZE }, scene);
    const groundMat = new BABYLON.StandardMaterial("groundMat", scene);
    groundMat.diffuseColor = BABYLON.Color3.FromHexString("#1f6f3e");
    ground.material = groundMat;

    const boundary = BABYLON.MeshBuilder.CreateLines(
      "boundary",
      {
        points: [
          new BABYLON.Vector3(-HALF, 0.04, -HALF),
          new BABYLON.Vector3(HALF, 0.04, -HALF),
          new BABYLON.Vector3(HALF, 0.04, HALF),
          new BABYLON.Vector3(-HALF, 0.04, HALF),
          new BABYLON.Vector3(-HALF, 0.04, -HALF),
        ],
      },
      scene,
    );
    boundary.color = BABYLON.Color3.White();

    // Visible boundary walls (treat edges like obstacles)
    const wallMat = new BABYLON.StandardMaterial("wallMat", scene);
    wallMat.diffuseColor = BABYLON.Color3.FromHexString("#334155");
    wallMat.emissiveColor = BABYLON.Color3.FromHexString("#0b1220");
    const wallH = 3.2;
    const wallT = 0.8;
    const wallY = wallH / 2;
    const wallN = BABYLON.MeshBuilder.CreateBox("wallN", { width: WORLD_SIZE, depth: wallT, height: wallH }, scene);
    wallN.position.set(0, wallY, -HALF);
    wallN.isPickable = false;
    wallN.material = wallMat;
    const wallS = BABYLON.MeshBuilder.CreateBox("wallS", { width: WORLD_SIZE, depth: wallT, height: wallH }, scene);
    wallS.position.set(0, wallY, HALF);
    wallS.isPickable = false;
    wallS.material = wallMat;
    const wallW = BABYLON.MeshBuilder.CreateBox("wallW", { width: wallT, depth: WORLD_SIZE, height: wallH }, scene);
    wallW.position.set(-HALF, wallY, 0);
    wallW.isPickable = false;
    wallW.material = wallMat;
    const wallE = BABYLON.MeshBuilder.CreateBox("wallE", { width: wallT, depth: WORLD_SIZE, height: wallH }, scene);
    wallE.position.set(HALF, wallY, 0);
    wallE.isPickable = false;
    wallE.material = wallMat;

    const obstacleMat = new BABYLON.StandardMaterial("obstacleMat", scene);
    obstacleMat.diffuseColor = BABYLON.Color3.FromHexString("#6b4f2a");
    const enemyMat = new BABYLON.StandardMaterial("enemyMat", scene);
    enemyMat.diffuseColor = BABYLON.Color3.FromHexString("#dc2626");
    const foodMat = new BABYLON.StandardMaterial("foodMat", scene);
    foodMat.diffuseColor = BABYLON.Color3.FromHexString("#f59e0b");
    foodMat.emissiveColor = BABYLON.Color3.FromHexString("#7c4a03");
    const botAliveMat = new BABYLON.StandardMaterial("botAliveMat", scene);
    botAliveMat.diffuseColor = BABYLON.Color3.FromHexString("#2563eb");
    const botDeadMat = new BABYLON.StandardMaterial("botDeadMat", scene);
    botDeadMat.diffuseColor = BABYLON.Color3.FromHexString("#374151");

    const obstacleMeshes = new Map();
    const enemyMeshes = new Map();
    const foodMeshes = new Map();

    const botMesh = BABYLON.MeshBuilder.CreateSphere("bot", { diameter: BOT_RADIUS * 2, segments: 24 }, scene);
    botMesh.position.y = BOT_RADIUS;
    botMesh.material = botAliveMat;

    let trailMesh = null;

    const syncObstacles = (obstacles) => {
      const ids = new Set(obstacles.map((o) => o.id));
      for (const [id, mesh] of obstacleMeshes.entries()) {
        if (!ids.has(id)) {
          mesh.dispose();
          obstacleMeshes.delete(id);
        }
      }
      for (const o of obstacles) {
        let mesh = obstacleMeshes.get(o.id);
        if (!mesh) {
          mesh = BABYLON.MeshBuilder.CreateBox(`ob-${o.id}`, { size: 1 }, scene);
          mesh.material = obstacleMat;
          obstacleMeshes.set(o.id, mesh);
        }
        mesh.scaling.set(o.size * 2, OBSTACLE_HEIGHT, o.size * 2);
        mesh.position.set(o.x, OBSTACLE_HEIGHT / 2, o.z);
      }
    };

    const syncFoods = (foods) => {
      const ids = new Set(foods.map((f) => f.id));
      for (const [id, mesh] of foodMeshes.entries()) {
        if (!ids.has(id)) {
          mesh.dispose();
          foodMeshes.delete(id);
        }
      }
      for (const f of foods) {
        let mesh = foodMeshes.get(f.id);
        if (!mesh) {
          mesh = BABYLON.MeshBuilder.CreateSphere(`food-${f.id}`, { diameter: FOOD_RADIUS * 2, segments: 18 }, scene);
          mesh.material = foodMat;
          foodMeshes.set(f.id, mesh);
        }
        mesh.position.set(f.x, FOOD_RADIUS, f.z);
      }
    };

    const syncEnemies = (enemies) => {
      const ids = new Set(enemies.map((e) => e.id));
      for (const [id, mesh] of enemyMeshes.entries()) {
        if (!ids.has(id)) {
          mesh.dispose();
          enemyMeshes.delete(id);
        }
      }
      for (const e of enemies) {
        let mesh = enemyMeshes.get(e.id);
        if (!mesh) {
          mesh = BABYLON.MeshBuilder.CreateSphere(`enemy-${e.id}`, { diameter: ENEMY_RADIUS * 2, segments: 20 }, scene);
          mesh.material = enemyMat;
          enemyMeshes.set(e.id, mesh);
        }
        mesh.position.set(e.x, ENEMY_RADIUS, e.z);
      }
    };

    const syncTrail = (trail) => {
      if (!trail || trail.length < 2) {
        if (trailMesh) {
          trailMesh.dispose();
          trailMesh = null;
        }
        return;
      }

      const points = trail.map((p) => new BABYLON.Vector3(p[0], p[1], p[2]));
      if (!trailMesh) {
        trailMesh = BABYLON.MeshBuilder.CreateLines("trail", { points, updatable: true }, scene);
        trailMesh.color = BABYLON.Color3.FromHexString("#93c5fd");
      } else {
        trailMesh = BABYLON.MeshBuilder.CreateLines("trail", { points, instance: trailMesh });
      }
    };

    scene.onPointerObservable.add((eventInfo) => {
      if (eventInfo.type !== BABYLON.PointerEventTypes.POINTERPICK) return;
      const pick = eventInfo.pickInfo;
      if (!pick?.hit || pick.pickedMesh !== ground || !pick.pickedPoint) return;

      const s = simRef.current;
      const { x, z } = pick.pickedPoint;
      if (s.mode === "obstacle") s.addObstacle(x, z);
      if (s.mode === "enemy") s.addEnemy(x, z);
      if (s.mode === "food") s.addFood(x, z);
    });

    engine.runRenderLoop(() => {
      const s = simRef.current;
      s.step(Math.min(engine.getDeltaTime() / 1000, 0.05));

      syncObstacles(s.obstacles);
      syncFoods(s.foods);
      syncEnemies(s.enemies);
      syncTrail(s.botTrail);

      botMesh.position.set(s.bot.x, BOT_RADIUS, s.bot.z);
      botMesh.material = s.running ? botAliveMat : botDeadMat;

      scene.render();
    });

    const onResize = () => engine.resize();
    window.addEventListener("resize", onResize);

    return () => {
      window.removeEventListener("resize", onResize);
      engine.dispose();
    };
  }, []);

  return <canvas id="renderCanvas" ref={canvasRef} className="fixed inset-0 block h-screen w-screen" />;
}

export default function SurvivalSim3D() {
  const sim = useSimulation();
  const [controlsCollapsed, setControlsCollapsed] = useState(false);

  return (
    <>
      <aside className="controls-panel">
        <div className="controls-title" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
          <span>Controls</span>
          <button onClick={() => setControlsCollapsed((v) => !v)}>{controlsCollapsed ? "Expand" : "Collapse"}</button>
        </div>

        {!controlsCollapsed && (
          <>
            <div className="controls-row">
              <label>Placement</label>
              <select value={sim.mode} onChange={(e) => sim.setMode(e.target.value)}>
                <option value="obstacle">Obstacle</option>
                <option value="enemy">Enemy</option>
                <option value="food">Food</option>
              </select>
            </div>
            <div className="controls-grid">
              <button onClick={() => sim.setPaused((prev) => !prev)}>{sim.paused ? "Resume" : "Pause"}</button>
              <button onClick={sim.reset}>Reset Bot</button>
              <button onClick={sim.randomize}>Random World</button>
              <button onClick={sim.clearWorld}>Clear World</button>
            </div>
            <div className="controls-row">
              <label>Bot Speed: {sim.botSpeed.toFixed(1)}</label>
              <input
                type="range"
                min="1"
                max="6"
                step="0.1"
                value={sim.botSpeed}
                onChange={(e) => sim.setBotSpeed(Number(e.target.value))}
              />
            </div>
            <div className="controls-row">
              <label>Enemy Speed: {sim.enemySpeed.toFixed(1)}</label>
              <input
                type="range"
                min="0.5"
                max="6"
                step="0.1"
                value={sim.enemySpeed}
                onChange={(e) => sim.setEnemySpeed(Number(e.target.value))}
              />
            </div>
            <label className="controls-check">
              <span>Auto-spawn food</span>
              <input type="checkbox" checked={sim.spawnFood} onChange={(e) => sim.setSpawnFood(e.target.checked)} />
            </label>
          </>
        )}
      </aside>

      <BabylonWorld sim={sim} />
    </>
  );
}