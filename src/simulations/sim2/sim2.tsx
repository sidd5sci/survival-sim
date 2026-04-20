import { useEffect, useMemo, useRef, useState } from "react";
import * as BABYLON from "babylonjs";
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card";
import { Button } from "../../components/ui/button";
import { Slider } from "../../components/ui/slider";
import { Badge } from "../../components/ui/badge";

type Vec2 = { x: number; z: number };
type Food = Vec2 & { id: string };
type Obstacle = Vec2 & { id: string; size: number; height: number };
type Gene = Vec2;

type Agent = {
	id: string;
	x: number;
	z: number;
	vx: number;
	vz: number;
	foodsEaten: number;
	firstFoodTime: number | null;
	minFoodDistance: number;
	collisions: number;
	fitness: number;
	dna: Gene[];
	eatenMask: boolean[];
};

type GaSettings = {
	population: number;
	mutationRate: number;
	mutationStrength: number;
	eliteCount: number;
	crossoverRate: number;
};

type SimState = {
	generation: number;
	elapsedSeconds: number;
	foods: Food[];
	obstacles: Obstacle[];
	agents: Agent[];
	bestEverFitness: number;
	bestLastGeneration: number;
};

const WORLD_SIZE = 80;
const HALF = WORLD_SIZE / 2;
const AGENT_RADIUS = 0.35;
const FOOD_RADIUS = 0.6;
const AGENT_ACCEL = 7.2;
const MAX_SPEED = 5.6;
const DRAG = 0.9;
const GENE_STEP_SECONDS = 0.33;
const GENERATION_SECONDS = 300;
const FOOD_COUNT = 8;
const OBSTACLE_COUNT = 12;

function clamp(v: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, v));
}

function dist2(a: Vec2, b: Vec2): number {
	const dx = a.x - b.x;
	const dz = a.z - b.z;
	return Math.sqrt(dx * dx + dz * dz);
}

function normalize(x: number, z: number): Vec2 {
	const len = Math.sqrt(x * x + z * z) || 1;
	return { x: x / len, z: z / len };
}

function rand(min: number, max: number): number {
	return Math.random() * (max - min) + min;
}

function nearestFoodDistance(pos: Vec2, foods: Food[]): number {
	let nearest = Number.POSITIVE_INFINITY;
	for (const food of foods) {
		nearest = Math.min(nearest, dist2(pos, food));
	}
	return nearest;
}

function randomGene(): Gene {
	const angle = Math.random() * Math.PI * 2;
	return { x: Math.cos(angle), z: Math.sin(angle) };
}

function randomDna(length: number): Gene[] {
	return Array.from({ length }, () => randomGene());
}

function randomId(prefix: string): string {
	return `${prefix}-${crypto.randomUUID()}`;
}

function createFoods(): Food[] {
	const foods: Food[] = [];
	for (let i = 0; i < FOOD_COUNT; i += 1) {
		foods.push({
			id: randomId("food"),
			x: rand(HALF - 20, HALF - 8),
			z: rand(HALF - 20, HALF - 8),
		});
	}
	return foods;
}

function createObstacles(): Obstacle[] {
	const obstacles: Obstacle[] = [];
	for (let i = 0; i < OBSTACLE_COUNT; i += 1) {
		obstacles.push({
			id: randomId("obs"),
			x: rand(-8, HALF - 6),
			z: rand(-HALF + 8, HALF - 8),
			size: rand(1.3, 3),
			height: rand(1.8, 4),
		});
	}
	return obstacles;
}

function createAgent(start: Vec2, foods: Food[], dna: Gene[], geneLength: number): Agent {
	return {
		id: randomId("agent"),
		x: start.x + rand(-1, 1),
		z: start.z + rand(-1, 1),
		vx: 0,
		vz: 0,
		foodsEaten: 0,
		firstFoodTime: null,
		minFoodDistance: nearestFoodDistance(start, foods),
		collisions: 0,
		fitness: 0,
		dna: dna.length ? dna : randomDna(geneLength),
		eatenMask: new Array(foods.length).fill(false),
	};
}

function createInitialState(settings: GaSettings): SimState {
	const foods = createFoods();
	const obstacles = createObstacles();
	const start = { x: -HALF + 8, z: -HALF + 8 };
	const geneLength = Math.ceil(GENERATION_SECONDS / GENE_STEP_SECONDS) + 2;
	const agents = Array.from({ length: settings.population }, () => createAgent(start, foods, randomDna(geneLength), geneLength));

	return {
		generation: 1,
		elapsedSeconds: 0,
		foods,
		obstacles,
		agents,
		bestEverFitness: 0,
		bestLastGeneration: 0,
	};
}

function obstacleRepel(pos: Vec2, obstacles: Obstacle[]): Vec2 {
	let rx = 0;
	let rz = 0;

	for (const obstacle of obstacles) {
		const dx = pos.x - obstacle.x;
		const dz = pos.z - obstacle.z;
		const d = Math.sqrt(dx * dx + dz * dz) || 0.0001;
		const safe = obstacle.size + AGENT_RADIUS + 1.3;
		if (d < safe) {
			const push = (safe - d) / safe;
			rx += (dx / d) * push;
			rz += (dz / d) * push;
		}
	}

	return { x: rx, z: rz };
}

function computeFitness(agent: Agent, elapsed: number, initialNearest: number): number {
	const progress = clamp(initialNearest - agent.minFoodDistance, 0, initialNearest);
	const speedReward = progress / Math.max(elapsed, 1);
	const firstFoodBonus = agent.firstFoodTime == null ? 0 : Math.max(0, GENERATION_SECONDS - agent.firstFoodTime) * 0.75;
	return agent.foodsEaten * 1400 + progress * 12 + speedReward * 450 + firstFoodBonus - agent.collisions * 18;
}

function selectParent(pool: Agent[]): Agent {
	const minFitness = Math.min(...pool.map((agent) => agent.fitness));
	const offset = minFitness < 0 ? Math.abs(minFitness) + 1 : 1;
	const total = pool.reduce((sum, agent) => sum + agent.fitness + offset, 0);
	let roll = Math.random() * total;

	for (const agent of pool) {
		roll -= agent.fitness + offset;
		if (roll <= 0) return agent;
	}
	return pool[pool.length - 1];
}

function crossoverDna(a: Gene[], b: Gene[], crossoverRate: number): Gene[] {
	if (Math.random() > crossoverRate) return a.map((gene) => ({ ...gene }));
	const pivot = Math.floor(Math.random() * a.length);
	const out: Gene[] = [];
	for (let i = 0; i < a.length; i += 1) {
		out.push(i < pivot ? { ...a[i] } : { ...b[i] });
	}
	return out;
}

function mutateDna(dna: Gene[], mutationRate: number, mutationStrength: number): Gene[] {
	return dna.map((gene) => {
		if (Math.random() > mutationRate) return { ...gene };
		const nx = gene.x + rand(-mutationStrength, mutationStrength);
		const nz = gene.z + rand(-mutationStrength, mutationStrength);
		return normalize(nx, nz);
	});
}

function evolve(state: SimState, settings: GaSettings): SimState {
	const start = { x: -HALF + 8, z: -HALF + 8 };
	const foods = createFoods();
	const obstacles = createObstacles();
	const baseNearest = nearestFoodDistance(start, foods);
	const ranked = [...state.agents].sort((a, b) => b.fitness - a.fitness);
	const bestPrev = ranked[0]?.fitness ?? 0;
	const elites = ranked.slice(0, clamp(settings.eliteCount, 1, ranked.length));
	const geneLength = elites[0]?.dna.length ?? Math.ceil(GENERATION_SECONDS / GENE_STEP_SECONDS) + 2;

	const nextAgents: Agent[] = [];
	for (const elite of elites) {
		nextAgents.push(createAgent(start, foods, elite.dna.map((g) => ({ ...g })), geneLength));
	}

	while (nextAgents.length < settings.population) {
		const parentA = selectParent(ranked);
		const parentB = selectParent(ranked);
		const crossed = crossoverDna(parentA.dna, parentB.dna, settings.crossoverRate);
		const mutated = mutateDna(crossed, settings.mutationRate, settings.mutationStrength);
		nextAgents.push(createAgent(start, foods, mutated, geneLength));
	}

	for (const agent of nextAgents) {
		agent.minFoodDistance = baseNearest;
	}

	return {
		generation: state.generation + 1,
		elapsedSeconds: 0,
		foods,
		obstacles,
		agents: nextAgents,
		bestLastGeneration: bestPrev,
		bestEverFitness: Math.max(state.bestEverFitness, bestPrev),
	};
}

function stepSimulation(prev: SimState, dt: number, settings: GaSettings, startPos: Vec2): SimState {
	const elapsed = prev.elapsedSeconds + dt;
	const geneIndex = Math.floor(elapsed / GENE_STEP_SECONDS);
	const baseNearest = nearestFoodDistance(startPos, prev.foods);

	const updatedAgents = prev.agents.map((agent) => {
		const gene = agent.dna[Math.min(geneIndex, agent.dna.length - 1)] ?? { x: 0, z: 0 };
		const repel = obstacleRepel(agent, prev.obstacles);
		const steer = normalize(gene.x + repel.x * 1.8, gene.z + repel.z * 1.8);

		let vx = (agent.vx + steer.x * AGENT_ACCEL * dt) * DRAG;
		let vz = (agent.vz + steer.z * AGENT_ACCEL * dt) * DRAG;
		const vLen = Math.sqrt(vx * vx + vz * vz) || 1;
		if (vLen > MAX_SPEED) {
			vx = (vx / vLen) * MAX_SPEED;
			vz = (vz / vLen) * MAX_SPEED;
		}

		let x = clamp(agent.x + vx * dt, -HALF + AGENT_RADIUS, HALF - AGENT_RADIUS);
		let z = clamp(agent.z + vz * dt, -HALF + AGENT_RADIUS, HALF - AGENT_RADIUS);
		let collisions = agent.collisions;

		for (const obstacle of prev.obstacles) {
			const dx = x - obstacle.x;
			const dz = z - obstacle.z;
			const d = Math.sqrt(dx * dx + dz * dz) || 0.0001;
			const minD = obstacle.size + AGENT_RADIUS;
			if (d < minD) {
				x = obstacle.x + (dx / d) * minD;
				z = obstacle.z + (dz / d) * minD;
				collisions += 1;
			}
		}

		let foodsEaten = agent.foodsEaten;
		let firstFoodTime = agent.firstFoodTime;
		const eatenMask = [...agent.eatenMask];

		for (let i = 0; i < prev.foods.length; i += 1) {
			if (eatenMask[i]) continue;
			const d = dist2({ x, z }, prev.foods[i]);
			if (d < FOOD_RADIUS + AGENT_RADIUS) {
				eatenMask[i] = true;
				foodsEaten += 1;
				if (firstFoodTime == null) firstFoodTime = elapsed;
			}
		}

		const minFoodDistance = Math.min(agent.minFoodDistance, nearestFoodDistance({ x, z }, prev.foods));
		const nextAgent: Agent = {
			...agent,
			x,
			z,
			vx,
			vz,
			collisions,
			foodsEaten,
			firstFoodTime,
			minFoodDistance,
			eatenMask,
			fitness: 0,
		};
		nextAgent.fitness = computeFitness(nextAgent, elapsed, baseNearest);
		return nextAgent;
	});

	const nextState: SimState = {
		...prev,
		elapsedSeconds: elapsed,
		agents: updatedAgents,
	};

	if (elapsed >= GENERATION_SECONDS) {
		return evolve(nextState, settings);
	}

	return nextState;
}

function hslToColor3(h: number, s: number, l: number): BABYLON.Color3 {
	const sat = s / 100;
	const light = l / 100;
	const c = (1 - Math.abs(2 * light - 1)) * sat;
	const hp = h / 60;
	const x = c * (1 - Math.abs((hp % 2) - 1));
	let r = 0;
	let g = 0;
	let b = 0;
	if (hp >= 0 && hp < 1) [r, g, b] = [c, x, 0];
	else if (hp < 2) [r, g, b] = [x, c, 0];
	else if (hp < 3) [r, g, b] = [0, c, x];
	else if (hp < 4) [r, g, b] = [0, x, c];
	else if (hp < 5) [r, g, b] = [x, 0, c];
	else [r, g, b] = [c, 0, x];
	const m = light - c / 2;
	return new BABYLON.Color3(r + m, g + m, b + m);
}

function BabylonWorld({ state }: { state: SimState }) {
	const canvasRef = useRef<HTMLCanvasElement | null>(null);
	const stateRef = useRef(state);

	useEffect(() => {
		stateRef.current = state;
	}, [state]);

	useEffect(() => {
		if (!canvasRef.current) return;

		const engine = new BABYLON.Engine(canvasRef.current, true, { preserveDrawingBuffer: true, stencil: true });
		const scene = new BABYLON.Scene(engine);
		scene.clearColor = BABYLON.Color4.FromHexString("#020617FF");

		const camera = new BABYLON.ArcRotateCamera("camera", -Math.PI / 4, 1.05, 62, BABYLON.Vector3.Zero(), scene);
		camera.attachControl(canvasRef.current, true);
		camera.lowerRadiusLimit = 10;
		camera.upperRadiusLimit = 120;

		const hemi = new BABYLON.HemisphericLight("ambient", new BABYLON.Vector3(0, 1, 0), scene);
		hemi.intensity = 0.9;
		const sun = new BABYLON.DirectionalLight("sun", new BABYLON.Vector3(-0.5, -1, -0.4), scene);
		sun.position = new BABYLON.Vector3(20, 24, 12);
		sun.intensity = 1.1;

		const ground = BABYLON.MeshBuilder.CreateGround("ground", { width: WORLD_SIZE, height: WORLD_SIZE }, scene);
		const groundMat = new BABYLON.StandardMaterial("groundMat", scene);
		groundMat.diffuseColor = BABYLON.Color3.FromHexString("#103023");
		ground.material = groundMat;

		const boundary = BABYLON.MeshBuilder.CreateLines(
			"boundary",
			{
				points: [
					new BABYLON.Vector3(-HALF, 0.03, -HALF),
					new BABYLON.Vector3(HALF, 0.03, -HALF),
					new BABYLON.Vector3(HALF, 0.03, HALF),
					new BABYLON.Vector3(-HALF, 0.03, HALF),
					new BABYLON.Vector3(-HALF, 0.03, -HALF),
				],
			},
			scene,
		);
		boundary.color = BABYLON.Color3.FromHexString("#64748b");

		const obstacleMeshes = new Map<string, BABYLON.Mesh>();
		const foodMeshes = new Map<string, BABYLON.Mesh>();
		const agentMeshes = new Map<string, BABYLON.Mesh>();

		const obstacleMat = new BABYLON.StandardMaterial("obstacleMat", scene);
		obstacleMat.diffuseColor = BABYLON.Color3.FromHexString("#64748b");
		const foodMat = new BABYLON.StandardMaterial("foodMat", scene);
		foodMat.diffuseColor = BABYLON.Color3.FromHexString("#facc15");
		foodMat.emissiveColor = BABYLON.Color3.FromHexString("#854d0e");

		const syncObstacles = (obstacles: Obstacle[]) => {
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
				mesh.scaling.set(o.size * 2, o.height, o.size * 2);
				mesh.position.set(o.x, o.height / 2, o.z);
			}
		};

		const syncFoods = (foods: Food[]) => {
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
					mesh = BABYLON.MeshBuilder.CreateSphere(`food-${f.id}`, { diameter: FOOD_RADIUS * 2, segments: 16 }, scene);
					mesh.material = foodMat;
					foodMeshes.set(f.id, mesh);
				}
				mesh.position.set(f.x, FOOD_RADIUS, f.z);
			}
		};

		const syncAgents = (agents: Agent[]) => {
			const best = [...agents].sort((a, b) => b.fitness - a.fitness)[0] ?? null;
			const ids = new Set(agents.map((a) => a.id));
			for (const [id, mesh] of agentMeshes.entries()) {
				if (!ids.has(id)) {
					mesh.dispose();
					agentMeshes.delete(id);
				}
			}

			for (const a of agents) {
				let mesh = agentMeshes.get(a.id);
				if (!mesh) {
					mesh = BABYLON.MeshBuilder.CreateCylinder(`agent-${a.id}`, { height: 0.7, diameterTop: 0, diameterBottom: 0.72, tessellation: 3 }, scene);
					agentMeshes.set(a.id, mesh);
				}

				const yaw = Math.atan2(a.vx || 0.001, a.vz || 0.001);
				mesh.position.set(a.x, AGENT_RADIUS, a.z);
				mesh.rotation.set(0, yaw, 0);

				const isBest = best?.id === a.id;
				const hue = clamp(120 - a.foodsEaten * 18 - a.collisions * 0.8, 0, 120);
				const color = isBest ? BABYLON.Color3.FromHexString("#22d3ee") : hslToColor3(hue, 82, 48);
				let mat = mesh.material as BABYLON.StandardMaterial | null;
				if (!mat) {
					mat = new BABYLON.StandardMaterial(`agent-mat-${a.id}`, scene);
					mesh.material = mat;
				}
				mat.diffuseColor = color;
			}
		};

		engine.runRenderLoop(() => {
			const s = stateRef.current;
			syncObstacles(s.obstacles);
			syncFoods(s.foods);
			syncAgents(s.agents);
			scene.render();
		});

		const onResize = () => engine.resize();
		window.addEventListener("resize", onResize);

		return () => {
			window.removeEventListener("resize", onResize);
			engine.dispose();
		};
	}, []);

	return <canvas ref={canvasRef} className="fixed inset-0 block h-screen w-screen" />;
}

export default function GeneticTrianglesSim3D() {
	const [settings, setSettings] = useState<GaSettings>({
		population: 100,
		mutationRate: 0.08,
		mutationStrength: 0.35,
		eliteCount: 10,
		crossoverRate: 0.82,
	});
	const [paused, setPaused] = useState(false);
	const [state, setState] = useState<SimState>(() => createInitialState(settings));
	const [controlsCollapsed, setControlsCollapsed] = useState(false);
	const startPos = useMemo(() => ({ x: -HALF + 8, z: -HALF + 8 }), []);

	useEffect(() => {
		let raf = 0;
		let last = performance.now();

		const tick = (now: number) => {
			const dt = Math.min((now - last) / 1000, 0.04);
			last = now;

			if (!paused) {
				setState((prev) => stepSimulation(prev, dt, settings, startPos));
			}

			raf = window.requestAnimationFrame(tick);
		};

		raf = window.requestAnimationFrame(tick);
		return () => window.cancelAnimationFrame(raf);
	}, [paused, settings, startPos]);

	const bestCurrent = useMemo(() => {
		return [...state.agents].sort((a, b) => b.fitness - a.fitness)[0] ?? null;
	}, [state.agents]);

	const restartWorld = () => {
		setState(createInitialState(settings));
	};

	return (
			<>
				<BabylonWorld state={state} />
				<aside className="controls-panel">
                        <Card className="bg-transparent border-0 shadow-none">
                            <div className="controls-title" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
                                <span>Controls</span>
                                <button onClick={() => setControlsCollapsed((v) => !v)}>{controlsCollapsed ? "Expand" : "Collapse"}</button>
                            </div>
							{!controlsCollapsed && (
							<CardContent className="space-y-4">
                                <div className="grid grid-cols-2 gap-2">
                                    <Button onClick={() => setPaused((value) => !value)}>{paused ? "Resume" : "Pause"}</Button>
                                    <Button variant="secondary" onClick={restartWorld}>Restart</Button>
                                </div>

                                <div className="rounded-xl border border-slate-700 p-3 text-sm space-y-1">
                                    <div>Generation: <strong>{state.generation}</strong></div>
                                    <div>Timer: <strong>{state.elapsedSeconds.toFixed(1)}s / 300s</strong></div>
                                    <div>Population: <strong>{state.agents.length}</strong></div>
                                    <div>Best (current): <strong>{bestCurrent ? bestCurrent.fitness.toFixed(1) : "0.0"}</strong></div>
                                    <div>Best (prev gen): <strong>{state.bestLastGeneration.toFixed(1)}</strong></div>
                                    <div>Best (ever): <strong>{state.bestEverFitness.toFixed(1)}</strong></div>
                                    <div>Best foods eaten: <strong>{bestCurrent?.foodsEaten ?? 0}</strong></div>
                                </div>

                                <div className="space-y-3">
                                    <div>
                                        <div className="flex justify-between text-sm mb-1"><span>Mutation Rate</span><span>{settings.mutationRate.toFixed(2)}</span></div>
                                        <Slider value={[settings.mutationRate]} min={0} max={0.45} step={0.01} onValueChange={(value: number[]) => setSettings((prev) => ({ ...prev, mutationRate: value[0] }))} />
                                    </div>
                                    <div>
                                        <div className="flex justify-between text-sm mb-1"><span>Mutation Strength</span><span>{settings.mutationStrength.toFixed(2)}</span></div>
                                        <Slider value={[settings.mutationStrength]} min={0.05} max={1.2} step={0.01} onValueChange={(value: number[]) => setSettings((prev) => ({ ...prev, mutationStrength: value[0] }))} />
                                    </div>
                                    <div>
                                        <div className="flex justify-between text-sm mb-1"><span>Elite Count</span><span>{Math.round(settings.eliteCount)}</span></div>
                                        <Slider value={[settings.eliteCount]} min={2} max={30} step={1} onValueChange={(value: number[]) => setSettings((prev) => ({ ...prev, eliteCount: Math.round(value[0]) }))} />
                                    </div>
                                    <div>
                                        <div className="flex justify-between text-sm mb-1"><span>Crossover Rate</span><span>{settings.crossoverRate.toFixed(2)}</span></div>
                                        <Slider value={[settings.crossoverRate]} min={0.1} max={1} step={0.01} onValueChange={(value: number[]) => setSettings((prev) => ({ ...prev, crossoverRate: value[0] }))} />
                                    </div>
                                    <div>
                                        <div className="flex justify-between text-sm mb-1"><span>Population (restart)</span><span>{Math.round(settings.population)}</span></div>
                                        <Slider value={[settings.population]} min={40} max={180} step={1} onValueChange={(value: number[]) => setSettings((prev) => ({ ...prev, population: Math.round(value[0]) }))} />
                                    </div>
                                </div>

                                <div className="rounded-xl border border-slate-700 p-3 text-xs text-slate-300 leading-relaxed">
                                    Fitness = foods eaten + distance progress + faster arrival reward - collision penalty. Each generation runs for 5 minutes, then selects elites and breeds a new DNA population.
                                </div>
							</CardContent>
							)}
                        </Card>
                    </aside>
			</>
	);
}
