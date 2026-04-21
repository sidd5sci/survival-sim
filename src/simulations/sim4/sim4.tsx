import { useEffect, useMemo, useRef, useState } from "react";
import * as BABYLON from "babylonjs";
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card";
import { Button } from "../../components/ui/button";
import { Slider } from "../../components/ui/slider";
import { Badge } from "../../components/ui/badge";

type Vec2 = { x: number; z: number };
type Food = Vec2 & { id: string };
type Obstacle = Vec2 & { id: string; size: number; height: number };

type BrainGenome = {
	hiddenLayers: number;
	hiddenSize: number;
	inputHiddenWeights: number[][];
	inputHiddenMask: number[][];
	hiddenBiases: number[][];
	hiddenHiddenWeights: number[][][];
	hiddenHiddenMask: number[][][];
	hiddenOutputWeights: number[][];
	hiddenOutputMask: number[][];
	outputBiases: number[];
};

type Agent = {
	id: string;
	spawnX: number;
	spawnZ: number;
	lazySeconds: number;
	x: number;
	z: number;
	vx: number;
	vz: number;
	headingX: number;
	headingZ: number;
	energy: number;
	alive: boolean;
	foodsEaten: number;
	firstFoodTime: number | null;
	minFoodDistance: number;
	collisions: number;
	fitness: number;
	// DNA: inherited genome encoding topology (hiddenSize), connection masks, and weights/biases.
	genome: BrainGenome;
	// Brain: runtime neural net state (not inherited). Useful for inspection/debugging.
	brain: {
		inputs: number[];
		hidden: number[][];
		outputs: number[];
	};
	eatenMask: boolean[];
};

type EvoSettings = {
	population: number;
	mutationRate: number;
	mutationStrength: number;
	eliteCount: number;
	topologyMutationRate: number;
	crossoverRate: number;
	// Generation duration in seconds.
	generationSeconds: number;
	// If enabled, agents can update their own DNA weights/biases during a generation.
	learningEnabled?: boolean;
	learningRate?: number;
};

type SimState = {
	generation: number;
	elapsedSeconds: number;
	foods: Food[];
	obstacles: Obstacle[];
	agents: Agent[];
	bestLastGeneration: number;
	bestEverFitness: number;
};

type SavedSim3GenomesV1 = {
	version: 1;
	sim: "sim3";
	createdAt: string;
	// Optional metadata (ignored by the loader/validator, but useful for debugging compatibility).
	net?: {
		inputs: number;
		outputs: number;
	};
	top: Array<{
		fitness: number;
		energy: number;
		foodsEaten: number;
		firstFoodTime: number | null;
		genome: BrainGenome;
	}>;
};

const WORLD_SIZE = 84;
const HALF = WORLD_SIZE / 2;
const AGENT_RADIUS = 0.35;
const FOOD_RADIUS = 0.6;
const DEFAULT_GENERATION_SECONDS = 180;
const FOOD_COUNT = 16;
const OBSTACLE_COUNT = 30;
const DEFAULT_VISION_RADIUS = 5;
const INPUTS = 10;
const OUTPUTS = 5;
const OUT_LEFT = 0;
const OUT_RIGHT = 1;
const OUT_FORWARD = 2;
const OUT_BACKWARD = 3;
const OUT_SPEED = 4;
const MAX_SPEED = 4.9;
const ACCEL = 8.5;
const DRAG = 0.89;

const MAX_HIDDEN_LAYERS = 5;

const START_ENERGY = 100;
const MAX_ENERGY = 150;
const ENERGY_GAIN_PER_FOOD = 25;
const ENERGY_DRAIN_BASE_PER_SEC = 0.04;
const ENERGY_DRAIN_SPEED_PER_UNIT = 0.06;
const ENERGY_DRAIN_ACCEL_PER_UNIT = 0.015;
const COLLISION_ENERGY_LOSS = 0.01;

const LAZY_DISTANCE_THRESHOLD = 10;
const LAZY_FITNESS_PENALTY_PER_SEC = 0.01;

const START_CLEAR_RADIUS = 6;

// Add a small amount of additional exploration specifically during crossover so
// offspring don't frequently end up identical to a parent (especially with low mutation).
const CROSSOVER_JITTER_RATE = 0.035;
const CROSSOVER_JITTER_STRENGTH = 0.22;


const CROSSOVER_PARENT_POOL_SIZE = 10;

const VISION_FOV_DEG = 120;
const VISION_HALF_FOV_RAD = (VISION_FOV_DEG * Math.PI) / 360;
const VISION_COS_THRESHOLD = Math.cos(VISION_HALF_FOV_RAD);

function clamp(v: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, v));
}

function rand(min: number, max: number): number {
	return Math.random() * (max - min) + min;
}

function normalize(x: number, z: number): Vec2 {
	const len = Math.sqrt(x * x + z * z) || 1;
	return { x: x / len, z: z / len };
}

function dist(a: Vec2, b: Vec2): number {
	const dx = a.x - b.x;
	const dz = a.z - b.z;
	return Math.sqrt(dx * dx + dz * dz);
}

function randomId(prefix: string): string {
	return `${prefix}-${crypto.randomUUID()}`;
}

function downloadJson(filename: string, data: unknown) {
	const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
	const url = URL.createObjectURL(blob);
	const a = document.createElement("a");
	a.href = url;
	a.download = filename;
	document.body.appendChild(a);
	a.click();
	a.remove();
	URL.revokeObjectURL(url);
}

function isSavedSim3GenomesV1(v: any): v is SavedSim3GenomesV1 {
	return (
		v &&
		v.version === 1 &&
		v.sim === "sim3" &&
		typeof v.createdAt === "string" &&
		Array.isArray(v.top) &&
		v.top.every((e: any) => e && typeof e === "object" && e.genome && typeof e.genome === "object")
	);
}

function createMazeWorld(): { obstacles: Obstacle[]; openCells: Vec2[] } {
	const padding = 0;
	const min = -HALF + padding;
	const max = HALF - padding;
	const usableSize = max - min;

	// Choose maze density first, then fit tiles exactly into the world.
	const desiredTileSize = 6;

	let gridW = Math.floor(usableSize / desiredTileSize);
	let gridH = Math.floor(usableSize / desiredTileSize);

	gridW = Math.max(5, gridW);
	gridH = Math.max(5, gridH);

	// Force odd dimensions so DFS-on-odd-cells behaves well.
	if (gridW % 2 === 0) gridW -= 1;
	if (gridH % 2 === 0) gridH -= 1;

	// IMPORTANT: fit exactly to bounds so no extra thick strip on top/right.
	const tileSizeX = usableSize / gridW;
	const tileSizeZ = usableSize / gridH;

	const originX = min;
	const originZ = min;

	const start = { x: -HALF + 8, z: -HALF + 8 };

	const grid: boolean[][] = Array.from({ length: gridH }, () =>
		Array.from({ length: gridW }, () => true),
	);

	const inBounds = (x: number, z: number) => x > 0 && x < gridW - 1 && z > 0 && z < gridH - 1;

	const dirs = [
		{ dx: 0, dz: -2 },
		{ dx: 2, dz: 0 },
		{ dx: 0, dz: 2 },
		{ dx: -2, dz: 0 },
	];

	const shuffle = <T,>(arr: T[]) => {
		for (let i = arr.length - 1; i > 0; i -= 1) {
			const j = Math.floor(rand(0, i + 1));
			[arr[i], arr[j]] = [arr[j], arr[i]];
		}
		return arr;
	};

	const startGX = 1;
	const startGZ = 1;

	const stack: Array<{ x: number; z: number }> = [{ x: startGX, z: startGZ }];
	grid[startGZ][startGX] = false;

	while (stack.length) {
		const cur = stack[stack.length - 1];

		const nextOptions = shuffle([...dirs]).filter((d) => {
			const nx = cur.x + d.dx;
			const nz = cur.z + d.dz;
			return inBounds(nx, nz) && grid[nz][nx];
		});

		if (nextOptions.length === 0) {
			stack.pop();
			continue;
		}

		const d = nextOptions[0];
		const nx = cur.x + d.dx;
		const nz = cur.z + d.dz;

		grid[cur.z + d.dz / 2][cur.x + d.dx / 2] = false;
		grid[nz][nx] = false;

		stack.push({ x: nx, z: nz });
	}

	// Keep your entrance opening exactly as before.
	if (gridH > startGZ && gridW > 0) {
		grid[startGZ][0] = false;
		grid[startGZ][startGX] = false;
	}

	const obstacles: Obstacle[] = [];
	const openCells: Vec2[] = [];

	const wallHeight = 3.2;
	const wallSize = Math.min(tileSizeX, tileSizeZ) * 0.50;

	for (let gz = 0; gz < gridH; gz += 1) {
		for (let gx = 0; gx < gridW; gx += 1) {
			const wx = originX + (gx + 0.5) * tileSizeX;
			const wz = originZ + (gz + 0.5) * tileSizeZ;

			if (grid[gz][gx]) {
				const clearance = START_CLEAR_RADIUS + wallSize + AGENT_RADIUS;
				if (dist({ x: wx, z: wz }, start) >= clearance) {
					obstacles.push({
						id: randomId("obs"),
						x: wx,
						z: wz,
						size: wallSize,
						height: wallHeight,
					});
				}
			} else {
				openCells.push({ x: wx, z: wz });
			}
		}
	}

	return { obstacles, openCells };
}

function createWorld(): { foods: Food[]; obstacles: Obstacle[] } {
	const { obstacles, openCells } = createMazeWorld();
	const start = { x: -HALF + 8, z: -HALF + 8 };
	// Pick many fruit locations spread across the maze (cell centers are guaranteed reachable in a perfect maze).
	const candidates = [...openCells]
		.filter((p) => dist(p, start) > START_CLEAR_RADIUS + 6)
		.sort(() => Math.random() - 0.5);

	const foods: Food[] = [];
	for (const p of candidates) {
		if (foods.length >= FOOD_COUNT) break;
		// Keep fruits away from walls.
		let ok = true;
		for (const o of obstacles) {
			if (dist(p, o) < o.size + FOOD_RADIUS + 0.6) {
				ok = false;
				break;
			}
		}
		if (!ok) continue;
		foods.push({ id: randomId("food"), x: p.x, z: p.z });
	}

	// Fallback: if the maze is dense and we didn't get enough, try random free spots.
	let attempts = 0;
	while (foods.length < FOOD_COUNT && attempts < 4000) {
		attempts += 1;
		const p = { x: rand(-HALF + 10, HALF - 10), z: rand(-HALF + 10, HALF - 10) };
		if (dist(p, start) <= START_CLEAR_RADIUS + 6) continue;
		let ok = true;
		for (const o of obstacles) {
			if (dist(p, o) < o.size + FOOD_RADIUS + 0.7) {
				ok = false;
				break;
			}
		}
		if (!ok) continue;
		foods.push({ id: randomId("food"), x: p.x, z: p.z });
	}

	return { foods, obstacles };
}

function nearestFoodDistance(pos: Vec2, foods: Food[]): number {
	let nearest = Number.POSITIVE_INFINITY;
	for (const food of foods) nearest = Math.min(nearest, dist(pos, food));
	return nearest;
}

function randomWeight(): number {
	return rand(-1, 1);
}

function randomMask(): number {
	return Math.random() < 0.7 ? 1 : 0;
}

function ensureGenomeInputShape(genome: BrainGenome): BrainGenome {
	// Backward/forward compatibility: if saved genomes were created with a different INPUTS count,
	// pad/slice input weight + mask rows so forwardPass can safely iterate 0..INPUTS-1.
	const next = cloneGenome(genome);

	const fixRowLen = (row: number[], targetLen: number, fill: () => number) => {
		if (row.length > targetLen) return row.slice(0, targetLen);
		if (row.length === targetLen) return row;
		const out = [...row];
		while (out.length < targetLen) out.push(fill());
		return out;
	};

	while (next.inputHiddenWeights.length < INPUTS) next.inputHiddenWeights.push([]);
	while (next.inputHiddenMask.length < INPUTS) next.inputHiddenMask.push([]);
	if (next.inputHiddenWeights.length > INPUTS) next.inputHiddenWeights = next.inputHiddenWeights.slice(0, INPUTS);
	if (next.inputHiddenMask.length > INPUTS) next.inputHiddenMask = next.inputHiddenMask.slice(0, INPUTS);

	for (let i = 0; i < INPUTS; i += 1) {
		next.inputHiddenWeights[i] = fixRowLen(next.inputHiddenWeights[i] ?? [], next.hiddenSize, randomWeight);
		next.inputHiddenMask[i] = fixRowLen(next.inputHiddenMask[i] ?? [], next.hiddenSize, randomMask);
	}

	return next;
}

function applyCrossoverJitter(genome: BrainGenome): BrainGenome {
	let changed = 0;
	const jitter = (v: number) => clamp(v + rand(-CROSSOVER_JITTER_STRENGTH, CROSSOVER_JITTER_STRENGTH), -3, 3);

	for (let i = 0; i < INPUTS; i += 1) {
		for (let h = 0; h < genome.hiddenSize; h += 1) {
			if (Math.random() < CROSSOVER_JITTER_RATE) {
				genome.inputHiddenWeights[i][h] = jitter(genome.inputHiddenWeights[i][h]);
				changed += 1;
			}
		}
	}

	for (let l = 0; l < genome.hiddenLayers; l += 1) {
		for (let h = 0; h < genome.hiddenSize; h += 1) {
			if (Math.random() < CROSSOVER_JITTER_RATE) {
				genome.hiddenBiases[l][h] = jitter(genome.hiddenBiases[l][h]);
				changed += 1;
			}
		}
	}

	for (let l = 0; l < Math.max(0, genome.hiddenLayers - 1); l += 1) {
		for (let i = 0; i < genome.hiddenSize; i += 1) {
			for (let h = 0; h < genome.hiddenSize; h += 1) {
				if (Math.random() < CROSSOVER_JITTER_RATE) {
					genome.hiddenHiddenWeights[l][i][h] = jitter(genome.hiddenHiddenWeights[l][i][h]);
					changed += 1;
				}
			}
		}
	}

	for (let h = 0; h < genome.hiddenSize; h += 1) {
		for (let o = 0; o < OUTPUTS; o += 1) {
			if (Math.random() < CROSSOVER_JITTER_RATE) {
				genome.hiddenOutputWeights[h][o] = jitter(genome.hiddenOutputWeights[h][o]);
				changed += 1;
			}
		}
	}

	for (let o = 0; o < OUTPUTS; o += 1) {
		if (Math.random() < CROSSOVER_JITTER_RATE) {
			genome.outputBiases[o] = jitter(genome.outputBiases[o]);
			changed += 1;
		}
	}

	// Guarantee at least one change on crossover to avoid exact clones.
	if (changed === 0) {
		const o = Math.floor(rand(0, OUTPUTS));
		genome.outputBiases[o] = jitter(genome.outputBiases[o]);
	}

	return genome;
}

function createRandomGenome(hiddenSize = Math.floor(rand(3, 8)), hiddenLayers = Math.floor(rand(1, 3))): BrainGenome {
	const layers = clamp(hiddenLayers, 1, MAX_HIDDEN_LAYERS);
	const inputHiddenWeights = Array.from({ length: INPUTS }, () => Array.from({ length: hiddenSize }, randomWeight));
	const inputHiddenMask = Array.from({ length: INPUTS }, () => Array.from({ length: hiddenSize }, randomMask));
	const hiddenBiases = Array.from({ length: layers }, () => Array.from({ length: hiddenSize }, randomWeight));
	const hiddenHiddenWeights = Array.from({ length: Math.max(0, layers - 1) }, () =>
		Array.from({ length: hiddenSize }, () => Array.from({ length: hiddenSize }, randomWeight)),
	);
	const hiddenHiddenMask = Array.from({ length: Math.max(0, layers - 1) }, () =>
		Array.from({ length: hiddenSize }, () => Array.from({ length: hiddenSize }, randomMask)),
	);
	const hiddenOutputWeights = Array.from({ length: hiddenSize }, () => Array.from({ length: OUTPUTS }, randomWeight));
	const hiddenOutputMask = Array.from({ length: hiddenSize }, () => Array.from({ length: OUTPUTS }, randomMask));
	const outputBiases = Array.from({ length: OUTPUTS }, randomWeight);

	return {
		hiddenLayers: layers,
		hiddenSize,
		inputHiddenWeights,
		inputHiddenMask,
		hiddenBiases,
		hiddenHiddenWeights,
		hiddenHiddenMask,
		hiddenOutputWeights,
		hiddenOutputMask,
		outputBiases,
	};
}

function cloneGenome(genome: BrainGenome): BrainGenome {
	return {
		hiddenLayers: genome.hiddenLayers,
		hiddenSize: genome.hiddenSize,
		inputHiddenWeights: genome.inputHiddenWeights.map((row) => [...row]),
		inputHiddenMask: genome.inputHiddenMask.map((row) => [...row]),
		hiddenBiases: genome.hiddenBiases.map((row) => [...row]),
		hiddenHiddenWeights: genome.hiddenHiddenWeights.map((m) => m.map((row) => [...row])),
		hiddenHiddenMask: genome.hiddenHiddenMask.map((m) => m.map((row) => [...row])),
		hiddenOutputWeights: genome.hiddenOutputWeights.map((row) => [...row]),
		hiddenOutputMask: genome.hiddenOutputMask.map((row) => [...row]),
		outputBiases: [...genome.outputBiases],
	};
}


function resizeGenome(genome: BrainGenome, nextHiddenSize: number, nextHiddenLayers: number): BrainGenome {
	const resized = createRandomGenome(nextHiddenSize, nextHiddenLayers);
	const overlap = Math.min(genome.hiddenSize, nextHiddenSize);
	const layerOverlap = Math.min(genome.hiddenLayers, resized.hiddenLayers);

	for (let i = 0; i < INPUTS; i += 1) {
		for (let h = 0; h < overlap; h += 1) {
			resized.inputHiddenWeights[i][h] = genome.inputHiddenWeights[i][h];
			resized.inputHiddenMask[i][h] = genome.inputHiddenMask[i][h];
		}
	}

	for (let l = 0; l < layerOverlap; l += 1) {
		for (let h = 0; h < overlap; h += 1) {
			resized.hiddenBiases[l][h] = genome.hiddenBiases[l][h];
		}
	}

	const hhOverlap = Math.min(Math.max(0, genome.hiddenLayers - 1), Math.max(0, resized.hiddenLayers - 1));
	for (let l = 0; l < hhOverlap; l += 1) {
		for (let i = 0; i < overlap; i += 1) {
			for (let h = 0; h < overlap; h += 1) {
				resized.hiddenHiddenWeights[l][i][h] = genome.hiddenHiddenWeights[l][i][h];
				resized.hiddenHiddenMask[l][i][h] = genome.hiddenHiddenMask[l][i][h];
			}
		}
	}

	for (let h = 0; h < overlap; h += 1) {
		for (let o = 0; o < OUTPUTS; o += 1) {
			resized.hiddenOutputWeights[h][o] = genome.hiddenOutputWeights[h][o];
			resized.hiddenOutputMask[h][o] = genome.hiddenOutputMask[h][o];
		}
	}

	for (let o = 0; o < OUTPUTS; o += 1) {
		if (genome.outputBiases[o] != null) resized.outputBiases[o] = genome.outputBiases[o];
	}
	return resized;
}

function mutateGenome(parent: BrainGenome, settings: EvoSettings): BrainGenome {
	let genome = cloneGenome(parent);

	if (Math.random() < settings.topologyMutationRate) {
		const deltaSize = Math.random() < 0.5 ? -1 : 1;
		const deltaLayers = Math.random() < 0.5 ? -1 : 1;
		const newSize = clamp(genome.hiddenSize + deltaSize, 2, 12);
		const newLayers = clamp(genome.hiddenLayers + deltaLayers, 1, MAX_HIDDEN_LAYERS);
		genome = resizeGenome(genome, newSize, newLayers);
	}

	for (let i = 0; i < INPUTS; i += 1) {
		for (let h = 0; h < genome.hiddenSize; h += 1) {
			if (Math.random() < settings.mutationRate) {
				genome.inputHiddenWeights[i][h] += rand(-settings.mutationStrength, settings.mutationStrength);
			}
			if (Math.random() < settings.mutationRate * 0.4) {
				genome.inputHiddenMask[i][h] = genome.inputHiddenMask[i][h] > 0.5 ? 0 : 1;
			}
		}
	}

	for (let l = 0; l < genome.hiddenLayers; l += 1) {
		for (let h = 0; h < genome.hiddenSize; h += 1) {
			if (Math.random() < settings.mutationRate) {
				genome.hiddenBiases[l][h] += rand(-settings.mutationStrength, settings.mutationStrength);
			}
		}
	}

	for (let l = 0; l < Math.max(0, genome.hiddenLayers - 1); l += 1) {
		for (let i = 0; i < genome.hiddenSize; i += 1) {
			for (let h = 0; h < genome.hiddenSize; h += 1) {
				if (Math.random() < settings.mutationRate) {
					genome.hiddenHiddenWeights[l][i][h] += rand(-settings.mutationStrength, settings.mutationStrength);
				}
				if (Math.random() < settings.mutationRate * 0.4) {
					genome.hiddenHiddenMask[l][i][h] = genome.hiddenHiddenMask[l][i][h] > 0.5 ? 0 : 1;
				}
			}
		}
	}

	for (let h = 0; h < genome.hiddenSize; h += 1) {
		for (let o = 0; o < OUTPUTS; o += 1) {
			if (Math.random() < settings.mutationRate) {
				genome.hiddenOutputWeights[h][o] += rand(-settings.mutationStrength, settings.mutationStrength);
			}
			if (Math.random() < settings.mutationRate * 0.4) {
				genome.hiddenOutputMask[h][o] = genome.hiddenOutputMask[h][o] > 0.5 ? 0 : 1;
			}
		}
	}

	for (let o = 0; o < OUTPUTS; o += 1) {
		if (Math.random() < settings.mutationRate) {
			genome.outputBiases[o] += rand(-settings.mutationStrength, settings.mutationStrength);
		}
	}

	return genome;
}

function crossoverGenomes(a: BrainGenome, b: BrainGenome, crossoverRate: number): BrainGenome {
	if (Math.random() > crossoverRate) return cloneGenome(a);

	const hiddenSize = Math.random() < 0.5 ? a.hiddenSize : b.hiddenSize;
	let hiddenLayers = Math.random() < 0.5 ? a.hiddenLayers : b.hiddenLayers;
	// Allow child to gain a hidden layer during crossover.
	if (Math.random() < 0.22) hiddenLayers = clamp(hiddenLayers + 1, 1, MAX_HIDDEN_LAYERS);
	const ra = resizeGenome(a, hiddenSize, hiddenLayers);
	const rb = resizeGenome(b, hiddenSize, hiddenLayers);
	const child = createRandomGenome(hiddenSize, hiddenLayers);

	for (let i = 0; i < INPUTS; i += 1) {
		for (let h = 0; h < hiddenSize; h += 1) {
			const chooseA = Math.random() < 0.5;
			child.inputHiddenWeights[i][h] = chooseA ? ra.inputHiddenWeights[i][h] : rb.inputHiddenWeights[i][h];
			child.inputHiddenMask[i][h] = chooseA ? ra.inputHiddenMask[i][h] : rb.inputHiddenMask[i][h];
		}
	}

	for (let l = 0; l < hiddenLayers; l += 1) {
		for (let h = 0; h < hiddenSize; h += 1) {
			child.hiddenBiases[l][h] = Math.random() < 0.5 ? ra.hiddenBiases[l][h] : rb.hiddenBiases[l][h];
		}
	}

	for (let l = 0; l < Math.max(0, hiddenLayers - 1); l += 1) {
		for (let i = 0; i < hiddenSize; i += 1) {
			for (let h = 0; h < hiddenSize; h += 1) {
				const chooseA = Math.random() < 0.5;
				child.hiddenHiddenWeights[l][i][h] = chooseA ? ra.hiddenHiddenWeights[l][i][h] : rb.hiddenHiddenWeights[l][i][h];
				child.hiddenHiddenMask[l][i][h] = chooseA ? ra.hiddenHiddenMask[l][i][h] : rb.hiddenHiddenMask[l][i][h];
			}
		}
	}

	for (let h = 0; h < hiddenSize; h += 1) {
		for (let o = 0; o < OUTPUTS; o += 1) {
			const chooseA = Math.random() < 0.5;
			child.hiddenOutputWeights[h][o] = chooseA ? ra.hiddenOutputWeights[h][o] : rb.hiddenOutputWeights[h][o];
			child.hiddenOutputMask[h][o] = chooseA ? ra.hiddenOutputMask[h][o] : rb.hiddenOutputMask[h][o];
		}
	}

	for (let o = 0; o < OUTPUTS; o += 1) {
		child.outputBiases[o] = Math.random() < 0.5 ? ra.outputBiases[o] : rb.outputBiases[o];
	}
	return applyCrossoverJitter(child);
}

function createAgent(start: Vec2, foods: Food[], genome: BrainGenome): Agent {
	const safeGenome = ensureGenomeInputShape(genome);
	const spawnX = start.x + rand(-0.8, 0.8);
	const spawnZ = start.z + rand(-0.8, 0.8);
	return {
		id: randomId("agent"),
		spawnX,
		spawnZ,
		lazySeconds: 0,
		x: spawnX,
		z: spawnZ,
		vx: 0,
		vz: 0,
		headingX: 0,
		headingZ: 1,
		energy: START_ENERGY,
		alive: true,
		foodsEaten: 0,
		firstFoodTime: null,
		minFoodDistance: nearestFoodDistance(start, foods),
		collisions: 0,
		fitness: 100,
		genome: safeGenome,
		brain: {
			inputs: new Array(INPUTS).fill(0),
			hidden: Array.from({ length: safeGenome.hiddenLayers }, () => new Array(safeGenome.hiddenSize).fill(0)),
			outputs: new Array(OUTPUTS).fill(0),
		},
		eatenMask: new Array(foods.length).fill(false),
	};
}

function createInitialState(settings: EvoSettings, seedGenomes?: BrainGenome[]): SimState {
	const { foods, obstacles } = createWorld();
	const start = { x: -HALF + 8, z: -HALF + 8 };
	const seeds = (seedGenomes ?? []).filter(Boolean).slice(0, settings.population);
	const agents: Agent[] = [];
	for (const g of seeds) agents.push(createAgent(start, foods, cloneGenome(g)));
	while (agents.length < settings.population) agents.push(createAgent(start, foods, createRandomGenome()));

	return {
		generation: 1,
		elapsedSeconds: 0,
		foods,
		obstacles,
		agents,
		bestLastGeneration: 0,
		bestEverFitness: 0,
	};
}

function getVisionInputs(agent: Agent, foods: Food[], obstacles: Obstacle[], visionRadius: number): number[] {
	const h = normalize(agent.headingX, agent.headingZ);
	const hx = h.x;
	const hz = h.z;
	// Right vector in XZ plane (90° clockwise from forward).
	const rx = hz;
	const rz = -hx;

	let nearestFood: Food | null = null;
	let nearestFoodDist = visionRadius + 1;

	for (let i = 0; i < foods.length; i += 1) {
		if (agent.eatenMask[i]) continue;
		const d = dist(agent, foods[i]);
		if (d > visionRadius || d >= nearestFoodDist) continue;
		const nxz = normalize(foods[i].x - agent.x, foods[i].z - agent.z);
		const dot = nxz.x * hx + nxz.z * hz;
		if (dot < VISION_COS_THRESHOLD) continue;
		{
			nearestFoodDist = d;
			nearestFood = foods[i];
		}
	}

	let nearestObs: Obstacle | null = null;
	let nearestObsDist = visionRadius + 1;

	for (const obstacle of obstacles) {
		const d = Math.max(0, dist(agent, obstacle) - obstacle.size);
		if (d > visionRadius || d >= nearestObsDist) continue;
		const nxz = normalize(obstacle.x - agent.x, obstacle.z - agent.z);
		const dot = nxz.x * hx + nxz.z * hz;
		if (dot < VISION_COS_THRESHOLD) continue;
		{
			nearestObsDist = d;
			nearestObs = obstacle;
		}
	}

	// Boundary walls as a "virtual obstacle".
	// IMPORTANT: use nearest wall distance (NOT FOV-gated).
	// When an agent slides along a wall, the wall is usually sideways; if we rely on FOV
	// or forward-ray intersection, walls become effectively invisible.
	let nearestWall: Vec2 | null = null;
	let nearestWallDist = visionRadius + 1;
	{
		const dxPos = HALF - agent.x;
		const dxNeg = agent.x + HALF;
		const dzPos = HALF - agent.z;
		const dzNeg = agent.z + HALF;
		let bestD = dxPos;
		let wallPoint: Vec2 = { x: HALF, z: agent.z };
		if (dxNeg < bestD) {
			bestD = dxNeg;
			wallPoint = { x: -HALF, z: agent.z };
		}
		if (dzPos < bestD) {
			bestD = dzPos;
			wallPoint = { x: agent.x, z: HALF };
		}
		if (dzNeg < bestD) {
			bestD = dzNeg;
			wallPoint = { x: agent.x, z: -HALF };
		}

		if (bestD <= visionRadius) {
			nearestWallDist = bestD;
			nearestWall = wallPoint;
		}
	}

	// Choose the closer of: real obstacle or boundary wall.
	const useWall = nearestWall && nearestWallDist < nearestObsDist;

	let foodDx = 0;
	let foodDz = 0;
	let foodDist = 0;
	if (nearestFood) {
		const n = normalize(nearestFood.x - agent.x, nearestFood.z - agent.z);
		// Creature-local direction: x=right, z=forward.
		foodDx = n.x * rx + n.z * rz;
		foodDz = n.x * hx + n.z * hz;
		foodDist = 1 - clamp(nearestFoodDist / Math.max(visionRadius, 0.0001), 0, 1);
	}

	let obsDx = 0;
	let obsDz = 0;
	let obsDist = 0;
	if (useWall && nearestWall) {
		const n = normalize(nearestWall.x - agent.x, nearestWall.z - agent.z);
		// Creature-local direction: x=right, z=forward.
		obsDx = n.x * rx + n.z * rz;
		obsDz = n.x * hx + n.z * hz;
		obsDist = 1 - clamp(nearestWallDist / Math.max(visionRadius, 0.0001), 0, 1);
	} else if (nearestObs) {
		const n = normalize(nearestObs.x - agent.x, nearestObs.z - agent.z);
		// Creature-local direction: x=right, z=forward.
		obsDx = n.x * rx + n.z * rz;
		obsDz = n.x * hx + n.z * hz;
		obsDist = 1 - clamp(nearestObsDist / Math.max(visionRadius, 0.0001), 0, 1);
	}

	return [
		foodDx,
		foodDz,
		foodDist,
		obsDx,
		obsDz,
		obsDist,
		clamp(agent.vx / MAX_SPEED, -1, 1),
		clamp(agent.vz / MAX_SPEED, -1, 1),
		clamp(agent.energy / Math.max(1, MAX_ENERGY), 0, 1),
		clamp(agent.fitness / 100, 0, 1),
	];
}

function neuralSteer(genome: BrainGenome, inputs: number[]): Vec2 {
	const { steer } = forwardPass(genome, inputs);
	return steer;
}

// Forward pass through the creature's neural net.
// Returns both movement steering (phenotype) and intermediate activations for learning + visualization.
function forwardPass(genome: BrainGenome, inputs: number[]) {
	const hiddenLayers: number[][] = Array.from({ length: genome.hiddenLayers }, () => new Array(genome.hiddenSize).fill(0));

	// Input -> first hidden
	for (let h = 0; h < genome.hiddenSize; h += 1) {
		let sum = genome.hiddenBiases[0][h];
		for (let i = 0; i < INPUTS; i += 1) {
			if (genome.inputHiddenMask[i][h] > 0.5) sum += inputs[i] * genome.inputHiddenWeights[i][h];
		}
		hiddenLayers[0][h] = Math.tanh(sum);
	}

	// Hidden -> hidden
	for (let l = 1; l < genome.hiddenLayers; l += 1) {
		for (let h = 0; h < genome.hiddenSize; h += 1) {
			let sum = genome.hiddenBiases[l][h];
			for (let i = 0; i < genome.hiddenSize; i += 1) {
				if (genome.hiddenHiddenMask[l - 1][i][h] > 0.5) sum += hiddenLayers[l - 1][i] * genome.hiddenHiddenWeights[l - 1][i][h];
			}
			hiddenLayers[l][h] = Math.tanh(sum);
		}
	}

	const lastHidden = hiddenLayers[genome.hiddenLayers - 1];

	const out = new Array(OUTPUTS).fill(0);
	for (let o = 0; o < OUTPUTS; o += 1) out[o] = genome.outputBiases[o] ?? 0;
	for (let h = 0; h < genome.hiddenSize; h += 1) {
		for (let o = 0; o < OUTPUTS; o += 1) {
			if (genome.hiddenOutputMask[h][o] > 0.5) out[o] += lastHidden[h] * genome.hiddenOutputWeights[h][o];
		}
	}

	// Output neurons are fixed actions:
	// left, right, forward, backward, speed.
	const act = out.map((v) => Math.tanh(v));
	let rawX = clamp((act[OUT_RIGHT] ?? 0) - (act[OUT_LEFT] ?? 0), -1, 1);
	let rawZ = clamp((act[OUT_FORWARD] ?? 0) - (act[OUT_BACKWARD] ?? 0), -1, 1);

	// IMPORTANT: Don't normalize tiny outputs to a unit vector.
	// Normalizing near-zero vectors makes agents move in arbitrary circles instead of stopping.
	const mag = Math.sqrt(rawX * rawX + rawZ * rawZ);
	const steer = mag < 0.08 ? { x: 0, z: 0 } : mag > 1 ? { x: rawX / mag, z: rawZ / mag } : { x: rawX, z: rawZ };

	const speed01 = clamp(((act[OUT_SPEED] ?? 0) + 1) * 0.5, 0, 1);

	return { hiddenLayers, out, steer, speed01 };
}

// Simple (optional) on-policy learning that writes back into DNA (Lamarckian).
// This is intentionally lightweight and heuristic: if a creature makes progress toward food,
// it slightly reinforces the weights that produced that action.
function learnGenome(genome: BrainGenome, inputs: number[], hiddenLayers: number[][], out: number[], reward: number, learningRate: number): BrainGenome {
	if (!Number.isFinite(reward) || Math.abs(reward) < 1e-6) return genome;
	const lr = learningRate * reward;
	const firstHidden = hiddenLayers[0] ?? new Array(genome.hiddenSize).fill(0);
	const lastHidden = hiddenLayers[genome.hiddenLayers - 1] ?? firstHidden;

	const next: BrainGenome = {
		hiddenLayers: genome.hiddenLayers,
		hiddenSize: genome.hiddenSize,
		inputHiddenWeights: genome.inputHiddenWeights.map((row) => [...row]),
		inputHiddenMask: genome.inputHiddenMask.map((row) => [...row]),
		hiddenBiases: genome.hiddenBiases.map((row) => [...row]),
		hiddenHiddenWeights: genome.hiddenHiddenWeights.map((m) => m.map((row) => [...row])),
		hiddenHiddenMask: genome.hiddenHiddenMask.map((m) => m.map((row) => [...row])),
		hiddenOutputWeights: genome.hiddenOutputWeights.map((row) => [...row]),
		hiddenOutputMask: genome.hiddenOutputMask.map((row) => [...row]),
		outputBiases: [...genome.outputBiases],
	};

	// Output layer reinforcement
	for (let h = 0; h < next.hiddenSize; h += 1) {
		for (let o = 0; o < OUTPUTS; o += 1) {
			if (next.hiddenOutputMask[h][o] > 0.5) {
				next.hiddenOutputWeights[h][o] = clamp(next.hiddenOutputWeights[h][o] + lr * lastHidden[h] * Math.tanh(out[o] ?? 0), -3, 3);
			}
		}
	}
	for (let o = 0; o < OUTPUTS; o += 1) {
		next.outputBiases[o] = clamp(next.outputBiases[o] + lr * Math.tanh(out[o] ?? 0) * 0.25, -3, 3);
	}

	// Hidden layer reinforcement
	for (let i = 0; i < INPUTS; i += 1) {
		for (let h = 0; h < next.hiddenSize; h += 1) {
			if (next.inputHiddenMask[i][h] > 0.5) {
				next.inputHiddenWeights[i][h] = clamp(next.inputHiddenWeights[i][h] + lr * inputs[i] * firstHidden[h] * 0.15, -3, 3);
			}
		}
	}
	for (let h = 0; h < next.hiddenSize; h += 1) {
		next.hiddenBiases[0][h] = clamp(next.hiddenBiases[0][h] + lr * firstHidden[h] * 0.1, -3, 3);
		next.hiddenBiases[next.hiddenLayers - 1][h] = clamp(next.hiddenBiases[next.hiddenLayers - 1][h] + lr * lastHidden[h] * 0.05, -3, 3);
	}

	return next;
}

function computeFitness(agent: Agent, elapsed: number, generationSeconds: number): number {
	// Fitness is strictly 0..100.
	// Requirement: starting fitness is 100, and penalties reduce fitness.
	const base = 100;
	// Penalties (reduce fitness)
	const collisionPenalty = clamp(agent.collisions * 0.6, 0, 60);
	const energyPenalty = clamp((START_ENERGY - agent.energy) / Math.max(1, START_ENERGY), 0, 1) * 25;
	const timeRef = agent.firstFoodTime ?? elapsed;
	const timePenalty = clamp(timeRef / Math.max(1, generationSeconds), 0, 1) * 20;
	// Laziness: apply 0.01 fitness penalty per second while displacement < 10 units.
	const lazyPenalty = clamp(agent.lazySeconds, 0, generationSeconds) * LAZY_FITNESS_PENALTY_PER_SEC;

	// Bonuses (can recover fitness, but we still clamp to 100)
	const fruitsBonus = clamp(agent.foodsEaten / Math.max(1, FOOD_COUNT), 0, 1) * 40;
	const energyBonus = clamp((agent.energy - START_ENERGY) / Math.max(1, MAX_ENERGY - START_ENERGY), 0, 1) * 15;
	let speedBonus = 0;
	if (agent.foodsEaten > 0) {
		const rate = agent.foodsEaten / Math.max(0.0001, elapsed);
		const targetRate = FOOD_COUNT / Math.max(1, generationSeconds);
		const rate01 = clamp(rate / Math.max(0.0001, targetRate), 0, 1);
		speedBonus = rate01 * 15;
	}

	return clamp(base - collisionPenalty - energyPenalty - timePenalty - lazyPenalty + fruitsBonus + energyBonus + speedBonus, 0, 100);
}

function breedingScore(agent: Agent, generationSeconds: number): number {
	// Fitness already includes energy + fruits + time + collisions, and is 0..100.
	return agent.fitness;
}

function selectParent(pool: Agent[], generationSeconds: number): Agent {
	const offset = 1;
	const total = pool.reduce((sum, a) => sum + breedingScore(a, generationSeconds) + offset, 0);
	let roll = Math.random() * total;
	for (const a of pool) {
		roll -= breedingScore(a, generationSeconds) + offset;
		if (roll <= 0) return a;
	}
	return pool[pool.length - 1];
}

function evolve(state: SimState, settings: EvoSettings): SimState {
	const generationSeconds = clamp(Math.round(settings.generationSeconds), 20, 60 * 60);
	const start = { x: -HALF + 8, z: -HALF + 8 };
	const { foods, obstacles } = createWorld();
	const rankedByFitness = [...state.agents].sort((a, b) => b.fitness - a.fitness);
	const bestPrev = rankedByFitness[0]?.fitness ?? 0;
	const eliteCount = clamp(settings.eliteCount, 1, rankedByFitness.length);
	const elites = rankedByFitness.slice(0, eliteCount);
	// For crossover, only choose from the top 10 most fit creatures.
	const breedingPool = rankedByFitness.slice(0, Math.min(CROSSOVER_PARENT_POOL_SIZE, rankedByFitness.length));

	const nextAgents: Agent[] = [];
	for (const elite of elites) {
		nextAgents.push(createAgent(start, foods, cloneGenome(elite.genome)));
	}

	while (nextAgents.length < settings.population) {
		const parentA = selectParent(breedingPool, generationSeconds);
		const parentB = selectParent(breedingPool, generationSeconds);
		const crossed = crossoverGenomes(parentA.genome, parentB.genome, settings.crossoverRate);
		const mutated = mutateGenome(crossed, settings);
		nextAgents.push(createAgent(start, foods, mutated));
	}

	const initialNearest = nearestFoodDistance(start, foods);
	for (const agent of nextAgents) {
		agent.minFoodDistance = initialNearest;
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

function stepSimulation(prev: SimState, dt: number, settings: EvoSettings, startPos: Vec2, visionRadius: number): SimState {
	const generationSeconds = clamp(Math.round(settings.generationSeconds), 20, 60 * 60);
	const elapsed = prev.elapsedSeconds + dt;

	const updatedAgents = prev.agents.map((agent) => {
		if (!agent.alive || agent.energy <= 0) {
			const deadAgent: Agent = {
				...agent,
				alive: false,
				energy: 0,
				vx: 0,
				vz: 0,
				fitness: 0,
			};
			deadAgent.fitness = computeFitness(deadAgent, elapsed, generationSeconds);
			return deadAgent;
		}

		const inputs = getVisionInputs(agent, prev.foods, prev.obstacles, visionRadius);
		const { hiddenLayers, out, steer, speed01 } = forwardPass(agent.genome, inputs);
		const desiredMaxSpeed = MAX_SPEED * speed01;

		const prevVx = agent.vx;
		const prevVz = agent.vz;
		let vx = agent.vx;
		let vz = agent.vz;
		let x = agent.x;
		let z = agent.z;
		let collidedThisStep = false;

		// Sub-step integration to prevent tunneling through obstacles.
		const ax = steer.x * ACCEL * speed01;
		const az = steer.z * ACCEL * speed01;
		const estSpeed = Math.sqrt(vx * vx + vz * vz);
		const maxStepDist = Math.max(AGENT_RADIUS * 0.75, 0.12);
		const steps = clamp(Math.ceil((estSpeed * dt) / maxStepDist), 1, 18);
		const dtStep = dt / steps;
		const dragFactor = Math.pow(DRAG, 1 / steps);

		for (let s = 0; s < steps; s += 1) {
			vx = (vx + ax * dtStep) * dragFactor;
			vz = (vz + az * dtStep) * dragFactor;
			const speedNow = Math.sqrt(vx * vx + vz * vz) || 1;
			if (speedNow > Math.max(desiredMaxSpeed, 0.0001)) {
				vx = (vx / speedNow) * desiredMaxSpeed;
				vz = (vz / speedNow) * desiredMaxSpeed;
			}

			const rawX = x + vx * dtStep;
			const rawZ = z + vz * dtStep;
			x = clamp(rawX, -HALF + AGENT_RADIUS, HALF - AGENT_RADIUS);
			z = clamp(rawZ, -HALF + AGENT_RADIUS, HALF - AGENT_RADIUS);
			if (x !== rawX) {
				// Remove the velocity component pushing into the wall.
				vx = 0;
				collidedThisStep = true;
			}
			if (z !== rawZ) {
				vz = 0;
				collidedThisStep = true;
			}

			for (const obstacle of prev.obstacles) {
				const dx = x - obstacle.x;
				const dz = z - obstacle.z;
				const d = Math.sqrt(dx * dx + dz * dz) || 0.0001;
				const minD = obstacle.size + AGENT_RADIUS;
				if (d < minD) {
					x = obstacle.x + (dx / d) * minD;
					z = obstacle.z + (dz / d) * minD;
					collidedThisStep = true;
				}
			}
		}

		const speed = Math.sqrt(vx * vx + vz * vz);
		const accelMag = Math.sqrt((vx - prevVx) * (vx - prevVx) + (vz - prevVz) * (vz - prevVz)) / Math.max(dt, 1e-6);
		const drainPerSec = ENERGY_DRAIN_BASE_PER_SEC + ENERGY_DRAIN_SPEED_PER_UNIT * speed + ENERGY_DRAIN_ACCEL_PER_UNIT * accelMag;
		let energy = clamp(agent.energy - drainPerSec * dt - (collidedThisStep ? COLLISION_ENERGY_LOSS : 0), 0, MAX_ENERGY);
		let alive = energy > 0;
		if (!alive) {
			vx = 0;
			vz = 0;
		}

		let headingX = agent.headingX;
		let headingZ = agent.headingZ;
		const vmag = Math.sqrt(vx * vx + vz * vz);
		if (vmag > 0.02) {
			headingX = vx / vmag;
			headingZ = vz / vmag;
		}

		let collisions = agent.collisions + (collidedThisStep ? 1 : 0);

		let foodsEaten = agent.foodsEaten;
		let firstFoodTime = agent.firstFoodTime;
		const eatenMask = [...agent.eatenMask];

		for (let i = 0; i < prev.foods.length; i += 1) {
			if (eatenMask[i]) continue;
			if (dist({ x, z }, prev.foods[i]) < FOOD_RADIUS + AGENT_RADIUS) {
				eatenMask[i] = true;
				foodsEaten += 1;
				energy = clamp(energy + ENERGY_GAIN_PER_FOOD, 0, MAX_ENERGY);
				alive = energy > 0;
				if (firstFoodTime == null) firstFoodTime = elapsed;
			}
		}

		const minFoodDistance = Math.min(agent.minFoodDistance, nearestFoodDistance({ x, z }, prev.foods));
		// Reward signal for learning: progress toward food and successful eating.
		const foodsEatenDelta = foodsEaten - agent.foodsEaten;
		const distDelta = agent.minFoodDistance - minFoodDistance;
		const reward = clamp(distDelta * 0.25 + foodsEatenDelta * 1.2 - (collidedThisStep ? 0.05 : 0), -1, 1);

		const learningEnabled = settings.learningEnabled ?? false;
		const learningRate = settings.learningRate ?? 0;
		const nextGenome = learningEnabled && learningRate > 0 ? learnGenome(agent.genome, inputs, hiddenLayers, out, reward, learningRate) : agent.genome;

		const displacement = dist({ x, z }, { x: agent.spawnX, z: agent.spawnZ });
		const lazySeconds = agent.lazySeconds + (displacement < LAZY_DISTANCE_THRESHOLD ? dt : 0);

		const nextAgent: Agent = {
			...agent,
			x,
			z,
			lazySeconds,
			vx,
			vz,
			headingX,
			headingZ,
			energy,
			alive,
			collisions,
			foodsEaten,
			firstFoodTime,
			eatenMask,
			minFoodDistance,
			fitness: 0,
			genome: nextGenome,
			brain: {
				inputs,
				hidden: hiddenLayers,
				outputs: out,
			},
		};
		nextAgent.fitness = computeFitness(nextAgent, elapsed, generationSeconds);
		return nextAgent;
	});

	const next: SimState = { ...prev, elapsedSeconds: elapsed, agents: updatedAgents };
	if (elapsed >= generationSeconds) return evolve(next, settings);
	return next;
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

function BabylonWorld({
	state,
	visionRadius,
	showPath,
	selectedAgentId,
	onSelectAgent,
	onAddFood,
	onAddObstacle,
}: {
	state: SimState;
	visionRadius: number;
	showPath: boolean;
	selectedAgentId: string | null;
	onSelectAgent: (id: string) => void;
	onAddFood: (pos: Vec2) => void;
	onAddObstacle: (pos: Vec2) => void;
}) {
	const canvasRef = useRef<HTMLCanvasElement | null>(null);
	const stateRef = useRef(state);
	const selectedIdRef = useRef<string | null>(selectedAgentId);
	const onSelectAgentRef = useRef(onSelectAgent);
	const onAddFoodRef = useRef(onAddFood);
	const onAddObstacleRef = useRef(onAddObstacle);
	const visionRadiusRef = useRef<number>(visionRadius);
	const showPathRef = useRef<boolean>(showPath);
	const pointerGroundRef = useRef<Vec2 | null>(null);

	useEffect(() => {
		stateRef.current = state;
	}, [state]);

	useEffect(() => {
		selectedIdRef.current = selectedAgentId;
	}, [selectedAgentId]);

	useEffect(() => {
		onSelectAgentRef.current = onSelectAgent;
	}, [onSelectAgent]);

	useEffect(() => {
		onAddFoodRef.current = onAddFood;
	}, [onAddFood]);

	useEffect(() => {
		onAddObstacleRef.current = onAddObstacle;
	}, [onAddObstacle]);

	useEffect(() => {
		visionRadiusRef.current = visionRadius;
	}, [visionRadius]);

	useEffect(() => {
		showPathRef.current = showPath;
	}, [showPath]);

	const createAgentTriangle = (name: string, scene: BABYLON.Scene) => {
		// A small 3D wedge (tapered triangular prism) pointing along +Z.
		// We'll rotate it around Y to face its heading.
		const mesh = new BABYLON.Mesh(name, scene);
		const tipZ = 0.75;
		const baseZ = -0.42;
		const halfW = 0.42;
		const thicknessY = 0.34;
		const taper = 0.78;

		const positions = [
			// bottom (full size)
			0, 0, tipZ,
			-halfW, 0, baseZ,
			halfW, 0, baseZ,
			// top (slightly tapered so it reads as 3D)
			0, thicknessY, tipZ * (0.85 + taper * 0.15),
			-halfW * taper, thicknessY, baseZ * (0.85 + taper * 0.15),
			halfW * taper, thicknessY, baseZ * (0.85 + taper * 0.15),
		];

		const indices = [
			// bottom
			0, 1, 2,
			// top (reverse winding)
			3, 5, 4,
			// sides
			0, 3, 4,
			0, 4, 1,
			1, 4, 5,
			1, 5, 2,
			2, 5, 3,
			2, 3, 0,
		];

		const normals: number[] = [];
		BABYLON.VertexData.ComputeNormals(positions, indices, normals);
		const vd = new BABYLON.VertexData();
		vd.positions = positions;
		vd.indices = indices;
		vd.normals = normals;
		vd.applyToMesh(mesh);
		mesh.isPickable = true;
		return mesh;
	};

	useEffect(() => {
		if (!canvasRef.current) return;

		const engine = new BABYLON.Engine(canvasRef.current, true, { preserveDrawingBuffer: true, stencil: true });
		const scene = new BABYLON.Scene(engine);
		scene.clearColor = BABYLON.Color4.FromHexString("#020617FF");

		const camera = new BABYLON.ArcRotateCamera("camera", -Math.PI / 4, 1.05, 66, BABYLON.Vector3.Zero(), scene);
		camera.attachControl(canvasRef.current, true);
		camera.lowerRadiusLimit = 10;
		camera.upperRadiusLimit = 120;

		const hemi = new BABYLON.HemisphericLight("ambient", new BABYLON.Vector3(0, 1, 0), scene);
		hemi.intensity = 0.95;
		hemi.specular = BABYLON.Color3.Black();
		const sun = new BABYLON.DirectionalLight("sun", new BABYLON.Vector3(-0.5, -1, -0.4), scene);
		sun.position = new BABYLON.Vector3(20, 24, 16);
		sun.intensity = 1.15;
		sun.specular = BABYLON.Color3.Black();

		const ground = BABYLON.MeshBuilder.CreateGround("ground", { width: WORLD_SIZE, height: WORLD_SIZE }, scene);
		ground.isPickable = true;
		const groundMat = new BABYLON.StandardMaterial("groundMat", scene);
		groundMat.diffuseColor = BABYLON.Color3.FromHexString("#102f24");
		groundMat.specularColor = BABYLON.Color3.Black();
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

		// Visible boundary walls (agents treat edges like obstacles)
		const wallMat = new BABYLON.StandardMaterial("wallMat", scene);
		wallMat.diffuseColor = BABYLON.Color3.FromHexString("#334155");
		wallMat.emissiveColor = BABYLON.Color3.FromHexString("#0b1220");
		wallMat.specularColor = BABYLON.Color3.Black();
		const wallH = 3.4;
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

		const obstacleMeshes = new Map<string, BABYLON.Mesh>();
		const foodMeshes = new Map<string, BABYLON.Mesh>();
		const agentMeshes = new Map<string, BABYLON.Mesh>();

		// Vision ring shown around the currently selected agent.
		const visionRing = BABYLON.MeshBuilder.CreateTorus(
			"vision-ring",
			{ diameter: DEFAULT_VISION_RADIUS * 2, thickness: 0.05, tessellation: 64 },
			scene,
		);
		visionRing.isPickable = false;
		visionRing.position.y = 0.09;
		visionRing.renderingGroupId = 1;
		const ringMat = new BABYLON.StandardMaterial("visionRingMat", scene);
		ringMat.emissiveColor = BABYLON.Color3.FromHexString("#38bdf8");
		ringMat.diffuseColor = BABYLON.Color3.FromHexString("#0ea5e9");
		ringMat.alpha = 0.55;
		ringMat.specularColor = BABYLON.Color3.Black();
		visionRing.material = ringMat;
		visionRing.setEnabled(false);

		// 120° viewing segment around the selected creature (thick tube so it's visible).
		let fovTube: BABYLON.Mesh | null = null;
		const fovTubeMat = new BABYLON.StandardMaterial("fovTubeMat", scene);
		fovTubeMat.emissiveColor = BABYLON.Color3.FromHexString("#a78bfa");
		fovTubeMat.diffuseColor = BABYLON.Color3.FromHexString("#7c3aed");
		fovTubeMat.alpha = 0.85;
		fovTubeMat.specularColor = BABYLON.Color3.Black();

		// Optional trails for all creatures.
		type PathEntry = {
			tube: BABYLON.Mesh | null;
			points: BABYLON.Vector3[];
			lastPos: BABYLON.Vector3 | null;
		};
		const pathsByAgent = new Map<string, PathEntry>();
		let lastPathsVisible = false;
		const PATH_MAX_POINTS = 320;
		const PATH_MIN_DIST = 0.35;
		const PATH_TUBE_RADIUS = 0.055;
		const pathMat = new BABYLON.StandardMaterial("pathMat", scene);
		pathMat.emissiveColor = BABYLON.Color3.FromHexString("#38bdf8");
		pathMat.diffuseColor = BABYLON.Color3.FromHexString("#0ea5e9");
		pathMat.alpha = 0.7;
		pathMat.specularColor = BABYLON.Color3.Black();

		const obstacleMat = new BABYLON.StandardMaterial("obstacleMat", scene);
		obstacleMat.diffuseColor = BABYLON.Color3.FromHexString("#64748b");
		obstacleMat.specularColor = BABYLON.Color3.Black();
		const foodMat = new BABYLON.StandardMaterial("foodMat", scene);
		foodMat.diffuseColor = BABYLON.Color3.FromHexString("#facc15");
		foodMat.emissiveColor = BABYLON.Color3.FromHexString("#854d0e");
		foodMat.specularColor = BABYLON.Color3.Black();

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
			const ranked = [...agents].sort((a, b) => b.fitness - a.fitness);
			const best = ranked[0] ?? null;
			const top10 = new Set(ranked.slice(0, 10).map((a) => a.id));
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
					mesh = createAgentTriangle(`agent-${a.id}`, scene);
					mesh.metadata = { agentId: a.id };
					agentMeshes.set(a.id, mesh);
				}

				// Face the persistent heading (stable even at low speed).
				const yaw = Math.atan2(a.headingX, a.headingZ);
				mesh.rotation.set(0, yaw, 0);
				mesh.position.set(a.x, 0.02, a.z);

				const isBest = best?.id === a.id;
				const isTop10 = top10.has(a.id);
				const hue = clamp(120 - a.foodsEaten * 16 - a.collisions * 0.6, 0, 120);
				const aliveColor = isBest
					? BABYLON.Color3.FromHexString("#22d3ee")
					: isTop10
						? BABYLON.Color3.FromHexString("#38bdf8")
						: hslToColor3(hue, 82, 48);
				const color = a.alive ? aliveColor : BABYLON.Color3.FromHexString("#334155");
				let mat = mesh.material as BABYLON.StandardMaterial | null;
				if (!mat) {
					mat = new BABYLON.StandardMaterial(`agent-mat-${a.id}`, scene);
					mat.specularColor = BABYLON.Color3.Black();
					mesh.material = mat;
				}
				mat.diffuseColor = color;
				mat.alpha = a.alive ? 1 : 0.55;
			}

			// Update path history for all agents (tracks even when rendering is disabled).
			for (const [id, entry] of pathsByAgent.entries()) {
				if (!ids.has(id)) {
					if (entry.tube) {
						try {
							entry.tube.dispose();
						} catch {
							// ignore
						}
					}
					pathsByAgent.delete(id);
				}
			}

			const y = 0.12;
			const pathsVisible = Boolean(showPathRef.current);
			for (const a of agents) {
				let entry = pathsByAgent.get(a.id);
				if (!entry) {
					const p0 = new BABYLON.Vector3(a.x, y, a.z);
					entry = { tube: null, points: [p0], lastPos: p0.clone() };
					pathsByAgent.set(a.id, entry);
				}

				const p = new BABYLON.Vector3(a.x, y, a.z);
				const lastPos = entry.lastPos;
				const moved =
					!lastPos ||
					Math.sqrt((p.x - lastPos.x) * (p.x - lastPos.x) + (p.z - lastPos.z) * (p.z - lastPos.z)) >= PATH_MIN_DIST;
				if (moved) {
					entry.points.push(p);
					entry.lastPos = p;
					if (entry.points.length > PATH_MAX_POINTS) entry.points.splice(0, entry.points.length - PATH_MAX_POINTS);
				}

				if (!pathsVisible) continue;
				const safePath = entry.points.filter(
					(v): v is BABYLON.Vector3 =>
						Boolean(v) &&
						Number.isFinite((v as BABYLON.Vector3).x) &&
						Number.isFinite((v as BABYLON.Vector3).y) &&
						Number.isFinite((v as BABYLON.Vector3).z),
				);
				if (safePath.length < 2) continue;

				const needsCreate = !entry.tube || (entry.tube as any).isDisposed?.() === true;
				if (needsCreate) {
					if (entry.tube) {
						try {
							entry.tube.dispose();
						} catch {
							// ignore
						}
						entry.tube = null;
					}
					entry.tube = BABYLON.MeshBuilder.CreateTube(
						`path-${a.id}`,
						{ path: safePath, radius: PATH_TUBE_RADIUS, tessellation: 8, cap: BABYLON.Mesh.CAP_ALL, updatable: true },
						scene,
					);
					entry.tube.isPickable = false;
					entry.tube.material = pathMat;
					entry.tube.renderingGroupId = 1;
					entry.tube.setEnabled(true);
				} else if (moved) {
					try {
						BABYLON.MeshBuilder.CreateTube(
							`path-${a.id}`,
							{
								path: safePath,
								instance: entry.tube ?? undefined,
								radius: PATH_TUBE_RADIUS,
								tessellation: 8,
								cap: BABYLON.Mesh.CAP_ALL,
								updatable: true,
							},
							scene,
						);
						entry.tube?.setEnabled(true);
					} catch {
						try {
							entry.tube?.dispose();
						} catch {
							// ignore
						}
						entry.tube = null;
					}
				}
			}

			// If trails were just disabled, hide all existing tubes in one pass.
			if (lastPathsVisible && !pathsVisible) {
				for (const entry of pathsByAgent.values()) entry.tube?.setEnabled(false);
			}
			lastPathsVisible = pathsVisible;

			// Update vision ring around selected creature.
			const selectedId = selectedIdRef.current;
			if (!selectedId) {
				visionRing.setEnabled(false);
				if (fovTube) fovTube.setEnabled(false);
				return;
			}
			const selected = agents.find((a) => a.id === selectedId);
			if (!selected) {
				visionRing.setEnabled(false);
				if (fovTube) fovTube.setEnabled(false);
				return;
			}
			visionRing.setEnabled(true);
			const r = Math.max(0.1, visionRadiusRef.current);
			const scale = r / DEFAULT_VISION_RADIUS;
			visionRing.scaling.set(scale, 1, scale);
			// Defensive: in dev StrictMode an old render loop can briefly run after dispose.
			// Ensure the vector properties exist before writing into them.
			if (!(visionRing as any).position) (visionRing as any).position = new BABYLON.Vector3(0, 0, 0);
			(visionRing as any).position.x = selected.x;
			(visionRing as any).position.z = selected.z;

			// 120° viewing segment path in world space.
			const cx = selected.x;
			const cz = selected.z;
			const cy = 0.14;
			const hx = selected.headingX;
			const hz = selected.headingZ;
			const angleLeft = -VISION_HALF_FOV_RAD;
			const angleRight = VISION_HALF_FOV_RAD;
			const segs = 28;
			const path: BABYLON.Vector3[] = [];
			const center = new BABYLON.Vector3(cx, cy, cz);

			const rotate2 = (x: number, z: number, ang: number) => {
				const c = Math.cos(ang);
				const s = Math.sin(ang);
				return { x: x * c - z * s, z: x * s + z * c };
			};

			const leftDir = rotate2(hx, hz, angleLeft);
			const rightDir = rotate2(hx, hz, angleRight);
			const pLeft = new BABYLON.Vector3(cx + leftDir.x * r, cy, cz + leftDir.z * r);
			const pRight = new BABYLON.Vector3(cx + rightDir.x * r, cy, cz + rightDir.z * r);

			// center -> left
			path.push(center, pLeft);
			// arc from left to right
			for (let i = 0; i <= segs; i += 1) {
				const t = i / segs;
				const ang = angleLeft + (angleRight - angleLeft) * t;
				const d = rotate2(hx, hz, ang);
				path.push(new BABYLON.Vector3(cx + d.x * r, cy, cz + d.z * r));
			}
			// right -> center
			path.push(pRight, center);

			if (!fovTube) {
				fovTube = BABYLON.MeshBuilder.CreateTube(
					"fov-tube",
					{ path, radius: 0.06, tessellation: 12, cap: BABYLON.Mesh.CAP_ALL, updatable: true },
					scene,
				);
				fovTube.isPickable = false;
				fovTube.material = fovTubeMat;
				fovTube.renderingGroupId = 1;
				fovTube.setEnabled(true);
			} else {
				BABYLON.MeshBuilder.CreateTube("fov-tube", { path, instance: fovTube });
				fovTube.setEnabled(true);
			}
		};

		scene.onPointerObservable.add((eventInfo) => {
			if (eventInfo.type === BABYLON.PointerEventTypes.POINTERMOVE) {
				const pick = scene.pick(scene.pointerX, scene.pointerY, (m) => m === ground);
				if (pick?.hit && pick.pickedPoint) {
					pointerGroundRef.current = { x: pick.pickedPoint.x, z: pick.pickedPoint.z };
				}
				return;
			}

			if (eventInfo.type !== BABYLON.PointerEventTypes.POINTERPICK) return;
			const pick = eventInfo.pickInfo;
			if (!pick?.hit || !pick.pickedMesh) return;
			const agentId = (pick.pickedMesh as any).metadata?.agentId;
			if (typeof agentId === "string") {
				onSelectAgentRef.current(agentId);
			}
		});

		const onKeyDown = (e: KeyboardEvent) => {
			const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase?.() ?? "";
			if (tag === "input" || tag === "textarea" || tag === "select") return;

			const key = e.key?.toLowerCase?.();
			if (key !== "f" && key !== "o") return;
			const p = pointerGroundRef.current;
			if (!p) return;
			if (key === "f") onAddFoodRef.current(p);
			if (key === "o") onAddObstacleRef.current(p);
		};
		window.addEventListener("keydown", onKeyDown);

		// Performance: only sync meshes when the underlying arrays or relevant control
		// values change. Rendering still happens every frame for smooth camera input.
		let lastObstacles: Obstacle[] | null = null;
		let lastFoods: Food[] | null = null;
		let lastAgents: Agent[] | null = null;
		let lastSelected: string | null = null;
		let lastShowPath = false;
		let lastVisionRadius = -1;

		engine.runRenderLoop(() => {
			const s = stateRef.current;

			if (s.obstacles !== lastObstacles) {
				syncObstacles(s.obstacles);
				lastObstacles = s.obstacles;
			}
			if (s.foods !== lastFoods) {
				syncFoods(s.foods);
				lastFoods = s.foods;
			}

			const selected = selectedIdRef.current;
			const show = Boolean(showPathRef.current);
			const vr = visionRadiusRef.current;
			const needAgentSync =
				s.agents !== lastAgents ||
				selected !== lastSelected ||
				show !== lastShowPath ||
				vr !== lastVisionRadius;
			if (needAgentSync) {
				syncAgents(s.agents);
				lastAgents = s.agents;
				lastSelected = selected;
				lastShowPath = show;
				lastVisionRadius = vr;
			}

			scene.render();
		});

		const onResize = () => engine.resize();
		window.addEventListener("resize", onResize);

		return () => {
			engine.stopRenderLoop();
			window.removeEventListener("resize", onResize);
			window.removeEventListener("keydown", onKeyDown);
			engine.dispose();
		};
	}, []);

	return <canvas ref={canvasRef} className="fixed inset-0 block h-screen w-screen" />;
}

export default function NeuralVisionSim3D() {
	const [settings, setSettings] = useState<EvoSettings>({
		population: 100,
		mutationRate: 0.08,
		mutationStrength: 0.3,
		eliteCount: 10,
		topologyMutationRate: 0.06,
		crossoverRate: 0.8,
		generationSeconds: DEFAULT_GENERATION_SECONDS,
		// Optional learning (Lamarckian): creatures can slightly adjust their own weights during a generation.
		learningEnabled: true,
		learningRate: 0.03,
	});
	const [paused, setPaused] = useState(false);
	const [visionRadius, setVisionRadius] = useState<number>(DEFAULT_VISION_RADIUS);
	const [showPath, setShowPath] = useState<boolean>(false);
	const [genomeLoadError, setGenomeLoadError] = useState<string>("");
	const [state, setState] = useState<SimState>(() => createInitialState(settings));
	const startPos = useMemo(() => ({ x: -HALF + 8, z: -HALF + 8 }), []);

	const addFoodAt = (pos: Vec2) => {
		setState((prev) => {
			const x = clamp(pos.x, -HALF + 1, HALF - 1);
			const z = clamp(pos.z, -HALF + 1, HALF - 1);
			// Avoid stacking foods directly on top of obstacles.
			for (const o of prev.obstacles) {
				if (dist({ x, z }, o) < o.size + FOOD_RADIUS + 0.25) return prev;
			}
			const food: Food = { id: randomId("food"), x, z };
			return {
				...prev,
				foods: [...prev.foods, food],
				agents: prev.agents.map((a) => ({ ...a, eatenMask: [...a.eatenMask, false] })),
			};
		});
	};

	const addObstacleAt = (pos: Vec2) => {
		setState((prev) => {
			const start = { x: -HALF + 8, z: -HALF + 8 };
			const size = rand(1.2, 2.8);
			const height = rand(1.5, 4.2);
			const x = clamp(pos.x, -HALF + 8, HALF - 8);
			const z = clamp(pos.z, -HALF + 8, HALF - 8);
			// Respect the spawn clear zone.
			const clearance = START_CLEAR_RADIUS + size + AGENT_RADIUS;
			if (dist({ x, z }, start) < clearance) return prev;
			// Avoid overlapping existing obstacles and foods.
			for (const o of prev.obstacles) {
				if (dist({ x, z }, o) < o.size + size + 0.35) return prev;
			}
			for (const f of prev.foods) {
				if (dist({ x, z }, f) < size + FOOD_RADIUS + 0.25) return prev;
			}
			const obstacle: Obstacle = { id: randomId("obs"), x, z, size, height };
			return { ...prev, obstacles: [...prev.obstacles, obstacle] };
		});
	};

	useEffect(() => {
		let raf = 0;
		let last = performance.now();
		let acc = 0;
		// Throttle simulation + React/UI updates to reduce CPU and prevent long-run
		// instability/crashes from excessive re-renders.
		const UI_HZ = 20;
		const UI_DT = 1 / UI_HZ;

		const tick = (now: number) => {
			const dt = Math.min((now - last) / 1000, 0.04);
			last = now;

			if (!paused) {
				acc += dt;
				if (acc >= UI_DT) {
					const stepTotal = acc;
					acc = 0;
					setState((prev) => {
						let next = prev;
						let remaining = stepTotal;
						// Chunk large dt to keep physics stable.
						while (remaining > 0) {
							const chunk = Math.min(remaining, 0.04);
							next = stepSimulation(next, chunk, settings, startPos, visionRadius);
							remaining -= chunk;
						}
						return next;
					});
				}
			} else {
				acc = 0;
			}

			raf = window.requestAnimationFrame(tick);
		};

		raf = window.requestAnimationFrame(tick);
		return () => window.cancelAnimationFrame(raf);
	}, [paused, settings, startPos, visionRadius]);

	const bestCurrent = useMemo(() => {
		return [...state.agents].sort((a, b) => b.fitness - a.fitness)[0] ?? null;
	}, [state.agents]);

	const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
	const [controlsCollapsed, setControlsCollapsed] = useState(false);

	useEffect(() => {
		// Keep a stable selection. If nothing is selected yet, follow the current best.
		if (!selectedAgentId && bestCurrent) {
			setSelectedAgentId(bestCurrent.id);
			return;
		}
		if (selectedAgentId && !state.agents.some((a) => a.id === selectedAgentId)) {
			setSelectedAgentId(bestCurrent?.id ?? null);
		}
	}, [bestCurrent, selectedAgentId, state.agents]);

	const selectedAgent = useMemo(() => {
		return state.agents.find((a) => a.id === selectedAgentId) ?? bestCurrent;
	}, [bestCurrent, selectedAgentId, state.agents]);

	const topAgents = useMemo(() => {
		return [...state.agents].sort((a, b) => b.fitness - a.fitness).slice(0, 12);
	}, [state.agents]);

	const selectableAgents = useMemo(() => {
		if (!selectedAgent) return topAgents;
		if (topAgents.some((a) => a.id === selectedAgent.id)) return topAgents;
		return [selectedAgent, ...topAgents];
	}, [selectedAgent, topAgents]);

	const restart = (seedGenomes?: BrainGenome[]) => {
		setState(createInitialState(settings, seedGenomes));
	};

	const saveTop10 = () => {
		const top = [...state.agents]
			.sort((a, b) => b.fitness - a.fitness)
			.slice(0, 10)
			.map((a) => ({
				fitness: a.fitness,
				energy: a.energy,
				foodsEaten: a.foodsEaten,
				firstFoodTime: a.firstFoodTime,
				genome: a.genome,
			}));

		const payload: SavedSim3GenomesV1 = {
			version: 1,
			sim: "sim3",
			createdAt: new Date().toISOString(),
			net: {
				inputs: INPUTS,
				outputs: OUTPUTS,
			},
			top,
		};
		downloadJson("sim3-top10-genomes.json", payload);
	};

	const loadTop10FromFile = async (file: File) => {
		setGenomeLoadError("");
		try {
			const text = await file.text();
			const json = JSON.parse(text);
			let genomes: BrainGenome[] = [];
			if (isSavedSim3GenomesV1(json)) {
				genomes = json.top.map((e) => e.genome).filter(Boolean);
			} else if (Array.isArray(json)) {
				// Allow loading a raw array of genomes.
				genomes = json.filter(Boolean);
			} else {
				throw new Error("Invalid file format. Expected a Sim 3 genomes export.");
			}

			const safe = genomes.map((g) => ensureGenomeInputShape(g));
			restart(safe);
			setSelectedAgentId(null);
		} catch (e: any) {
			setGenomeLoadError(String(e?.message ?? e));
		}
	};

	return (
		<>
			<BabylonWorld
				state={state}
				visionRadius={visionRadius}
				showPath={showPath}
				selectedAgentId={selectedAgentId}
				onSelectAgent={setSelectedAgentId}
				onAddFood={addFoodAt}
				onAddObstacle={addObstacleAt}
			/>
			<aside className="controls-panel">
				<Card className="bg-transparent border-0 shadow-none">
					<div className="controls-title" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
						<span>Controls</span>
						<button onClick={() => setControlsCollapsed((v) => !v)}>{controlsCollapsed ? "Expand" : "Collapse"}</button>
					</div>
					{!controlsCollapsed && (
						<CardContent className="space-y-4">
							<div className="grid grid-cols-2 gap-2">
								<Button onClick={() => setPaused((p) => !p)}>{paused ? "Resume" : "Pause"}</Button>
								<Button variant="secondary" onClick={() => restart()}>Restart</Button>
							</div>

							<div className="rounded-xl border border-slate-700 p-3 text-sm space-y-1">
								<div>Generation: <strong>{state.generation}</strong></div>
								<div>Timer: <strong>{state.elapsedSeconds.toFixed(1)}s / {Math.round(settings.generationSeconds)}s</strong></div>
								<div>Population: <strong>{state.agents.length}</strong></div>
								<div>Vision Radius: <strong>{visionRadius.toFixed(1)} units</strong></div>
								<div>Best (current): <strong>{bestCurrent ? bestCurrent.fitness.toFixed(1) : "0.0"}</strong></div>
								<div>Best (prev gen): <strong>{state.bestLastGeneration.toFixed(1)}</strong></div>
								<div>Best (ever): <strong>{state.bestEverFitness.toFixed(1)}</strong></div>
								<div>Best eaten: <strong>{bestCurrent?.foodsEaten ?? 0}</strong></div>
								<div>Best hidden layers: <strong>{bestCurrent?.genome.hiddenLayers ?? 0}</strong></div>
								<div>Best hidden neurons: <strong>{bestCurrent?.genome.hiddenSize ?? 0}</strong></div>
							</div>

							<div className="space-y-3">
								<div>
									<div className="flex justify-between text-sm mb-1"><span>Vision Radius</span><span>{visionRadius.toFixed(1)}</span></div>
									<Slider value={[visionRadius]} min={1} max={16} step={0.5} onValueChange={(v: number[]) => setVisionRadius(v[0])} />
								</div>
								<div>
									<div className="flex justify-between text-sm mb-1"><span>Iteration Duration</span><span>{Math.round(settings.generationSeconds)}s</span></div>
									<Slider
										value={[settings.generationSeconds]}
										min={60}
										max={600}
										step={15}
										onValueChange={(v: number[]) =>
											setSettings((p) => ({ ...p, generationSeconds: Math.round(clamp(v[0], 20, 60 * 60)) }))
										}
									/>
									<div className="text-xs text-slate-300 mt-1">Default: 180s (3 min). No restart needed.</div>
								</div>
								<div>
									<div className="flex justify-between text-sm mb-1"><span>Paths</span><span>{showPath ? "On" : "Off"}</span></div>
									<label className="flex items-center justify-between rounded-xl border border-slate-700 p-3 text-sm">
										<span>Show paths (all creatures)</span>
										<input type="checkbox" checked={showPath} onChange={(e) => setShowPath(e.target.checked)} />
									</label>
								</div>
								<div>
									<div className="flex justify-between text-sm mb-1"><span>Learning</span><span>{settings.learningEnabled ? "On" : "Off"}</span></div>
									<label className="flex items-center justify-between rounded-xl border border-slate-700 p-3 text-sm">
										<span>Enable learning</span>
										<input
											type="checkbox"
											checked={Boolean(settings.learningEnabled)}
											onChange={(e) => setSettings((p) => ({ ...p, learningEnabled: e.target.checked }))}
										/>
									</label>
								</div>
								<div>
									<div className="flex justify-between text-sm mb-1"><span>Learning Rate</span><span>{(settings.learningRate ?? 0).toFixed(3)}</span></div>
									<Slider value={[settings.learningRate ?? 0]} min={0} max={0.12} step={0.005} onValueChange={(v: number[]) => setSettings((p) => ({ ...p, learningRate: v[0] }))} />
								</div>
								<div>
									<div className="flex justify-between text-sm mb-1"><span>Mutation Rate</span><span>{settings.mutationRate.toFixed(2)}</span></div>
									<Slider value={[settings.mutationRate]} min={0} max={0.45} step={0.01} onValueChange={(v: number[]) => setSettings((p) => ({ ...p, mutationRate: v[0] }))} />
								</div>
								<div>
									<div className="flex justify-between text-sm mb-1"><span>Mutation Strength</span><span>{settings.mutationStrength.toFixed(2)}</span></div>
									<Slider value={[settings.mutationStrength]} min={0.05} max={1.2} step={0.01} onValueChange={(v: number[]) => setSettings((p) => ({ ...p, mutationStrength: v[0] }))} />
								</div>
								<div>
									<div className="flex justify-between text-sm mb-1"><span>Topology Mutation</span><span>{settings.topologyMutationRate.toFixed(2)}</span></div>
									<Slider value={[settings.topologyMutationRate]} min={0} max={0.25} step={0.01} onValueChange={(v: number[]) => setSettings((p) => ({ ...p, topologyMutationRate: v[0] }))} />
								</div>
								<div>
									<div className="flex justify-between text-sm mb-1"><span>Crossover Rate</span><span>{settings.crossoverRate.toFixed(2)}</span></div>
									<Slider value={[settings.crossoverRate]} min={0.1} max={1} step={0.01} onValueChange={(v: number[]) => setSettings((p) => ({ ...p, crossoverRate: v[0] }))} />
								</div>
								<div>
									<div className="flex justify-between text-sm mb-1"><span>Elite Count</span><span>{settings.eliteCount}</span></div>
									<Slider value={[settings.eliteCount]} min={2} max={30} step={1} onValueChange={(v: number[]) => setSettings((p) => ({ ...p, eliteCount: Math.round(v[0]) }))} />
								</div>
								<div>
									<div className="flex justify-between text-sm mb-1"><span>Population (restart)</span><span>{settings.population}</span></div>
									<Slider value={[settings.population]} min={40} max={180} step={1} onValueChange={(v: number[]) => setSettings((p) => ({ ...p, population: Math.round(v[0]) }))} />
								</div>
							</div>

							<div className="rounded-xl border border-slate-700 p-3 text-sm space-y-2">
								<div className="font-semibold">Genomes</div>
								<div className="grid grid-cols-2 gap-2">
									<Button variant="secondary" onClick={saveTop10}>Save best 10</Button>
									<label className="w-full">
										<input
											type="file"
											accept="application/json,.json"
											className="w-full rounded-md border border-slate-700 bg-slate-900/50 p-2 text-sm"
											onChange={(e) => {
												const f = e.target.files?.[0];
												if (f) void loadTop10FromFile(f);
												// allow re-selecting the same file
												e.target.value = "";
											}}
										/>
									</label>
								</div>
								<div className="text-xs text-slate-300">
									Save downloads a JSON file. Loading a JSON file restarts the sim seeded with those genomes.
								</div>
								{genomeLoadError ? <div className="text-xs text-red-300">{genomeLoadError}</div> : null}
							</div>

							<div className="rounded-xl border border-slate-700 p-3 text-sm space-y-2">
								<div className="font-semibold">Selected creature</div>
								<select
									className="w-full rounded-md border border-slate-700 bg-slate-900/50 p-2 text-sm"
									value={selectedAgent?.id ?? ""}
									onChange={(e) => setSelectedAgentId(e.target.value)}
								>
									{selectableAgents.map((a) => (
										<option key={a.id} value={a.id}>
											{a.id.slice(0, 8)} · fit {a.fitness.toFixed(0)} · eaten {a.foodsEaten}
										</option>
									))}
								</select>
								{selectedAgent && (
									<>
										<div className="text-xs text-slate-300">Hidden layers: {selectedAgent.genome.hiddenLayers} · neurons/layer: {selectedAgent.genome.hiddenSize}</div>
										<div className="text-xs text-slate-300">Energy: {selectedAgent.energy.toFixed(1)} / {MAX_ENERGY} · {selectedAgent.alive ? "Alive" : "Dead"}</div>
										<NeuralNetGraph genome={selectedAgent.genome} brain={selectedAgent.brain} />
										{/* <div className="text-xs text-slate-300">DNA (genome)</div>
										<pre className="max-h-[220px] overflow-auto rounded-lg border border-slate-700 bg-slate-950/40 p-2 text-[11px] leading-snug text-slate-200">
											{JSON.stringify(selectedAgent.genome, null, 2)}
										</pre> */}
									</>
								)}
							</div>

							<div className="rounded-xl border border-slate-700 p-3 text-xs text-slate-300 leading-relaxed">
								Each agent sees only within {visionRadius.toFixed(1)} units. A tiny neural net converts local food/obstacle signals into movement. DNA includes weights, biases, neuron count, and connection masks, all inherited and mutated each generation.
							</div>
						</CardContent>
					)}
				</Card>
			</aside>
		</>
	);
}

// Visualize a genome as a simple layered neural net graph (inputs -> hidden -> outputs).
function NeuralNetGraph({
	genome,
	brain,
}: {
	genome: BrainGenome;
	brain?: {
		inputs: number[];
		hidden: number[][];
		outputs: number[];
	};
}) {
	const width = 300;
	const height = 180;
	const padY = 14;
	const inputX = 18;
	const outputX = 282;
	const hiddenXs = Array.from({ length: genome.hiddenLayers }, (_, l) => inputX + ((l + 1) * (outputX - inputX)) / (genome.hiddenLayers + 1));

	const inputYs = Array.from({ length: INPUTS }, (_, i) => padY + (i * (height - 2 * padY)) / Math.max(1, INPUTS - 1));
	const hiddenYs = Array.from({ length: genome.hiddenSize }, (_, i) => padY + (i * (height - 2 * padY)) / Math.max(1, genome.hiddenSize - 1));
	const outputYs = Array.from({ length: OUTPUTS }, (_, i) => padY + (i * (height - 2 * padY)) / Math.max(1, OUTPUTS - 1));

	const inputLabels = ["Fx", "Fz", "Fd", "Ox", "Oz", "Od", "Vx", "Vz", "En", "Fit"];
	const outputLabels = ["L", "R", "F", "B", "S"];

	const clamp01 = (v: number) => clamp(v, 0, 1);
	const normAct = (v: number | undefined) => {
		if (!Number.isFinite(v as number)) return 0;
		// Inputs/outputs are typically in [-1,1] (or [0,1]); normalize to [-1,1].
		return clamp(Number(v), -1, 1);
	};
	const actAlpha = (v: number | undefined) => 0.15 + 0.85 * clamp01(Math.abs(normAct(v)));
	const actColor = (v: number | undefined, fallback: string) => {
		const a = normAct(v);
		if (Math.abs(a) < 1e-3) return fallback;
		return a >= 0 ? "#38bdf8" : "#fb7185";
	};
	const actRadius = (base: number, v: number | undefined) => base * (0.85 + 0.55 * clamp01(Math.abs(normAct(v))));

	const strokeForWeight = (w: number, sourceAct?: number, targetAct?: number) => {
		const abs = Math.min(1, Math.abs(w) / 2);
		const width = 0.5 + abs * 2.2;
		const color = w >= 0 ? "#38bdf8" : "#fb7185";
		const actBoost = brain ? clamp01((Math.abs(normAct(sourceAct)) + Math.abs(normAct(targetAct))) * 0.5) : 0;
		const opacity = 0.08 + abs * 0.55 + actBoost * 0.35;
		return { width, color, opacity };
	};

	return (
		<svg width={width} height={height} className="w-full rounded-lg border border-slate-700 bg-slate-950/40">
			{/* Input -> Hidden[0] connections */}
			{Array.from({ length: INPUTS }, (_, i) =>
				Array.from({ length: genome.hiddenSize }, (_, h) => {
					if (genome.inputHiddenMask[i][h] <= 0.5) return null;
					const w = genome.inputHiddenWeights[i][h];
					const s = strokeForWeight(w, brain?.inputs?.[i], brain?.hidden?.[0]?.[h]);
					return (
						<line
							key={`ih-${i}-${h}`}
							x1={inputX}
							y1={inputYs[i]}
							x2={hiddenXs[0] ?? inputX}
							y2={hiddenYs[h]}
							stroke={s.color}
							strokeWidth={s.width}
							opacity={s.opacity}
						/>
					);
				}),
			)}

			{/* Hidden[l-1] -> Hidden[l] connections */}
			{Array.from({ length: Math.max(0, genome.hiddenLayers - 1) }, (_, l) =>
				Array.from({ length: genome.hiddenSize }, (_, i) =>
					Array.from({ length: genome.hiddenSize }, (_, h) => {
						if ((genome.hiddenHiddenMask[l]?.[i]?.[h] ?? 0) <= 0.5) return null;
						const w = genome.hiddenHiddenWeights[l]?.[i]?.[h] ?? 0;
						const s = strokeForWeight(w, brain?.hidden?.[l]?.[i], brain?.hidden?.[l + 1]?.[h]);
						return (
							<line
								key={`hh-${l}-${i}-${h}`}
								x1={hiddenXs[l] ?? inputX}
								y1={hiddenYs[i]}
								x2={hiddenXs[l + 1] ?? outputX}
								y2={hiddenYs[h]}
								stroke={s.color}
								strokeWidth={s.width}
								opacity={s.opacity * 0.9}
							/>
						);
					}),
				),
			)}

			{/* Hidden[last] -> Output connections */}
			{Array.from({ length: genome.hiddenSize }, (_, h) =>
				Array.from({ length: OUTPUTS }, (_, o) => {
					if (genome.hiddenOutputMask[h][o] <= 0.5) return null;
					const w = genome.hiddenOutputWeights[h][o];
					const s = strokeForWeight(w, brain?.hidden?.[genome.hiddenLayers - 1]?.[h], brain?.outputs?.[o]);
					return (
						<line
							key={`ho-${h}-${o}`}
							x1={hiddenXs[genome.hiddenLayers - 1] ?? inputX}
							y1={hiddenYs[h]}
							x2={outputX}
							y2={outputYs[o]}
							stroke={s.color}
							strokeWidth={s.width}
							opacity={s.opacity}
						/>
					);
				}),
			)}

			{/* Nodes */}
			{inputYs.map((y, i) => (
				<g key={`in-${i}`}>
					<circle
						cx={inputX}
						cy={y}
						r={actRadius(4, brain?.inputs?.[i])}
						fill={actColor(brain?.inputs?.[i], "#94a3b8")}
						opacity={actAlpha(brain?.inputs?.[i])}
					/>
					<text
						x={inputX}
						y={y}
						textAnchor="middle"
						dominantBaseline="middle"
						fontSize={9}
						fontWeight={700}
						fill="#0b1220"
						stroke="#e2e8f0"
						strokeWidth={0.9}
						paintOrder="stroke"
						style={{ pointerEvents: "none", userSelect: "none" }}
					>
						{inputLabels[i] ?? String(i + 1)}
					</text>
				</g>
			))}
			{hiddenXs.map((x, l) =>
				hiddenYs.map((y, h) => (
					<circle
						key={`h-${l}-${h}`}
						cx={x}
						cy={y}
						r={actRadius(4, brain?.hidden?.[l]?.[h])}
						fill={actColor(brain?.hidden?.[l]?.[h], "#e2e8f0")}
						opacity={actAlpha(brain?.hidden?.[l]?.[h])}
					/>
				)),
			)}
			{outputYs.map((y, o) => (
				<g key={`out-${o}`}>
					<circle
						cx={outputX}
						cy={y}
						r={actRadius(5, brain?.outputs?.[o])}
						fill={actColor(brain?.outputs?.[o], "#facc15")}
						opacity={actAlpha(brain?.outputs?.[o])}
					/>
					<text
						x={outputX}
						y={y}
						textAnchor="middle"
						dominantBaseline="middle"
						fontSize={10}
						fontWeight={800}
						fill="#0b1220"
						stroke="#e2e8f0"
						strokeWidth={1.0}
						paintOrder="stroke"
						style={{ pointerEvents: "none", userSelect: "none" }}
					>
						{outputLabels[o] ?? String(o + 1)}
					</text>
				</g>
			))}
		</svg>
	);
}

