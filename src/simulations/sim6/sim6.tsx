import { useEffect, useRef, useState } from "react";
import * as BABYLON from "babylonjs";
import { Card, CardContent } from "../../components/ui/card";
import { Button } from "../../components/ui/button";
import { Slider } from "../../components/ui/slider";
import { ensureLmStudioModelReady, extractJsonFromText, lmStudioChatText, lmStudioResponsesToolCall, normalizeLmStudioBaseUrl } from "../../services/lmstudio";
import { replicatePredict } from "../../services/replicate";

type Vec2 = { x: number; z: number };

type Enemy = {
	id: string;
	x: number;
	z: number;
	vx: number;
	vz: number;
	alive: boolean;
};

type Rock = {
	id: string;
	x: number;
	z: number;
	vx: number;
	vz: number;
	heldByBot: boolean;
	active: boolean;
};

type Food = {
	id: string;
	x: number;
	z: number;
	active: boolean;
};

type Obstacle = {
	id: string;
	x: number;
	z: number;
	radius: number;
	height: number;
};

type Bot = {
	x: number;
	z: number;
	vx: number;
	vz: number;
	headingX: number;
	headingZ: number;
	health: number;
	maxHealth: number;
	energy: number;
	maxEnergy: number;
	organs: number;
	escaped: boolean;
};

type BotAgent = {
	id: string;
	bot: Bot;
	mesh: BABYLON.Mesh;
	action: AiAction;
	fitness: number;
	prevExitDist: number;
	prevHeadingX: number;
	prevHeadingZ: number;
	enemyKills: number;
	directionChanges: number;
	deathPenaltyApplied: boolean;
	heldRockId: string | null;
	exploredCells: Set<number>;
};

type AiProvider = "genetic" | "local" | "replicate" | "none";
type AiModel = "large-nn-ga" | "gpt-5.2" | "gemini-3" | "gemma-4";

type Sim4Settings = {
	aiProvider: AiProvider;
	aiModel: AiModel;
	localEndpointUrl: string;
	localModelName: string;
	replicateApiToken: string;
	replicateModel: string;
	visionResolution: number;
	aiHz: number;
	generationSeconds: number;
	enemySpeed: number;
	enemyDamagePerSec: number;
	botMaxSpeed: number;
	botAccel: number;
	energyDrainPerSec: number;
};

type UiSnapshot = {
	health: number;
	energy: number;
	enemiesAlive: number;
	rocksActive: number;
	organs: number;
	escaped: boolean;
	dead: boolean;
	status: string;
};

type AiExchange = {
	id: string;
	provider: AiProvider;
	model: AiModel;
	asked: string;
	replied: string;
	at: number;
	error?: string;
};

type SelectedBotSnapshot = {
	index: number;
	id: string;
	health: number;
	energy: number;
	organs: number;
	escaped: boolean;
	dead: boolean;
	fitness: number;
	enemyKills: number;
	directionChanges: number;
	action: AiAction;
	heading: { x: number; z: number; yawDeg: number };
	position: { x: number; z: number };
};

type NeuralNetSnapshot = {
	inputSize: number;
	h1Size: number;
	h2Size: number;
	outputSize: number;
	generation: number;
	genomeIndex: number;
	populationSize: number;
	fitness: number;
	evalSeconds: number;
	weightAbsMean: {
		w1: number;
		w2: number;
		w3: number;
	};
	inputs: {
		indices: number[];
		activations: number[];
	};
	hidden1: {
		indices: number[];
		activations: number[];
	};
	hidden2: {
		indices: number[];
		activations: number[];
	};
	outputs: Array<{ index: number; name: string; value: number }>;
	w1: number[][];
	w2: number[][];
	w3: number[][];
};

const WORLD_SIZE = 120;
const HALF = WORLD_SIZE / 2;
const BOT_RADIUS = 0.6;
const ENEMY_RADIUS = 0.6;
const ROCK_RADIUS = 0.28;
const FOOD_RADIUS = 0.22;
const FOOD_ENERGY_GAIN = 0.5;
const FOOD_SPAWN_PER_SEC = 2;

const EXIT_X = HALF - 6;
const EXIT_Z = HALF - 6;
const EXIT_RADIUS = 3.0;
const BOT_COUNT = 10;
const ELITE_COUNT = 3;
const TOP_GREEN_COUNT = 4;
const GUIDED_GENERATION_LIMIT = 20;

function clamp(v: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, v));
}

function rand(min: number, max: number): number {
	return Math.random() * (max - min) + min;
}

function dist2(a: Vec2, b: Vec2): number {
	const dx = a.x - b.x;
	const dz = a.z - b.z;
	return dx * dx + dz * dz;
}

function normalize(x: number, z: number): Vec2 {
	const len = Math.sqrt(x * x + z * z) || 1;
	return { x: x / len, z: z / len };
}

function randomId(prefix: string): string {
	return `${prefix}-${crypto.randomUUID()}`;
}

type AiAction = {
	left: number;
	right: number;
	forward: number;
	backward: number;
	speed: number;
	pick?: boolean;
	throw?: boolean;
	grow?: boolean;
	seesGreenCheckpoint?: boolean;
	seesStone?: boolean;
	seesRedEnemy?: boolean;
};

function round3(v: number): number {
	return Math.round(v * 1000) / 1000;
}

type FitnessEvent = {
	survivalSec?: number;
	underThreat?: boolean;
	dodgedThreat?: boolean;
	damageTaken?: number;
	enemyHit?: boolean;
	successfulThrow?: boolean;
	enemyKill?: boolean;
	discoveredStone?: boolean;
	pickedStone?: boolean;
	explorationDelta?: number;
	progressDelta?: number;
	ateFood?: boolean;
	wastedThrow?: boolean;
	stuck?: boolean;
	idle?: boolean;
	died?: boolean;
};

function computeFitness(currentFitness: number, event: FitnessEvent): number {
	let next = currentFitness;

	// Always reward survival a little
	if (typeof event.survivalSec === "number" && Number.isFinite(event.survivalSec)) {
		next += Math.max(0, event.survivalSec) * 0.2;
	}

	// Survival is the top priority under danger
	if (event.underThreat) {
		if (event.dodgedThreat) next += 2.0;
		if (typeof event.damageTaken === "number") next -= event.damageTaken * 0.8;
		if (event.enemyHit) next += 2.0;
		if (event.successfulThrow) next += 3.0;
		if (event.enemyKill) next += 8.0;
	}

	// Exploration: always reward discovering new areas, regardless of threat
	if (typeof event.explorationDelta === "number") {
		next += event.explorationDelta * 4.0;
	}

	// Search mode: encourage finding useful resources
	if (!event.underThreat) {
		if (event.discoveredStone) next += 1.2;
		if (event.pickedStone) next += 1.0;
	}

	// Checkpoint only matters when safe
	if (!event.underThreat && typeof event.progressDelta === "number") {
		next += event.progressDelta * 20.0;
	}

	// Big rewards
	if (event.ateFood) next += 2.5;

	// Waste penalties
	if (event.wastedThrow) next -= 1.5;
	if (event.stuck) next -= 2.0;
	if (event.idle) next -= 0.6;

	// Strong death penalty
	if (event.died) next -= 40.0;

	return next;
}

type DenseNet = {
	inputSize: number;
	h1Size: number;
	h2Size: number;
	outputSize: number;
	w1: Float32Array;
	b1: Float32Array;
	w2: Float32Array;
	b2: Float32Array;
	w3: Float32Array;
	b3: Float32Array;
};

type Genome = {
	id: string;
	net: DenseNet;
	fitness: number;
};

type SerializedDenseNet = {
	inputSize: number;
	h1Size: number;
	h2Size: number;
	outputSize: number;
	w1: number[];
	b1: number[];
	w2: number[];
	b2: number[];
	w3: number[];
	b3: number[];
};

type SerializedGeneration = {
	version: 1;
	sim: "sim6";
	generation: number;
	botCount: number;
	inputSize: number;
	genomes: Array<{ id: string; fitness: number; net: SerializedDenseNet }>;
};

const VISION_RES = 16; // rendered image resolution for color-based perception
// Perceptual features extracted from the rendered image:
// 3 object types × 3 values = 9
// [food_seen, food_hAngle, food_dist, enemy_seen, enemy_hAngle, enemy_dist, rock_seen, rock_hAngle, rock_dist]
const PERCEPT_FEATURE_COUNT = 9;
const STATE_FEATURE_COUNT = 5; // health, energy, organs, speed, underThreat
const ACTION_OUTPUT_COUNT = 11;
const ACTION_OUTPUT_NAMES = ["left", "right", "forward", "backward", "speed", "pick", "throw", "grow", "seesGreenCheckpoint", "seesStone", "seesRedEnemy"];
const ACTION_OUTPUT_SHORT = ["L", "R", "F", "B", "Sp", "Pk", "Th", "Gr", "Cp", "St", "En"];

function meanAbs(arr: Float32Array): number {
	if (arr.length === 0) return 0;
	let sum = 0;
	for (let i = 0; i < arr.length; i += 1) sum += Math.abs(arr[i]);
	return sum / arr.length;
}

function sampleEvenIndices(total: number, maxNodes: number): number[] {
	if (total <= 0) return [];
	if (total <= maxNodes) return Array.from({ length: total }, (_, i) => i);
	const out: number[] = [];
	const denom = Math.max(1, maxNodes - 1);
	for (let i = 0; i < maxNodes; i += 1) {
		out.push(Math.round((i * (total - 1)) / denom));
	}
	return Array.from(new Set(out));
}

function sampleActivations(values: Float32Array, maxNodes: number): { indices: number[]; activations: number[] } {
	const indices = sampleEvenIndices(values.length, maxNodes);
	return {
		indices,
		activations: indices.map((idx) => values[idx] ?? 0),
	};
}

function sampleDenseWeightsW1(net: DenseNet, inputIdx: number[], h1Idx: number[]): number[][] {
	return inputIdx.map((src) => h1Idx.map((dst) => net.w1[dst * net.inputSize + src] ?? 0));
}

function sampleDenseWeightsW2(net: DenseNet, h1Idx: number[], h2Idx: number[]): number[][] {
	return h1Idx.map((src) => h2Idx.map((dst) => net.w2[dst * net.h1Size + src] ?? 0));
}

function sampleDenseWeightsW3(net: DenseNet, h2Idx: number[], outIdx: number[]): number[][] {
	return h2Idx.map((src) => outIdx.map((dst) => net.w3[dst * net.h2Size + src] ?? 0));
}

function randWeight(scale = 0.2): number {
	return (Math.random() * 2 - 1) * scale;
}

function createDenseNet(inputSize: number, h1Size = 32, h2Size = 16, outputSize = ACTION_OUTPUT_COUNT): DenseNet {
	const net: DenseNet = {
		inputSize,
		h1Size,
		h2Size,
		outputSize,
		w1: new Float32Array(inputSize * h1Size),
		b1: new Float32Array(h1Size),
		w2: new Float32Array(h1Size * h2Size),
		b2: new Float32Array(h2Size),
		w3: new Float32Array(h2Size * outputSize),
		b3: new Float32Array(outputSize),
	};
	for (let i = 0; i < net.w1.length; i += 1) net.w1[i] = randWeight();
	for (let i = 0; i < net.b1.length; i += 1) net.b1[i] = randWeight(0.05);
	for (let i = 0; i < net.w2.length; i += 1) net.w2[i] = randWeight();
	for (let i = 0; i < net.b2.length; i += 1) net.b2[i] = randWeight(0.05);
	for (let i = 0; i < net.w3.length; i += 1) net.w3[i] = randWeight();
	for (let i = 0; i < net.b3.length; i += 1) net.b3[i] = randWeight(0.05);
	return net;
}

function cloneDenseNet(src: DenseNet): DenseNet {
	return {
		inputSize: src.inputSize,
		h1Size: src.h1Size,
		h2Size: src.h2Size,
		outputSize: src.outputSize,
		w1: new Float32Array(src.w1),
		b1: new Float32Array(src.b1),
		w2: new Float32Array(src.w2),
		b2: new Float32Array(src.b2),
		w3: new Float32Array(src.w3),
		b3: new Float32Array(src.b3),
	};
}

function serializeDenseNet(net: DenseNet): SerializedDenseNet {
	return {
		inputSize: net.inputSize,
		h1Size: net.h1Size,
		h2Size: net.h2Size,
		outputSize: net.outputSize,
		w1: Array.from(net.w1),
		b1: Array.from(net.b1),
		w2: Array.from(net.w2),
		b2: Array.from(net.b2),
		w3: Array.from(net.w3),
		b3: Array.from(net.b3),
	};
}

function deserializeDenseNet(data: SerializedDenseNet, fallbackInputSize: number): DenseNet {
	const inputSize = Number.isFinite(data?.inputSize) ? data.inputSize : fallbackInputSize;
	const h1Size = Number.isFinite(data?.h1Size) ? data.h1Size : 128;
	const h2Size = Number.isFinite(data?.h2Size) ? data.h2Size : 64;
	const outputSize = Number.isFinite(data?.outputSize) ? data.outputSize : ACTION_OUTPUT_COUNT;
	const net = createDenseNet(inputSize, h1Size, h2Size, outputSize);

	const assignArray = (dst: Float32Array, src?: number[]) => {
		if (!Array.isArray(src)) return;
		const n = Math.min(dst.length, src.length);
		for (let i = 0; i < n; i += 1) {
			const v = Number(src[i]);
			dst[i] = Number.isFinite(v) ? v : dst[i];
		}
	};

	assignArray(net.w1, data?.w1);
	assignArray(net.b1, data?.b1);
	assignArray(net.w2, data?.w2);
	assignArray(net.b2, data?.b2);
	assignArray(net.w3, data?.w3);
	assignArray(net.b3, data?.b3);
	return net;
}

function mutateDenseNet(src: DenseNet, mutationRate = 0.1, mutationStrength = 0.2): DenseNet {
	const net = cloneDenseNet(src);
	const mutateArray = (arr: Float32Array) => {
		for (let i = 0; i < arr.length; i += 1) {
			if (Math.random() < mutationRate) {
				arr[i] = clamp(arr[i] + randWeight(mutationStrength), -3, 3);
			}
		}
	};
	mutateArray(net.w1);
	mutateArray(net.b1);
	mutateArray(net.w2);
	mutateArray(net.b2);
	mutateArray(net.w3);
	mutateArray(net.b3);
	return net;
}

function relu(v: number): number {
	return v > 0 ? v : 0;
}

function forwardDenseNet(net: DenseNet, input: Float32Array): Float32Array {
	const h1 = new Float32Array(net.h1Size);
	for (let j = 0; j < net.h1Size; j += 1) {
		let sum = net.b1[j];
		const row = j * net.inputSize;
		for (let i = 0; i < net.inputSize; i += 1) sum += net.w1[row + i] * input[i];
		h1[j] = relu(sum);
	}

	const h2 = new Float32Array(net.h2Size);
	for (let j = 0; j < net.h2Size; j += 1) {
		let sum = net.b2[j];
		const row = j * net.h1Size;
		for (let i = 0; i < net.h1Size; i += 1) sum += net.w2[row + i] * h1[i];
		h2[j] = relu(sum);
	}

	const out = new Float32Array(net.outputSize);
	for (let j = 0; j < net.outputSize; j += 1) {
		let sum = net.b3[j];
		const row = j * net.h2Size;
		for (let i = 0; i < net.h2Size; i += 1) sum += net.w3[row + i] * h2[i];
		out[j] = Math.tanh(sum);
	}
	return out;
}

function forwardDenseNetTrace(net: DenseNet, input: Float32Array): { h1: Float32Array; h2: Float32Array; out: Float32Array } {
	const h1 = new Float32Array(net.h1Size);
	for (let j = 0; j < net.h1Size; j += 1) {
		let sum = net.b1[j];
		const row = j * net.inputSize;
		for (let i = 0; i < net.inputSize; i += 1) sum += net.w1[row + i] * input[i];
		h1[j] = relu(sum);
	}

	const h2 = new Float32Array(net.h2Size);
	for (let j = 0; j < net.h2Size; j += 1) {
		let sum = net.b2[j];
		const row = j * net.h1Size;
		for (let i = 0; i < net.h1Size; i += 1) sum += net.w2[row + i] * h1[i];
		h2[j] = relu(sum);
	}

	const out = new Float32Array(net.outputSize);
	for (let j = 0; j < net.outputSize; j += 1) {
		let sum = net.b3[j];
		const row = j * net.h2Size;
		for (let i = 0; i < net.h2Size; i += 1) sum += net.w3[row + i] * h2[i];
		out[j] = Math.tanh(sum);
	}

	return { h1, h2, out };
}

// Deterministic color-based perception: scan the rendered image for known object colors.
// Returns 9 values: [food_seen, food_hAngle, food_dist, enemy_seen, enemy_hAngle, enemy_dist, rock_seen, rock_hAngle, rock_dist]
// food   = bright green (G dominant)
// enemy  = red/pink (R dominant)
// rock   = neutral gray (equal RGB, mid-brightness)
// hAngle: -1 = far left, +1 = far right
// dist:    1 = very close (bottom of frame), 0 = far (top of frame)
function extractPerceptualFeatures(rgba: Uint8ClampedArray, res: number): Float32Array {
	const sumX = [0, 0, 0];
	const sumY = [0, 0, 0];
	const cnt  = [0, 0, 0];

	for (let y = 0; y < res; y += 1) {
		for (let x = 0; x < res; x += 1) {
			const idx = (y * res + x) * 4;
			const r = rgba[idx];
			const g = rgba[idx + 1];
			const b = rgba[idx + 2];
			// Food: green dominant, reasonably bright
			if (g > 110 && g > r * 1.35 && g > b * 1.2) {
				sumX[0] += x; sumY[0] += y; cnt[0] += 1;
			// Enemy: red/pink dominant
			} else if (r > 100 && r > g * 1.25 && r > b * 1.2) {
				sumX[1] += x; sumY[1] += y; cnt[1] += 1;
			// Rock: near-gray, not too dark
			} else if (r > 70 && g > 70 && b > 70 &&
					Math.abs(r - g) < 48 && Math.abs(r - b) < 48 && Math.abs(g - b) < 48) {
				sumX[2] += x; sumY[2] += y; cnt[2] += 1;
			}
		}
	}

	const out = new Float32Array(PERCEPT_FEATURE_COUNT);
	for (let c = 0; c < 3; c += 1) {
		const base = c * 3;
		if (cnt[c] === 0) {
			out[base]     = 0; // not seen
			out[base + 1] = 0; // angle (center)
			out[base + 2] = 0; // distance (far)
		} else {
			const cx = sumX[c] / cnt[c];
			const cy = sumY[c] / cnt[c];
			out[base]     = 1;                         // seen
			out[base + 1] = (cx / res - 0.5) * 2;     // horizontal angle -1..1
			out[base + 2] = cy / res;                  // dist: 1=bottom(close), 0=top(far)
		}
	}
	return out;
}

function compactStateToFeatures(state: any): Float32Array {
	const f = new Float32Array(STATE_FEATURE_COUNT); // 5 values
	f[0] = clamp(Number(state?.health ?? 0) / 100, 0, 1);
	f[1] = clamp(Number(state?.energy ?? 0) / 140, 0, 1);
	f[2] = clamp(Number(state?.organs ?? 0) / 4, 0, 1);
	f[3] = clamp(Number(state?.botSpeed ?? 0) / 8, 0, 1);
	f[4] = state?.underThreat ? 1 : 0;
	return f;
}

function visionToFeatures(_rgba: Uint8ClampedArray, _res: number): Float32Array {
	return new Float32Array(PERCEPT_FEATURE_COUNT); // replaced by extractPerceptualFeatures
}

function buildNetInput(percept: Float32Array, stateFeat: Float32Array): Float32Array {
	const input = new Float32Array(percept.length + stateFeat.length);
	input.set(percept, 0);
	input.set(stateFeat, percept.length);
	return input;
}

function outputsToAction(out: Float32Array): AiAction {
	return sanitizeAction({
		left: Math.max(0, out[0]),
		right: Math.max(0, out[1]),
		forward: Math.max(0, out[2]),
		backward: Math.max(0, out[3]),
		speed: clamp((out[4] + 1) * 0.5, 0, 1),
		pick: out[5] > 0.2,
		throw: out[6] > 0.2,
		grow: out[7] > 0.2,
		seesGreenCheckpoint: out[8] > 0,
		seesStone: out[9] > 0,
		seesRedEnemy: out[10] > 0,
	});
}

function createPopulation(popSize: number, inputSize: number): Genome[] {
	const genomes: Genome[] = [];
	for (let i = 0; i < popSize; i += 1) {
		genomes.push({ id: randomId("g"), net: createDenseNet(inputSize), fitness: 0 });
	}
	return genomes;
}

function evolvePopulation(genomes: Genome[], eliteCount = 4): Genome[] {
	const sorted = [...genomes].sort((a, b) => b.fitness - a.fitness);
	const elites = sorted.slice(0, eliteCount);
	const next: Genome[] = elites.map((e) => ({ id: randomId("g"), net: cloneDenseNet(e.net), fitness: 0 }));
	while (next.length < genomes.length) {
		const p = elites[Math.floor(Math.random() * elites.length)] ?? sorted[0];
		next.push({ id: randomId("g"), net: mutateDenseNet(p.net, 0.1, 0.22), fitness: 0 });
	}
	return next;
}



function sanitizeAction(action: Partial<AiAction> | null | undefined): AiAction {
	const a = action ?? {};
	return {
		left: clamp(Number(a.left ?? 0), -1, 1),
		right: clamp(Number(a.right ?? 0), -1, 1),
		forward: clamp(Number(a.forward ?? 0), -1, 1),
		backward: clamp(Number(a.backward ?? 0), -1, 1),
		speed: clamp(Number(a.speed ?? 0.5), 0, 1),
		pick: a.pick === true,
		throw: a.throw === true,
		grow: a.grow === true,
		seesGreenCheckpoint: a.seesGreenCheckpoint === true,
		seesStone: a.seesStone === true,
		seesRedEnemy: a.seesRedEnemy === true,
	};
}

function defaultFallbackAction(): AiAction {
	return { left: 0, right: 0, forward: 1, backward: 0, speed: 0.6, pick: false, throw: false, grow: false };
}

function BabylonWorld({
	settings,
	paused,
	onUi,
	onVisionPreview,
	onAiExchange,
	onNeuralNetUpdate,
	onSelectedBotUpdate,
}: {
	settings: Sim4Settings;
	paused: boolean;
	onUi: (snap: UiSnapshot) => void;
	onVisionPreview: (dataUrl: string) => void;
	onAiExchange: (exchange: AiExchange) => void;
	onNeuralNetUpdate: (snap: NeuralNetSnapshot) => void;
	onSelectedBotUpdate: (snap: SelectedBotSnapshot) => void;
}) {
	const canvasRef = useRef<HTMLCanvasElement | null>(null);
	const settingsRef = useRef(settings);
	const pausedRef = useRef(paused);
	const onVisionPreviewRef = useRef(onVisionPreview);
	const onAiExchangeRef = useRef(onAiExchange);
	const onNeuralNetUpdateRef = useRef(onNeuralNetUpdate);
	const onSelectedBotUpdateRef = useRef(onSelectedBotUpdate);

	useEffect(() => {
		settingsRef.current = settings;
	}, [settings]);
	useEffect(() => {
		pausedRef.current = paused;
	}, [paused]);
	useEffect(() => {
		onVisionPreviewRef.current = onVisionPreview;
	}, [onVisionPreview]);
	useEffect(() => {
		onAiExchangeRef.current = onAiExchange;
	}, [onAiExchange]);
	useEffect(() => {
		onNeuralNetUpdateRef.current = onNeuralNetUpdate;
	}, [onNeuralNetUpdate]);
	useEffect(() => {
		onSelectedBotUpdateRef.current = onSelectedBotUpdate;
	}, [onSelectedBotUpdate]);

	useEffect(() => {
		if (!canvasRef.current) return;

		const engine = new BABYLON.Engine(canvasRef.current, true, { preserveDrawingBuffer: true, stencil: true });
		const scene = new BABYLON.Scene(engine);
		scene.clearColor = BABYLON.Color4.FromHexString("#020617FF");

		const camera = new BABYLON.ArcRotateCamera("camera", -Math.PI / 4, 1.12, 130, BABYLON.Vector3.Zero(), scene);
		camera.attachControl(canvasRef.current, true);
		camera.lowerRadiusLimit = 20;
		camera.upperRadiusLimit = 220;

		const hemi = new BABYLON.HemisphericLight("ambient", new BABYLON.Vector3(0, 1, 0), scene);
		hemi.intensity = 0.9;
		hemi.specular = BABYLON.Color3.Black();
		const sun = new BABYLON.DirectionalLight("sun", new BABYLON.Vector3(-0.7, -1, -0.3), scene);
		sun.position = new BABYLON.Vector3(50, 80, 10);
		sun.intensity = 1.05;
		sun.specular = BABYLON.Color3.Black();

		const ground = BABYLON.MeshBuilder.CreateGround("ground", { width: WORLD_SIZE, height: WORLD_SIZE }, scene);
		const groundMat = new BABYLON.StandardMaterial("groundMat", scene);
		groundMat.diffuseColor = BABYLON.Color3.FromHexString("#0b2a22");
		ground.material = groundMat;
		ground.isPickable = true;

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

		// Visible boundary walls (so the vision model can see the world edge)
		const wallMat = new BABYLON.StandardMaterial("wallMat", scene);
		wallMat.diffuseColor = BABYLON.Color3.FromHexString("#334155");
		wallMat.emissiveColor = BABYLON.Color3.FromHexString("#0b1220");
		const wallH = 3.6;
		const wallT = 0.9;
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

		// Exit marker removed — survival sim has no goal location

		// Procedural city obstacles (buildings as boxes + collision circles)
		const obstacles: Obstacle[] = [];
		const buildingMeshes = new Map<string, BABYLON.Mesh>();
		const buildingMat = new BABYLON.StandardMaterial("buildingMat", scene);
		buildingMat.diffuseColor = BABYLON.Color3.FromHexString("#1f2937");
		buildingMat.emissiveColor = BABYLON.Color3.FromHexString("#0b1220");

		const cityGrid = 9;
		const cell = WORLD_SIZE / cityGrid;
		for (let gx = 0; gx < cityGrid; gx += 1) {
			for (let gz = 0; gz < cityGrid; gz += 1) {
				// Keep a couple of diagonal “roads” to avoid sealing the bot.
				if (gx === gz || gx === gz + 1) continue;
				if (Math.random() < 0.25) continue;
				const w = rand(cell * 0.35, cell * 0.75);
				const d = rand(cell * 0.35, cell * 0.75);
				const h = rand(6, 26);
				const x = -HALF + cell * (gx + 0.5) + rand(-cell * 0.12, cell * 0.12);
				const z = -HALF + cell * (gz + 0.5) + rand(-cell * 0.12, cell * 0.12);
				// Leave some space around spawn and exit.
				if (dist2({ x, z }, { x: -HALF + 6, z: -HALF + 6 }) < 12 * 12) continue;
				if (dist2({ x, z }, { x: EXIT_X, z: EXIT_Z }) < 14 * 14) continue;

				const id = randomId("b");
				const mesh = BABYLON.MeshBuilder.CreateBox(id, { width: w, depth: d, height: h }, scene);
				mesh.position.set(x, h / 2, z);
				mesh.material = buildingMat;
				mesh.isPickable = false;
				buildingMeshes.set(id, mesh);

				obstacles.push({ id, x, z, radius: Math.sqrt((w * 0.5) ** 2 + (d * 0.5) ** 2), height: h });
			}
		}

		// 10-bot population meshes (one per genome)
		const botAgents: BotAgent[] = [];
		const botMat = new BABYLON.StandardMaterial("botMat", scene);
		botMat.diffuseColor = BABYLON.Color3.FromHexString("#e2e8f0");
		botMat.emissiveColor = BABYLON.Color3.FromHexString("#0f172a");
		const championMat = new BABYLON.StandardMaterial("championMat", scene);
		championMat.diffuseColor = BABYLON.Color3.FromHexString("#22c55e");
		championMat.emissiveColor = BABYLON.Color3.FromHexString("#14532d");
		const selectedMat = new BABYLON.StandardMaterial("selectedMat", scene);
		selectedMat.diffuseColor = BABYLON.Color3.FromHexString("#3b82f6");
		selectedMat.emissiveColor = BABYLON.Color3.FromHexString("#1e3a8a");
		const dirMat = new BABYLON.StandardMaterial("dirMat", scene);
		dirMat.diffuseColor = BABYLON.Color3.FromHexString("#f8fafc");
		dirMat.emissiveColor = BABYLON.Color3.FromHexString("#38bdf8");

		for (let i = 0; i < BOT_COUNT; i += 1) {
			const mesh = BABYLON.MeshBuilder.CreateCapsule(`bot-${i}`, { radius: 0.45, height: 2.2, tessellation: 8 }, scene);
			mesh.material = botMat;
			mesh.isPickable = true;
			mesh.metadata = { botIndex: i };
			const dir = BABYLON.MeshBuilder.CreateCylinder(`bot-dir-${i}`, { diameterTop: 0, diameterBottom: 0.2, height: 0.45, tessellation: 8 }, scene);
			dir.parent = mesh;
			dir.position.set(0, 0.75, 0.72);
			dir.rotation.x = Math.PI / 2;
			dir.isPickable = false;
			dir.material = dirMat;
			const spawnX = -HALF + 6 + (i % 5) * 1.6;
			const spawnZ = -HALF + 6 + Math.floor(i / 5) * 1.6;
			const bot: Bot = {
				x: spawnX,
				z: spawnZ,
				vx: 0,
				vz: 0,
				headingX: 0,
				headingZ: 1,
				health: 100,
				maxHealth: 100,
				energy: 100,
				maxEnergy: 120,
				organs: 0,
				escaped: false,
			};
			botAgents.push({
				id: randomId("bot"),
				bot,
				mesh,
				action: defaultFallbackAction(),
				fitness: 0,
				prevExitDist: Number.POSITIVE_INFINITY,
				prevHeadingX: bot.headingX,
				prevHeadingZ: bot.headingZ,
				enemyKills: 0,
				directionChanges: 0,
				deathPenaltyApplied: false,
				heldRockId: null,
				exploredCells: new Set<number>(),
			});
		}

		// Eye camera + render target for vision
		const eyeCam = new BABYLON.FreeCamera("eyeCam", new BABYLON.Vector3(0, 2.0, 0), scene);
		eyeCam.minZ = 0.1;
		eyeCam.fov = 0.9;
		eyeCam.rotation = new BABYLON.Vector3(0, 0, 0);

		const visionTarget = new BABYLON.RenderTargetTexture(
			"visionTarget",
			{ width: settingsRef.current.visionResolution, height: settingsRef.current.visionResolution },
			scene,
			false,
			true,
			BABYLON.Engine.TEXTURETYPE_UNSIGNED_INT,
		);
		visionTarget.activeCamera = eyeCam;
		visionTarget.renderParticles = false;
		visionTarget.renderList = [];
		scene.customRenderTargets.push(visionTarget);

		// Actors state

		const enemies: Enemy[] = [];
		const enemyMeshes = new Map<string, BABYLON.Mesh>();
		const enemyMat = new BABYLON.StandardMaterial("enemyMat", scene);
		enemyMat.diffuseColor = BABYLON.Color3.FromHexString("#fb7185");
		enemyMat.emissiveColor = BABYLON.Color3.FromHexString("#450a0a");

		const rocks: Rock[] = [];
		const rockMeshes = new Map<string, BABYLON.Mesh>();
		const rockMat = new BABYLON.StandardMaterial("rockMat", scene);
		rockMat.diffuseColor = BABYLON.Color3.FromHexString("#94a3b8");

		// ── Food particles ──────────────────────────────────────────
		const foods: Food[] = [];
		const foodMeshes = new Map<string, BABYLON.Mesh>();
		const foodMat = new BABYLON.StandardMaterial("foodMat", scene);
		foodMat.diffuseColor = BABYLON.Color3.FromHexString("#4ade80");
		foodMat.emissiveColor = BABYLON.Color3.FromHexString("#16a34a");

		// Selection ring: flat torus on the ground around the selected bot (5-unit radius)
		const selectionRing = BABYLON.MeshBuilder.CreateTorus("selectionRing", { diameter: 10, thickness: 0.18, tessellation: 48 }, scene);
		selectionRing.rotation.x = Math.PI / 2;
		selectionRing.isPickable = false;
		const selectionRingMat = new BABYLON.StandardMaterial("selectionRingMat", scene);
		selectionRingMat.diffuseColor = BABYLON.Color3.FromHexString("#3b82f6");
		selectionRingMat.emissiveColor = BABYLON.Color3.FromHexString("#60a5fa");
		selectionRingMat.alpha = 0.75;
		selectionRing.material = selectionRingMat;

		for (let i = 0; i < 16; i += 1) {
			const id = randomId("rock");
			rocks.push({
				id,
				x: rand(-HALF + 10, HALF - 10),
				z: rand(-HALF + 10, HALF - 10),
				vx: 0,
				vz: 0,
				heldByBot: false,
				active: true,
			});
		}

		const syncRocks = () => {
			for (const r of rocks) {
				let mesh = rockMeshes.get(r.id);
				if (!mesh) {
					mesh = BABYLON.MeshBuilder.CreateSphere(`rock-${r.id}`, { diameter: ROCK_RADIUS * 2, segments: 10 }, scene);
					mesh.material = rockMat;
					mesh.isPickable = false;
					rockMeshes.set(r.id, mesh);
				}
				mesh.setEnabled(r.active);
				if (!r.active) continue;
				if (r.heldByBot) {
					// Find holder and display rock above their head
					const holder = botAgents.find((a) => a.heldRockId === r.id);
					if (holder) {
						mesh.position.set(holder.bot.x, 2.4, holder.bot.z);
					}
				} else {
					mesh.position.set(r.x, ROCK_RADIUS, r.z);
				}
			}
		};

		const syncEnemies = () => {
			for (const e of enemies) {
				let mesh = enemyMeshes.get(e.id);
				if (!mesh) {
					mesh = BABYLON.MeshBuilder.CreateSphere(`enemy-${e.id}`, { diameter: ENEMY_RADIUS * 2, segments: 12 }, scene);
					mesh.material = enemyMat;
					mesh.isPickable = false;
					enemyMeshes.set(e.id, mesh);
				}
				mesh.setEnabled(e.alive);
				if (e.alive) mesh.position.set(e.x, ENEMY_RADIUS, e.z);
			}
		};

		const syncFoods = () => {
			for (const f of foods) {
				let mesh = foodMeshes.get(f.id);
				if (!mesh) {
					mesh = BABYLON.MeshBuilder.CreateSphere(`food-${f.id}`, { diameter: FOOD_RADIUS * 2, segments: 6 }, scene);
					mesh.material = foodMat;
					mesh.isPickable = false;
					foodMeshes.set(f.id, mesh);
				}
				mesh.setEnabled(f.active);
				if (f.active) mesh.position.set(f.x, FOOD_RADIUS, f.z);
			}
		};

		const spawnFood = () => {
			const id = randomId("food");
			foods.push({ id, x: rand(-HALF + 5, HALF - 5), z: rand(-HALF + 5, HALF - 5), active: true });
		};

		const spawnEnemyAt = (x: number, z: number) => {
			enemies.push({ id: randomId("enemy"), x, z, vx: 0, vz: 0, alive: true });
		};

		const spawnBaselineEnemies = () => {
			for (let i = 0; i < 6; i += 1) {
				const ex = rand(-HALF + 14, HALF - 14);
				const ez = rand(-HALF + 14, HALF - 14);
				if (dist2({ x: ex, z: ez }, { x: -HALF + 8, z: -HALF + 8 }) < 16 * 16) continue;
				if (dist2({ x: ex, z: ez }, { x: EXIT_X, z: EXIT_Z }) < 12 * 12) continue;
				spawnEnemyAt(ex, ez);
			}
		};

		const spawnStoneAt = (x: number, z: number) => {
			const id = randomId("rock");
			rocks.push({
				id,
				x,
				z,
				vx: 0,
				vz: 0,
				heldByBot: false,
				active: true,
			});
		};

		let selectedBotIdx = 0;

		const emitSelectedBotSnapshot = () => {
			const a = botAgents[selectedBotIdx] ?? botAgents[0];
			if (!a) return;
			const yawDeg = Math.atan2(a.bot.headingX, a.bot.headingZ) * (180 / Math.PI);
			onSelectedBotUpdateRef.current({
				index: selectedBotIdx,
				id: a.id,
				health: a.bot.health,
				energy: a.bot.energy,
				organs: a.bot.organs,
				escaped: a.bot.escaped,
				dead: a.bot.health <= 0,
				fitness: a.fitness,
				enemyKills: a.enemyKills,
				directionChanges: a.directionChanges,
				action: a.action,
				heading: { x: a.bot.headingX, z: a.bot.headingZ, yawDeg },
				position: { x: a.bot.x, z: a.bot.z },
			});
		};

		emitSelectedBotSnapshot();

		let lastPointerGround = { x: 0, z: 0 };

		// Click bot mesh to select it. Ground click only updates pointer anchor.
		scene.onPointerObservable.add((eventInfo) => {
			if (eventInfo.type !== BABYLON.PointerEventTypes.POINTERMOVE && eventInfo.type !== BABYLON.PointerEventTypes.POINTERPICK) return;
			const pick = eventInfo.pickInfo;
			if (!pick?.hit || !pick.pickedPoint) return;
			if (pick.pickedMesh?.name === "ground") {
				lastPointerGround = {
					x: clamp(pick.pickedPoint.x, -HALF + 2, HALF - 2),
					z: clamp(pick.pickedPoint.z, -HALF + 2, HALF - 2),
				};
			}
			if (eventInfo.type !== BABYLON.PointerEventTypes.POINTERPICK) return;
			const pickedMesh = pick.pickedMesh as any;
			const meshBotIndex = Number(pickedMesh?.metadata?.botIndex);
			if (Number.isInteger(meshBotIndex) && meshBotIndex >= 0 && meshBotIndex < botAgents.length) {
				selectedBotIdx = meshBotIndex;
				emitSelectedBotSnapshot();
				return;
			}
		});

		const keySpawnHandler = (ev: KeyboardEvent) => {
			const key = ev.key.toLowerCase();
			if (key === "e") {
				spawnEnemyAt(lastPointerGround.x, lastPointerGround.z);
			}
			if (key === "s") {
				spawnStoneAt(lastPointerGround.x, lastPointerGround.z);
			}
		};
		window.addEventListener("keydown", keySpawnHandler);

		// Vision rendering should include relevant meshes
		visionTarget.renderList = [ground, wallN, wallS, wallW, wallE, ...Array.from(buildingMeshes.values()), ...botAgents.map((a) => a.mesh)];
		// Enemies/rocks/food are added dynamically via sync functions; we will refresh renderList periodically.
		let lastRenderListRefresh = 0;

		let lastAiAt = 0;
		let pendingAi = false;
		let foodSpawnAccum = 0;

		const captureVision = async (viewer: Bot, res: number) => {
			// Ensure target size matches settings.
			if (visionTarget.getSize().width !== res) {
				visionTarget.resize({ width: res, height: res });
			}
			const eyeY = 1.3;
			eyeCam.position.set(viewer.x, eyeY, viewer.z);
			eyeCam.setTarget(new BABYLON.Vector3(viewer.x + viewer.headingX, eyeY, viewer.z + viewer.headingZ));
			visionTarget.render(true);
			const pixels = (await visionTarget.readPixels()) as Uint8Array | null;
			if (!pixels) return { dataUrl: "", features: new Float32Array(PERCEPT_FEATURE_COUNT) };

			// pixels are Uint8Array RGBA. Create a vertically flipped copy for display + matrix.
			const flipped = new Uint8ClampedArray(res * res * 4);
			for (let y = 0; y < res; y += 1) {
				const srcRow = y * res * 4;
				const dstRow = (res - 1 - y) * res * 4;
				flipped.set(pixels.subarray(srcRow, srcRow + res * 4), dstRow);
			}

			// Create a preview data URL.
			const offscreen = document.createElement("canvas");
			offscreen.width = res;
			offscreen.height = res;
			const ctx = offscreen.getContext("2d", { willReadFrequently: true });
			if (!ctx) return { dataUrl: "", features: extractPerceptualFeatures(flipped, res) };
			const img = new ImageData(flipped, res, res);
			ctx.putImageData(img, 0, 0);
			const dataUrl = offscreen.toDataURL("image/png");
			const features = extractPerceptualFeatures(flipped, res);
			return { dataUrl, features };
		};

		const computeCompactState = (viewer: Bot) => {
			const basis = computeBotHeadingBasis(viewer);

			let nearestEnemyDist = Infinity;
			let nearestEnemyForward = 0;
			let nearestEnemyRight = 0;
			for (const e of enemies) {
				if (!e.alive) continue;
				const dx = e.x - viewer.x;
				const dz = e.z - viewer.z;
				const d = Math.sqrt(dx * dx + dz * dz);
				if (d < nearestEnemyDist) {
					nearestEnemyDist = d;
					const dir = normalize(dx, dz);
					nearestEnemyForward = dir.x * basis.h.x + dir.z * basis.h.z;
					nearestEnemyRight = dir.x * basis.r.x + dir.z * basis.r.z;
				}
			}

			let nearestRockDist = Infinity;
			let nearestRockForward = 0;
			let nearestRockRight = 0;
			for (const r of rocks) {
				if (!r.active || r.heldByBot) continue;
				const dx = r.x - viewer.x;
				const dz = r.z - viewer.z;
				const d = Math.sqrt(dx * dx + dz * dz);
				if (d < nearestRockDist) {
					nearestRockDist = d;
					const dir = normalize(dx, dz);
					nearestRockForward = dir.x * basis.h.x + dir.z * basis.h.z;
					nearestRockRight = dir.x * basis.r.x + dir.z * basis.r.z;
				}
			}

			const heldRock = false;
			const botSpeed = Math.sqrt(viewer.vx * viewer.vx + viewer.vz * viewer.vz);
			const underThreat = Number.isFinite(nearestEnemyDist) && nearestEnemyDist < 4.2;

			return {
				health: round3(viewer.health),
				energy: round3(viewer.energy),
				organs: viewer.organs,
				botSpeed: round3(botSpeed),
				nearestEnemyDist: Number.isFinite(nearestEnemyDist) ? round3(nearestEnemyDist) : null,
				nearestEnemyForward: Number.isFinite(nearestEnemyDist) ? round3(nearestEnemyForward) : null,
				nearestEnemyRight: Number.isFinite(nearestEnemyDist) ? round3(nearestEnemyRight) : null,
				nearestRockDist: Number.isFinite(nearestRockDist) ? round3(nearestRockDist) : null,
				nearestRockForward: Number.isFinite(nearestRockDist) ? round3(nearestRockForward) : null,
				nearestRockRight: Number.isFinite(nearestRockDist) ? round3(nearestRockRight) : null,
				heldRock,
				underThreat,
			};
		};

		const computeBotHeadingBasis = (viewer: Bot) => {
			const h = normalize(viewer.headingX, viewer.headingZ);
			const r = { x: h.z, z: -h.x };
			return { h, r };
		};

		const resolveObstacleCollision = (x: number, z: number, radius: number) => {
			let nx = x;
			let nz = z;
			for (const o of obstacles) {
				const dx = nx - o.x;
				const dz = nz - o.z;
				const d = Math.sqrt(dx * dx + dz * dz) || 0.0001;
				const minD = o.radius + radius;
				if (d < minD) {
					nx = o.x + (dx / d) * minD;
					nz = o.z + (dz / d) * minD;
				}
			}
			return { x: nx, z: nz };
		};

		const resetWorld = () => {
			for (let i = 0; i < botAgents.length; i += 1) {
				const a = botAgents[i];
				const b = a.bot;
				b.x = -HALF + 6 + (i % 5) * 1.6;
				b.z = -HALF + 6 + Math.floor(i / 5) * 1.6;
				b.vx = 0;
				b.vz = 0;
				b.headingX = 0;
				b.headingZ = 1;
				b.health = 100;
				b.maxHealth = 100;
				b.energy = 100;
				b.maxEnergy = 120;
				b.organs = 0;
				b.escaped = false;
				a.action = defaultFallbackAction();
				a.fitness = 0;
				a.prevExitDist = Number.POSITIVE_INFINITY;
				a.prevHeadingX = b.headingX;
				a.prevHeadingZ = b.headingZ;
				a.enemyKills = 0;
				a.directionChanges = 0;
				a.deathPenaltyApplied = false;
				// Release any held rock
				if (a.heldRockId) {
					const hr = rocks.find((r) => r.id === a.heldRockId);
					if (hr) hr.heldByBot = false;
					a.heldRockId = null;
				}
				a.exploredCells.clear();
			}
			enemies.length = 0;
			for (const e of enemyMeshes.values()) e.dispose();
			enemyMeshes.clear();
			spawnBaselineEnemies();
			// Clear food so each generation starts clean
			foods.length = 0;
			for (const m of foodMeshes.values()) m.dispose();
			foodMeshes.clear();
			for (const r of rocks) {
				r.x = rand(-HALF + 10, HALF - 10);
				r.z = rand(-HALF + 10, HALF - 10);
				r.vx = 0;
				r.vz = 0;
				r.heldByBot = false;
				r.active = true;
			}
			for (const g of population) g.fitness = 0;
		};

		spawnBaselineEnemies();

		// Expose reset via window for now (controls panel in React triggers via custom event)
		const resetHandler = () => resetWorld();
		window.addEventListener("sim4-reset", resetHandler as any);

		let lastUiAt = 0;
		let lastT = performance.now();
		let lastAiSentImage = false;
		const inputSize = PERCEPT_FEATURE_COUNT + STATE_FEATURE_COUNT; // 9 percept + 5 state = 14
		let population = createPopulation(BOT_COUNT, inputSize);
		let generation = 1;
		let evalStartAt = performance.now();
		let generationBestIdx = 0;
		let topGreenIdxSet = new Set<number>([0]);

		const buildGenerationPayload = (): SerializedGeneration => ({
			version: 1,
			sim: "sim6",
			generation,
			botCount: population.length,
			inputSize,
			genomes: population.map((g) => ({
				id: g.id,
				fitness: g.fitness,
				net: serializeDenseNet(g.net),
			})),
		});

		const normalizeLoadedPopulation = (loaded: Genome[]): Genome[] => {
			const next: Genome[] = [];
			for (let i = 0; i < BOT_COUNT; i += 1) {
				const src = loaded[i] ?? loaded[0];
				if (src) {
					next.push({ id: randomId("g"), net: cloneDenseNet(src.net), fitness: 0 });
				} else {
					next.push({ id: randomId("g"), net: createDenseNet(inputSize), fitness: 0 });
				}
			}
			return next;
		};

		const exportHandler = (event: Event) => {
			const onData = (event as CustomEvent<{ onData?: (payload: SerializedGeneration) => void }>).detail?.onData;
			if (typeof onData === "function") onData(buildGenerationPayload());
		};

		const loadHandler = (event: Event) => {
			const payload = (event as CustomEvent<{ payload?: SerializedGeneration }>).detail?.payload;
			if (!payload || payload.sim !== "sim6" || !Array.isArray(payload.genomes)) return;
			const loaded = payload.genomes.map((g) => ({
				id: randomId("g"),
				net: deserializeDenseNet(g.net, inputSize),
				fitness: 0,
			}));
			population = normalizeLoadedPopulation(loaded);
			generation = Math.max(1, Number(payload.generation) || 1);
			generationBestIdx = 0;
			selectedBotIdx = 0;
			evalStartAt = performance.now();
			resetWorld();
			emitSelectedBotSnapshot();
		};

		window.addEventListener("sim6-export-generation", exportHandler as EventListener);
		window.addEventListener("sim6-load-generation", loadHandler as EventListener);

		// Local AI bootstrap (load/download model once, then run inference calls without re-checking every tick).
		const localBootstrap = {
			ready: false,
			bootstrapping: false,
			lastAttemptAt: 0,
			status: "",
			baseUrl: "",
			modelKey: "",
		};

		const maybeBootstrapLocal = (nowMs: number) => {
			const s = settingsRef.current;
			if (s.aiProvider !== "local") return;
			const baseUrl = normalizeLmStudioBaseUrl(s.localEndpointUrl);
			const modelKey = (s.localModelName || "").trim();
			if (!baseUrl || !modelKey) return;

			// If settings changed, re-bootstrap once for the new key.
			const changed = baseUrl !== localBootstrap.baseUrl || modelKey !== localBootstrap.modelKey;
			if (changed) {
				localBootstrap.ready = false;
				localBootstrap.bootstrapping = false;
				localBootstrap.lastAttemptAt = 0;
				localBootstrap.status = "";
				localBootstrap.baseUrl = baseUrl;
				localBootstrap.modelKey = modelKey;
			}

			if (localBootstrap.ready || localBootstrap.bootstrapping) return;
			if (nowMs - localBootstrap.lastAttemptAt < 900) return;
			localBootstrap.lastAttemptAt = nowMs;
			localBootstrap.bootstrapping = true;
			localBootstrap.status = "Loading model…";

			(void (async () => {
				try {
					const result = await ensureLmStudioModelReady(baseUrl, modelKey);
					localBootstrap.ready = result.state === "ready";
					localBootstrap.status = result.message;
				} catch (err: any) {
					const msg = String(err?.message ?? err ?? "");
					localBootstrap.status = msg || "Model not ready";
					// If downloading, keep retrying at a low rate.
					if (/downloading/i.test(msg)) {
						localBootstrap.ready = false;
					} else {
						// Hard failure: do not spam retries.
						localBootstrap.ready = false;
						localBootstrap.lastAttemptAt = Number.POSITIVE_INFINITY;
					}
				} finally {
					localBootstrap.bootstrapping = false;
				}
			})());
		};

		// Kick off bootstrap immediately on init.
		maybeBootstrapLocal(performance.now());

		engine.runRenderLoop(() => {
			const now = performance.now();
			const dt = Math.min((now - lastT) / 1000, 0.04);
			lastT = now;

			// Ensure local model is loaded (init-only, then only retries while downloading).
			maybeBootstrapLocal(now);

			// Refresh vision render list occasionally.
			if (now - lastRenderListRefresh > 800) {
				lastRenderListRefresh = now;
				visionTarget.renderList = [ground, wallN, wallS, wallW, wallE, ...Array.from(buildingMeshes.values()), ...botAgents.map((a) => a.mesh), ...Array.from(enemyMeshes.values()), ...Array.from(rockMeshes.values()), ...Array.from(foodMeshes.values())];
			}

			const s = settingsRef.current;
			if (!pausedRef.current) {
				// Spawn food continuously at FOOD_SPAWN_PER_SEC per second
				foodSpawnAccum += dt * FOOD_SPAWN_PER_SEC;
				const foodToSpawn = Math.floor(foodSpawnAccum);
				foodSpawnAccum -= foodToSpawn;
				for (let fi = 0; fi < foodToSpawn; fi += 1) spawnFood();

				// AI tick
				const aiPeriodMs = 1000 / Math.max(0.2, s.aiHz);
				if (!pendingAi && now - lastAiAt >= aiPeriodMs) {
					pendingAi = true;
					(void (async () => {
						let asked = "";
						try {
							const tickRes = Math.round(clamp(s.visionResolution, 16, 96));
							const traces = new Map<number, { netInput: Float32Array; trace: { h1: Float32Array; h2: Float32Array; out: Float32Array } }>();

							for (let i = 0; i < botAgents.length; i += 1) {
								const agent = botAgents[i];
								const viewer = agent.bot;
								if (viewer.health <= 0 || viewer.escaped) continue;

								const cap = await captureVision(viewer, tickRes);
								if (i === selectedBotIdx && cap?.dataUrl) onVisionPreviewRef.current(cap.dataUrl);
								const compactState = computeCompactState(viewer);
								if (i === selectedBotIdx) asked = JSON.stringify({ state: compactState });

								const genome = population[i];
								const perceptFeatures = cap?.features ?? new Float32Array(PERCEPT_FEATURE_COUNT);
								const stateFeatures = compactStateToFeatures(compactState);
								const netInput = buildNetInput(perceptFeatures, stateFeatures);
								const trace = forwardDenseNetTrace(genome.net, netInput);
								traces.set(i, { netInput, trace });

const action = outputsToAction(trace.out);
								agent.action = action;

								// --- Fitness: reward survival time + enemy kills ---
								// +1 per second alive (proportional to AI tick interval)
								const aiTickIntervalSec = (performance.now() - lastAiAt) / 1000;
								agent.fitness = computeFitness(agent.fitness, { survivalSec: aiTickIntervalSec });

								// Bonus: actively dodge when threatened
								if (compactState.underThreat) {
									const moveForward = action.forward - action.backward;
									const enemyForward = Number(compactState.nearestEnemyForward ?? 0);
									if (moveForward * enemyForward < 0) {
										agent.fitness = computeFitness(agent.fitness, { underThreat: true, dodgedThreat: true });
									}
								}

								const movementIntent = Math.abs(action.forward - action.backward) + Math.abs(action.right - action.left);
								if (movementIntent < 0.08 && !compactState.underThreat) {
									agent.fitness = computeFitness(agent.fitness, { idle: true, underThreat: false });
								}

								agent.prevHeadingX = viewer.headingX;
								agent.prevHeadingZ = viewer.headingZ;
								genome.fitness = agent.fitness;
							}

							const ranked = botAgents
								.map((a, i) => ({ i, f: a.fitness }))
								.sort((a, b) => b.f - a.f);
							generationBestIdx = ranked[0]?.i ?? 0;
							topGreenIdxSet = new Set(ranked.slice(0, TOP_GREEN_COUNT).map((r) => r.i));

							const focusIdx = traces.has(selectedBotIdx) ? selectedBotIdx : generationBestIdx;
							const bestGenome = population[focusIdx];
							const bestTraceData = traces.get(focusIdx);
							if (bestGenome && bestTraceData) {
								const evalSeconds = (performance.now() - evalStartAt) / 1000;
								const sampledInput = sampleActivations(bestTraceData.netInput, 18);
								const sampledH1 = sampleActivations(bestTraceData.trace.h1, 14);
								const sampledH2 = sampleActivations(bestTraceData.trace.h2, 12);
								const outIdx = Array.from({ length: bestGenome.net.outputSize }, (_, i) => i);
								onNeuralNetUpdateRef.current({
									inputSize: bestGenome.net.inputSize,
									h1Size: bestGenome.net.h1Size,
									h2Size: bestGenome.net.h2Size,
									outputSize: bestGenome.net.outputSize,
									generation,
									genomeIndex: focusIdx,
									populationSize: population.length,
									fitness: round3(bestGenome.fitness),
									evalSeconds: round3(evalSeconds),
									weightAbsMean: {
										w1: round3(meanAbs(bestGenome.net.w1)),
										w2: round3(meanAbs(bestGenome.net.w2)),
										w3: round3(meanAbs(bestGenome.net.w3)),
									},
									inputs: sampledInput,
									hidden1: sampledH1,
									hidden2: sampledH2,
									outputs: ACTION_OUTPUT_NAMES.map((name, i) => ({ index: i, name, value: round3(bestTraceData.trace.out[i] ?? 0) })),
									w1: sampleDenseWeightsW1(bestGenome.net, sampledInput.indices, sampledH1.indices),
									w2: sampleDenseWeightsW2(bestGenome.net, sampledH1.indices, sampledH2.indices),
									w3: sampleDenseWeightsW3(bestGenome.net, sampledH2.indices, outIdx),
								});
							}
							emitSelectedBotSnapshot();

							const evalSeconds = (performance.now() - evalStartAt) / 1000;
							const allTerminal = botAgents.every((a) => a.bot.health <= 0);
							const shouldAdvance = allTerminal || evalSeconds > Math.max(5, s.generationSeconds);
							if (shouldAdvance) {
								for (let i = 0; i < botAgents.length; i += 1) population[i].fitness = botAgents[i].fitness;
								population = evolvePopulation(population, ELITE_COUNT);
								generation += 1;
								generationBestIdx = 0;
								topGreenIdxSet = new Set<number>([0]);
								selectedBotIdx = 0;
								evalStartAt = performance.now();
								resetWorld();
								emitSelectedBotSnapshot();
							}

							lastAiSentImage = true;
							const top3 = [...botAgents]
								.map((a, i) => ({ idx: i, fitness: round3(a.fitness), kills: a.enemyKills }))
								.sort((a, b) => b.fitness - a.fitness)
								.slice(0, 3);
							onAiExchangeRef.current({
								id: randomId("ai"),
								provider: "genetic",
								model: "large-nn-ga",
								asked,
								replied: JSON.stringify({
									generation,
									bots: BOT_COUNT,
									eliteCount: ELITE_COUNT,
									top3,
								}),
								at: Date.now(),
							});
						} catch (err: any) {
							for (const a of botAgents) a.action = defaultFallbackAction();
							onAiExchangeRef.current({
								id: randomId("ai"),
								provider: s.aiProvider,
								model: s.aiModel,
								asked,
								replied: JSON.stringify({ fallback: true }),
								at: Date.now(),
								error: String((err as any)?.message ?? err ?? "Unknown error"),
							});
						} finally {
							lastAiAt = performance.now();
							pendingAi = false;
						}
					})());
				}

				// Vision preview tick (independent of AI)
				const previewPeriodMs = 250;
				// eslint-disable-next-line @typescript-eslint/no-explicit-any
				const previewState = (engine as any).__sim4PreviewState || { lastAt: 0, pending: false };
				// eslint-disable-next-line @typescript-eslint/no-explicit-any
				(engine as any).__sim4PreviewState = previewState;
				if (!previewState.pending && now - previewState.lastAt >= previewPeriodMs) {
					previewState.lastAt = now;
					previewState.pending = true;
					(void (async () => {
						try {
							const previewBot = botAgents[selectedBotIdx]?.bot ?? botAgents[0]?.bot;
							if (!previewBot) return;
							const cap = await captureVision(previewBot, Math.round(clamp(settingsRef.current.visionResolution, 16, 96)));
							if (cap?.dataUrl) onVisionPreviewRef.current(cap.dataUrl);
						} finally {
							previewState.pending = false;
						}
					})());
				}

				if (!pendingAi) {
					for (let i = 0; i < botAgents.length; i += 1) {
						const agent = botAgents[i];
						const bot = agent.bot;
						if (bot.health <= 0 || bot.escaped) continue;
						const action = agent.action;

						const basis = computeBotHeadingBasis(bot);
						const ax = (action.right - action.left) * s.botAccel;
						const az = (action.forward - action.backward) * s.botAccel;
						let steerX = basis.r.x * ax + basis.h.x * az;
						let steerZ = basis.r.z * ax + basis.h.z * az;
						const steerMag = Math.sqrt(steerX * steerX + steerZ * steerZ);
						if (steerMag > 1e-6) {
							steerX /= steerMag;
							steerZ /= steerMag;
						}
						const desiredMaxSpeed = s.botMaxSpeed * clamp(action.speed, 0, 1);

						let vx = bot.vx;
						let vz = bot.vz;
						let x = bot.x;
						let z = bot.z;

						const estSpeed = Math.sqrt(vx * vx + vz * vz);
						const maxStepDist = Math.max(BOT_RADIUS * 0.75, 0.14);
						const steps = clamp(Math.ceil((estSpeed * dt) / maxStepDist), 1, 18);
						const dtStep = dt / steps;
						const dragFactor = Math.pow(0.90, 1 / steps);

						for (let si = 0; si < steps; si += 1) {
							vx = (vx + steerX * s.botAccel * dtStep) * dragFactor;
							vz = (vz + steerZ * s.botAccel * dtStep) * dragFactor;
							const sp = Math.sqrt(vx * vx + vz * vz) || 1;
							if (sp > Math.max(desiredMaxSpeed, 0.0001)) {
								vx = (vx / sp) * desiredMaxSpeed;
								vz = (vz / sp) * desiredMaxSpeed;
							}
							x = clamp(x + vx * dtStep, -HALF + BOT_RADIUS, HALF - BOT_RADIUS);
							z = clamp(z + vz * dtStep, -HALF + BOT_RADIUS, HALF - BOT_RADIUS);
							const resolved = resolveObstacleCollision(x, z, BOT_RADIUS);
							x = resolved.x;
							z = resolved.z;
						}

						bot.vx = vx;
						bot.vz = vz;
						bot.x = x;
						bot.z = z;

						const vmag = Math.sqrt(vx * vx + vz * vz);
						if (vmag > 0.02) {
							bot.headingX = vx / vmag;
							bot.headingZ = vz / vmag;
						}

						// Exploration: reward entering a new map cell (8-unit grid)
						const ecx = Math.floor((bot.x + HALF) / 8);
						const ecz = Math.floor((bot.z + HALF) / 8);
						const cellKey = ecx * 20 + ecz;
						if (!agent.exploredCells.has(cellKey)) {
							agent.exploredCells.add(cellKey);
							agent.fitness = computeFitness(agent.fitness, { explorationDelta: 1 });
							population[i].fitness = agent.fitness;
						}

						bot.energy = clamp(bot.energy - s.energyDrainPerSec * (0.15 + vmag * 0.45) * dt, 0, bot.maxEnergy);

						// Eat food: +FOOD_ENERGY_GAIN energy per food particle touched
						const eatRadiusSq = (BOT_RADIUS + FOOD_RADIUS) ** 2;
						for (const f of foods) {
							if (!f.active) continue;
							if (dist2({ x: bot.x, z: bot.z }, { x: f.x, z: f.z }) < eatRadiusSq) {
								f.active = false;
								bot.energy = clamp(bot.energy + FOOD_ENERGY_GAIN, 0, bot.maxEnergy);
								agent.fitness = computeFitness(agent.fitness, { ateFood: true });
								population[i].fitness = agent.fitness;
							}
						}
						if (bot.energy <= 0) bot.health = clamp(bot.health - 10 * dt, 0, bot.maxHealth);
						if (bot.health <= 0 && !agent.deathPenaltyApplied) {
							agent.deathPenaltyApplied = true;
							agent.fitness = computeFitness(agent.fitness, { died: true });
							population[i].fitness = agent.fitness;
						}

						// --- Pick action: pick up nearest free rock within 1.5 units ---
						const PICK_RANGE = 1.5;
						const THROW_AUTO_RANGE = 5;
						if (action.pick && !agent.heldRockId) {
							let bestRockDist = PICK_RANGE * PICK_RANGE;
							let bestRock: typeof rocks[0] | null = null;
							for (const r of rocks) {
								if (!r.active || r.heldByBot) continue;
								const rd = dist2({ x: bot.x, z: bot.z }, { x: r.x, z: r.z });
								if (rd < bestRockDist) { bestRockDist = rd; bestRock = r; }
							}
							if (bestRock) {
								bestRock.heldByBot = true;
								agent.heldRockId = bestRock.id;
							}
						}

						// --- Auto-throw: if holding a rock and an enemy is within 5 units, throw at it ---
						if (agent.heldRockId) {
							let nearestEnemy: typeof enemies[0] | null = null;
							let nearestEnemyDist2 = THROW_AUTO_RANGE * THROW_AUTO_RANGE;
							for (const e of enemies) {
								if (!e.alive) continue;
								const ed = dist2({ x: bot.x, z: bot.z }, { x: e.x, z: e.z });
								if (ed < nearestEnemyDist2) { nearestEnemyDist2 = ed; nearestEnemy = e; }
							}
							if (nearestEnemy) {
								// Throw: launch rock toward enemy
								const toEnemy = normalize(nearestEnemy.x - bot.x, nearestEnemy.z - bot.z);
								const hr = rocks.find((r) => r.id === agent.heldRockId);
								if (hr) {
									const THROW_SPEED = 14;
									hr.x = bot.x + bot.headingX * (BOT_RADIUS + ROCK_RADIUS + 0.1);
									hr.z = bot.z + bot.headingZ * (BOT_RADIUS + ROCK_RADIUS + 0.1);
									hr.vx = toEnemy.x * THROW_SPEED;
									hr.vz = toEnemy.z * THROW_SPEED;
									hr.heldByBot = false;
								}
								agent.heldRockId = null;
							} else if (action.throw) {
								// Manual throw in heading direction
								const hr = rocks.find((r) => r.id === agent.heldRockId);
								if (hr) {
									const THROW_SPEED = 14;
									hr.x = bot.x + bot.headingX * (BOT_RADIUS + ROCK_RADIUS + 0.1);
									hr.z = bot.z + bot.headingZ * (BOT_RADIUS + ROCK_RADIUS + 0.1);
									hr.vx = bot.headingX * THROW_SPEED;
									hr.vz = bot.headingZ * THROW_SPEED;
									hr.heldByBot = false;
								}
								agent.heldRockId = null;
							}
						}
					}

					for (const e of enemies) {
						if (!e.alive) continue;
						let target: Bot | null = null;
						let bestD = Number.POSITIVE_INFINITY;
						for (const a of botAgents) {
							if (a.bot.health <= 0 || a.bot.escaped) continue;
							const d = dist2({ x: e.x, z: e.z }, { x: a.bot.x, z: a.bot.z });
							if (d < bestD) {
								bestD = d;
								target = a.bot;
							}
						}
						if (!target) continue;

						const toBot = normalize(target.x - e.x, target.z - e.z);
						let ex = e.x;
						let ez = e.z;
						let evx = (e.vx + toBot.x * s.enemySpeed * 2.2 * dt) * 0.86;
						let evz = (e.vz + toBot.z * s.enemySpeed * 2.2 * dt) * 0.86;
						const esp = Math.sqrt(evx * evx + evz * evz) || 1;
						if (esp > s.enemySpeed) {
							evx = (evx / esp) * s.enemySpeed;
							evz = (evz / esp) * s.enemySpeed;
						}

						const maxEDist = Math.max(ENEMY_RADIUS * 0.75, 0.14);
						const stepsE = clamp(Math.ceil((esp * dt) / maxEDist), 1, 12);
						const dtE = dt / stepsE;
						for (let si = 0; si < stepsE; si += 1) {
							ex = clamp(ex + evx * dtE, -HALF + ENEMY_RADIUS, HALF - ENEMY_RADIUS);
							ez = clamp(ez + evz * dtE, -HALF + ENEMY_RADIUS, HALF - ENEMY_RADIUS);
							const resolved = resolveObstacleCollision(ex, ez, ENEMY_RADIUS);
							ex = resolved.x;
							ez = resolved.z;
						}

						e.x = ex;
						e.z = ez;
						e.vx = evx;
						e.vz = evz;

						for (let i = 0; i < botAgents.length; i += 1) {
							const a = botAgents[i];
							if (a.bot.health <= 0 || a.bot.escaped) continue;
							const hit = dist2({ x: a.bot.x, z: a.bot.z }, { x: e.x, z: e.z }) < (BOT_RADIUS + ENEMY_RADIUS) ** 2;
							if (!hit) continue;
							const botSpeed = Math.sqrt(a.bot.vx * a.bot.vx + a.bot.vz * a.bot.vz);
							if (botSpeed > 2.25) {
								e.alive = false;
								a.enemyKills += 1;
								a.fitness = computeFitness(a.fitness, { underThreat: true, enemyKill: true, enemyHit: true });
								population[i].fitness = a.fitness;
								break;
							}
							const damage = s.enemyDamagePerSec * dt;
							a.bot.health = clamp(a.bot.health - damage, 0, a.bot.maxHealth);
							a.fitness = computeFitness(a.fitness, { underThreat: true, damageTaken: damage });
							population[i].fitness = a.fitness;
						}
					}

					for (const r of rocks) {
						if (!r.active || r.heldByBot) continue;
						r.x = clamp(r.x + r.vx * dt, -HALF + ROCK_RADIUS, HALF - ROCK_RADIUS);
						r.z = clamp(r.z + r.vz * dt, -HALF + ROCK_RADIUS, HALF - ROCK_RADIUS);
						r.vx *= 0.93;
						r.vz *= 0.93;
						if (Math.abs(r.vx) + Math.abs(r.vz) < 0.05) {
							r.vx = 0;
							r.vz = 0;
						}
						// Check if fast-moving rock hits an enemy
						const rockSpeed2 = r.vx * r.vx + r.vz * r.vz;
						if (rockSpeed2 > 4) {
							for (let ei = 0; ei < enemies.length; ei += 1) {
								const e = enemies[ei];
								if (!e.alive) continue;
								if (dist2({ x: r.x, z: r.z }, { x: e.x, z: e.z }) < (ROCK_RADIUS + ENEMY_RADIUS) ** 2) {
									e.alive = false;
									r.vx = 0; r.vz = 0;
									// Credit the bot that threw this rock
									for (let bi = 0; bi < botAgents.length; bi += 1) {
										// We can't track exactly who threw it after release,
										// so credit the nearest living bot as the thrower
									}
									// Give kill credit to the nearest bot that is alive and closest
									let closestBot = -1;
									let closestDist = Number.POSITIVE_INFINITY;
									for (let bi = 0; bi < botAgents.length; bi += 1) {
										const ba = botAgents[bi];
										if (ba.bot.health <= 0 || ba.bot.escaped) continue;
										const bd = dist2({ x: ba.bot.x, z: ba.bot.z }, { x: r.x, z: r.z });
										if (bd < closestDist) { closestDist = bd; closestBot = bi; }
									}
									if (closestBot >= 0 && closestDist < 10 * 10) {
										botAgents[closestBot].enemyKills += 1;
										botAgents[closestBot].fitness = computeFitness(botAgents[closestBot].fitness, {
											underThreat: true,
											enemyKill: true,
											successfulThrow: true,
											enemyHit: true,
										});
										population[closestBot].fitness = botAgents[closestBot].fitness;
									}
									break;
								}
							}
						}
					}
				}
			}

			for (let i = 0; i < botAgents.length; i += 1) {
				const a = botAgents[i];
				a.mesh.position.set(a.bot.x, 1.1, a.bot.z);
				a.mesh.rotation.y = Math.atan2(a.bot.headingX, a.bot.headingZ);
				a.mesh.material = i === selectedBotIdx ? selectedMat : topGreenIdxSet.has(i) ? championMat : botMat;
			}

			// Update selection ring to follow selected bot
			const selBot = botAgents[selectedBotIdx];
			if (selBot && selBot.bot.health > 0) {
				selectionRing.setEnabled(true);
				selectionRing.position.set(selBot.bot.x, 0.08, selBot.bot.z);
			} else {
				selectionRing.setEnabled(false);
			}

			syncEnemies();
			syncRocks();
			syncFoods();

			// UI snapshot
			if (now - lastUiAt > 160) {
				lastUiAt = now;
				const aliveEnemies = enemies.filter((e) => e.alive).length;
				const activeRocks = rocks.filter((r) => r.active).length;
				const bestAgent = botAgents[generationBestIdx] ?? botAgents[0];
				const bestBot = bestAgent?.bot;
				const localHint = settingsRef.current.aiProvider === "local" && !localBootstrap.ready ? ` (${localBootstrap.status})` : "";
				const imgHint = settingsRef.current.aiProvider === "local" ? (lastAiSentImage ? " [img]" : " [no-img]") : "";
				const status = (bestBot?.health ?? 0) <= 0
					? `All dead G${generation}`
					: pausedRef.current
						? "Paused"
						: pendingAi
							? "Waiting for AI..."
							: settingsRef.current.aiProvider === "none"
								? "No AI configured (fallback running)"
								: `Running G${generation} · 10 bots · survive & explore${localHint}${imgHint}`;

				onUi({
					health: bestBot?.health ?? 0,
					energy: bestBot?.energy ?? 0,
					enemiesAlive: aliveEnemies,
					rocksActive: activeRocks,
					organs: bestBot?.organs ?? 0,
					escaped: false,
					dead: (bestBot?.health ?? 0) <= 0,
					status,
				});
				emitSelectedBotSnapshot();
			}

			scene.render();
		});

		const onResize = () => engine.resize();
		window.addEventListener("resize", onResize);

		return () => {
			window.removeEventListener("resize", onResize);
			window.removeEventListener("sim4-reset", resetHandler as any);
			window.removeEventListener("sim6-export-generation", exportHandler as EventListener);
			window.removeEventListener("sim6-load-generation", loadHandler as EventListener);
			window.removeEventListener("keydown", keySpawnHandler);
			engine.dispose();
		};
	}, []);

	return <canvas ref={canvasRef} className="fixed inset-0 block h-screen w-screen" />;
}

function DenseNeuralNetGraph({ snap }: { snap: NeuralNetSnapshot }) {
	const width = 340;
	const graphHeight = 210;
	const legendHeight = 14;
	const height = graphHeight + legendHeight;
	const padY = 14;
	const inputX = 22;
	const h1X = 124;
	const h2X = 222;
	const outputX = 318;

	const inputYs = Array.from({ length: snap.inputs.activations.length }, (_, i) => padY + (i * (graphHeight - 2 * padY)) / Math.max(1, snap.inputs.activations.length - 1));
	const h1Ys = Array.from({ length: snap.hidden1.activations.length }, (_, i) => padY + (i * (graphHeight - 2 * padY)) / Math.max(1, snap.hidden1.activations.length - 1));
	const h2Ys = Array.from({ length: snap.hidden2.activations.length }, (_, i) => padY + (i * (graphHeight - 2 * padY)) / Math.max(1, snap.hidden2.activations.length - 1));
	const outYs = Array.from({ length: snap.outputs.length }, (_, i) => padY + (i * (graphHeight - 2 * padY)) / Math.max(1, snap.outputs.length - 1));

	const clamp01 = (v: number) => clamp(v, 0, 1);
	const normAct = (v: number | undefined) => {
		if (!Number.isFinite(v as number)) return 0;
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
		const lineWidth = 0.5 + abs * 2.1;
		const color = w >= 0 ? "#38bdf8" : "#fb7185";
		const actBoost = clamp01((Math.abs(normAct(sourceAct)) + Math.abs(normAct(targetAct))) * 0.5);
		const opacity = 0.08 + abs * 0.55 + actBoost * 0.35;
		return { lineWidth, color, opacity };
	};

	return (
		<svg width={width} height={height} className="w-full rounded-lg border border-slate-700 bg-slate-950/40">
			<defs>
				<linearGradient id="denseActScale" x1="0" y1="0" x2="1" y2="0">
					<stop offset="0%" stopColor="#fb7185" />
					<stop offset="50%" stopColor="#94a3b8" />
					<stop offset="100%" stopColor="#38bdf8" />
				</linearGradient>
			</defs>

			{snap.w1.map((row, i) =>
				row.map((w, h) => {
					const s = strokeForWeight(w, snap.inputs.activations[i], snap.hidden1.activations[h]);
					return <line key={`w1-${i}-${h}`} x1={inputX} y1={inputYs[i]} x2={h1X} y2={h1Ys[h]} stroke={s.color} strokeWidth={s.lineWidth} opacity={s.opacity} />;
				}),
			)}

			{snap.w2.map((row, i) =>
				row.map((w, h) => {
					const s = strokeForWeight(w, snap.hidden1.activations[i], snap.hidden2.activations[h]);
					return <line key={`w2-${i}-${h}`} x1={h1X} y1={h1Ys[i]} x2={h2X} y2={h2Ys[h]} stroke={s.color} strokeWidth={s.lineWidth} opacity={s.opacity * 0.95} />;
				}),
			)}

			{snap.w3.map((row, i) =>
				row.map((w, o) => {
					const s = strokeForWeight(w, snap.hidden2.activations[i], snap.outputs[o]?.value);
					return <line key={`w3-${i}-${o}`} x1={h2X} y1={h2Ys[i]} x2={outputX} y2={outYs[o]} stroke={s.color} strokeWidth={s.lineWidth} opacity={s.opacity} />;
				}),
			)}

			{inputYs.map((y, i) => (
				<g key={`in-${i}`}>
					<circle cx={inputX} cy={y} r={actRadius(3.8, snap.inputs.activations[i])} fill={actColor(snap.inputs.activations[i], "#94a3b8")} opacity={actAlpha(snap.inputs.activations[i])} />
					<text x={inputX} y={y} textAnchor="middle" dominantBaseline="middle" fontSize={8} fontWeight={700} fill="#0b1220" stroke="#e2e8f0" strokeWidth={0.8} paintOrder="stroke" style={{ pointerEvents: "none", userSelect: "none" }}>
						I
					</text>
				</g>
			))}

			{h1Ys.map((y, i) => (
				<circle key={`h1-${i}`} cx={h1X} cy={y} r={actRadius(3.8, snap.hidden1.activations[i])} fill={actColor(snap.hidden1.activations[i], "#e2e8f0")} opacity={actAlpha(snap.hidden1.activations[i])} />
			))}

			{h2Ys.map((y, i) => (
				<circle key={`h2-${i}`} cx={h2X} cy={y} r={actRadius(3.8, snap.hidden2.activations[i])} fill={actColor(snap.hidden2.activations[i], "#e2e8f0")} opacity={actAlpha(snap.hidden2.activations[i])} />
			))}

			{outYs.map((y, o) => (
				<g key={`out-${o}`}>
					<circle cx={outputX} cy={y} r={actRadius(4.8, snap.outputs[o]?.value)} fill={actColor(snap.outputs[o]?.value, "#facc15")} opacity={actAlpha(snap.outputs[o]?.value)} />
					<text x={outputX} y={y} textAnchor="middle" dominantBaseline="middle" fontSize={8} fontWeight={800} fill="#0b1220" stroke="#e2e8f0" strokeWidth={0.9} paintOrder="stroke" style={{ pointerEvents: "none", userSelect: "none" }}>
						{ACTION_OUTPUT_SHORT[o] ?? String(o + 1)}
					</text>
				</g>
			))}

			<text x={inputX} y={12} textAnchor="middle" fontSize={8} fill="#cbd5e1">In</text>
			<text x={h1X} y={12} textAnchor="middle" fontSize={8} fill="#cbd5e1">H1</text>
			<text x={h2X} y={12} textAnchor="middle" fontSize={8} fill="#cbd5e1">H2</text>
			<text x={outputX} y={12} textAnchor="middle" fontSize={8} fill="#cbd5e1">Out</text>

			<rect x={14} y={graphHeight + 5} width={width - 28} height={6} rx={3} fill="url(#denseActScale)" opacity={0.95} stroke="#334155" strokeOpacity={0.7} />
			<text x={14} y={graphHeight + 4} textAnchor="start" fontSize={9} fontWeight={700} fill="#e2e8f0" opacity={0.9} style={{ userSelect: "none" }}>
				Low activation
			</text>
			<text x={width - 14} y={graphHeight + 4} textAnchor="end" fontSize={9} fontWeight={700} fill="#e2e8f0" opacity={0.9} style={{ userSelect: "none" }}>
				High activation
			</text>
		</svg>
	);
}

export default function CityEscapeGeneticSim3D() {
	const defaultLocalUrl = import.meta.env.DEV ? "/lmstudio" : "http://127.0.0.1:1234";
	const [settings, setSettings] = useState<Sim4Settings>({
		aiProvider: "genetic",
		aiModel: "large-nn-ga",
		localEndpointUrl: defaultLocalUrl,
		localModelName: "google/gemma-4-e2b",
		replicateApiToken: "",
		replicateModel: "",
		visionResolution: 64,
		aiHz: 2,
		generationSeconds: 300,
		enemySpeed: 3.2,
		enemyDamagePerSec: 14,
		botMaxSpeed: 6,
		botAccel: 10,
		energyDrainPerSec: 0.3,
	});
	const [paused, setPaused] = useState(false);
	const [controlsCollapsed, setControlsCollapsed] = useState(false);
	const [visionPreviewUrl, setVisionPreviewUrl] = useState<string>("");
	const [aiLog, setAiLog] = useState<AiExchange[]>([]);
	const [nnSnapshot, setNnSnapshot] = useState<NeuralNetSnapshot | null>(null);
	const [selectedBot, setSelectedBot] = useState<SelectedBotSnapshot | null>(null);
	const [generationIoStatus, setGenerationIoStatus] = useState<string>("");
	const generationLoadInputRef = useRef<HTMLInputElement | null>(null);
	const prevAiExchangeRef = useRef<AiExchange | null>(null);
	const [ui, setUi] = useState<UiSnapshot>(() => ({
		health: 100,
		energy: 100,
		enemiesAlive: 0,
		rocksActive: 0,
		organs: 0,
		escaped: false,
		dead: false,
		status: "Running",
	}));

	const restart = () => {
		window.dispatchEvent(new Event("sim4-reset"));
		prevAiExchangeRef.current = null;
		setAiLog([]);
		setNnSnapshot(null);
		setSelectedBot(null);
	};

	const pushAiExchange = (exchange: AiExchange) => {
		setAiLog((prev) => {
			prevAiExchangeRef.current = prev[0] ?? null;
			return [exchange];
		});
	};

	const exportGeneration = () => {
		window.dispatchEvent(
			new CustomEvent("sim6-export-generation", {
				detail: {
					onData: (payload: SerializedGeneration) => {
						const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
						const url = URL.createObjectURL(blob);
						const a = document.createElement("a");
						a.href = url;
						a.download = `sim6-generation-g${payload.generation}.json`;
						a.click();
						URL.revokeObjectURL(url);
						setGenerationIoStatus(`Exported generation ${payload.generation}`);
					},
				},
			}),
		);
	};

	const onLoadGenerationFile = (ev: React.ChangeEvent<HTMLInputElement>) => {
		const file = ev.target.files?.[0];
		if (!file) return;
		const reader = new FileReader();
		reader.onload = () => {
			try {
				const payload = JSON.parse(String(reader.result ?? "{}")) as SerializedGeneration;
				window.dispatchEvent(new CustomEvent("sim6-load-generation", { detail: { payload } }));
				setGenerationIoStatus(`Loaded generation ${payload.generation || 1}`);
			} catch {
				setGenerationIoStatus("Failed to load generation file");
			}
		};
		reader.readAsText(file);
		ev.target.value = "";
	};

	return (
		<>
			<BabylonWorld settings={settings} paused={paused} onUi={setUi} onVisionPreview={setVisionPreviewUrl} onAiExchange={pushAiExchange} onNeuralNetUpdate={setNnSnapshot} onSelectedBotUpdate={setSelectedBot} />
			<aside className="selected-panel">
								<Card className="bg-transparent border-0 shadow-none mt-3">
					<div className="controls-title" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
						<span>Bot Vision</span>
						<span className="text-xs text-slate-300">selected bot</span>
					</div>
					<CardContent className="space-y-2">
						{visionPreviewUrl ? (
							<img
								src={visionPreviewUrl}
								alt="Bot vision"
								className="w-full rounded-md border border-slate-700 bg-slate-950/40"
								style={{ imageRendering: "pixelated", aspectRatio: "1 / 1" }}
							/>
						) : (
							<div className="rounded-xl border border-slate-700 p-3 text-xs text-slate-300">Waiting for first frame...</div>
						)}
						<div className="text-xs text-slate-300">Resolution: {Math.round(settings.visionResolution)}x{Math.round(settings.visionResolution)}</div>
					</CardContent>
				</Card>

				<Card className="bg-transparent border-0 shadow-none mt-3">
					<div className="controls-title" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
						<span>Neural Net</span>
						<span className="text-xs text-slate-300">live</span>
					</div>
					<CardContent className="space-y-2">
						{!nnSnapshot ? (
							<div className="rounded-xl border border-slate-700 p-3 text-xs text-slate-300">Waiting for first neural pass...</div>
						) : (
							<div className="rounded-xl border border-slate-700 p-2 text-xs text-left space-y-2">
								<div className="grid grid-cols-2 gap-2">
									<div className="rounded-md border border-slate-700 bg-slate-950/50 p-2">
										<div className="text-slate-400">Architecture</div>
										<div className="font-mono text-slate-100 mt-1">{nnSnapshot.inputSize} → {nnSnapshot.h1Size} → {nnSnapshot.h2Size} → {nnSnapshot.outputSize}</div>
									</div>
									<div className="rounded-md border border-slate-700 bg-slate-950/50 p-2">
										<div className="text-slate-400">Genome</div>
										<div className="font-mono text-slate-100 mt-1">G{nnSnapshot.generation} · {nnSnapshot.genomeIndex + 1}/{nnSnapshot.populationSize}</div>
									</div>
								</div>
								
								<div>
									<div className="font-semibold text-slate-200 mb-1"> Neural net (live)</div>
									<DenseNeuralNetGraph snap={nnSnapshot} />
									{/* <div className="grid grid-cols-2 gap-1 mt-2">
										{nnSnapshot.outputs.map((o) => (
											<div key={o.name} className="rounded border border-slate-700 bg-slate-950/40 px-2 py-1">
												<div className="text-slate-400">{o.name}</div>
												<div className="font-mono text-slate-100">{o.value.toFixed(3)}</div>
											</div>
										))}
									</div> */}
								</div>
							</div>
						)}
					</CardContent>
				</Card>
				<Card className="bg-transparent border-0 shadow-none">
					<div className="controls-title" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
						<span>Selected Bot</span>
						<span className="text-xs text-slate-300">click bot to select</span>
					</div>
					<CardContent className="space-y-3">
						{!selectedBot ? (
							<div className="rounded-xl border border-slate-700 p-3 text-xs text-slate-300">Select a bot in the scene to inspect it.</div>
						) : (
							<div className="rounded-xl border border-slate-700 p-2 text-xs text-left space-y-2">
								<div className="grid grid-cols-2 gap-2">
									<div className="rounded-md border border-slate-700 bg-slate-950/50 p-2">
										<div className="text-slate-400">Bot</div>
										<div className="font-mono text-slate-100 mt-1">#{selectedBot.index + 1}</div>
									</div>
									<div className="rounded-md border border-slate-700 bg-slate-950/50 p-2">
										<div className="text-slate-400">Fitness</div>
										<div className="font-mono text-slate-100 mt-1">{selectedBot.fitness.toFixed(3)}</div>
									</div>
								</div>
								<table className="w-full table-fixed border-collapse text-left">
									<tbody>
										<tr><th className="w-[42%] pr-2 py-[2px] text-slate-400 font-normal">Health</th><td className="py-[2px] text-slate-100 font-mono">{selectedBot.health.toFixed(1)}</td></tr>
										<tr><th className="w-[42%] pr-2 py-[2px] text-slate-400 font-normal">Energy</th><td className="py-[2px] text-slate-100 font-mono">{selectedBot.energy.toFixed(1)}</td></tr>
										<tr><th className="w-[42%] pr-2 py-[2px] text-slate-400 font-normal">Kills</th><td className="py-[2px] text-slate-100 font-mono">{selectedBot.enemyKills}</td></tr>
										<tr><th className="w-[42%] pr-2 py-[2px] text-slate-400 font-normal">Direction Changes</th><td className="py-[2px] text-slate-100 font-mono">{selectedBot.directionChanges}</td></tr>
										<tr><th className="w-[42%] pr-2 py-[2px] text-slate-400 font-normal">Facing (x,z)</th><td className="py-[2px] text-slate-100 font-mono">{selectedBot.heading.x.toFixed(2)}, {selectedBot.heading.z.toFixed(2)}</td></tr>
										<tr><th className="w-[42%] pr-2 py-[2px] text-slate-400 font-normal">Facing Yaw</th><td className="py-[2px] text-slate-100 font-mono">{selectedBot.heading.yawDeg.toFixed(1)} deg</td></tr>
										<tr><th className="w-[42%] pr-2 py-[2px] text-slate-400 font-normal">Position</th><td className="py-[2px] text-slate-100 font-mono">{selectedBot.position.x.toFixed(1)}, {selectedBot.position.z.toFixed(1)}</td></tr>
									</tbody>
								</table>
								<div>
									<div className="font-semibold text-slate-200 mb-1">Current Action</div>
									<div className="grid grid-cols-2 gap-1">
										{Object.entries(selectedBot.action).map(([k, v]) => (
											<div key={k} className="rounded border border-slate-700 bg-slate-950/40 px-2 py-1">
												<div className="text-slate-400">{k}</div>
												<div className="font-mono text-slate-100">{typeof v === "number" ? v.toFixed(3) : String(v)}</div>
											</div>
										))}
									</div>
								</div>
							</div>
						)}
					</CardContent>
				</Card>

			</aside>
			<aside className="controls-panel">
				<Card className="bg-transparent border-0 shadow-none">
					<div className="controls-title" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
						<span>Controls</span>
						<button onClick={() => setControlsCollapsed((v) => !v)}>{controlsCollapsed ? "Expand" : "Collapse"}</button>
					</div>

					{!controlsCollapsed && (
						<CardContent className="space-y-4">
							<div className="rounded-xl border border-slate-700 p-3 text-sm space-y-2">
								<div className="font-semibold">Generation Controls</div>
								<div className="grid grid-cols-2 gap-2">
									<Button variant="secondary" onClick={exportGeneration}>Export Generation</Button>
									<Button variant="secondary" onClick={() => generationLoadInputRef.current?.click()}>Load Generation</Button>
								</div>
								<input
									ref={generationLoadInputRef}
									type="file"
									accept="application/json"
									onChange={onLoadGenerationFile}
									style={{ display: "none" }}
								/>
								{generationIoStatus ? <div className="text-xs text-slate-300">{generationIoStatus}</div> : null}
							</div>

							<div className="rounded-xl border border-slate-700 p-3 text-sm space-y-1">
								<div>Status: <strong>{ui.status}</strong></div>
								<div>Enemies alive: <strong>{ui.enemiesAlive}</strong></div>
								<div>Rocks active: <strong>{ui.rocksActive}</strong></div>
								<div className="text-xs text-slate-300">Tip: click a bot to select. Move mouse on ground, press E to add enemy, S to add stone.</div>
							</div>

							<div className="rounded-xl border border-slate-700 p-3 text-sm space-y-2">
								<div className="font-semibold">Genetic Brain Log</div>
								{aiLog.length === 0 ? (
									<div className="text-xs text-slate-300">Waiting for first AI exchange...</div>
								) : (
									<div className="max-h-[18vh] overflow-auto space-y-1 pr-1">
										{aiLog.map((item) => (
											<div key={item.id} className="rounded border border-slate-700 bg-slate-950/40 px-2 py-1 text-xs">
												<div className="text-slate-300">{new Date(item.at).toLocaleTimeString()} · {item.provider}/{item.model}</div>
												<div className="text-slate-400 truncate">{item.replied}</div>
											</div>
										))}
									</div>
								)}
							</div>

							<div className="grid grid-cols-2 gap-2">
								<Button onClick={() => setPaused((p) => !p)}>{paused ? "Resume" : "Pause"}</Button>
								<Button variant="secondary" onClick={restart}>
									Restart
								</Button>
							</div>

							<div className="space-y-3">
								<div>
									<div className="flex justify-between text-sm mb-1"><span>AI Provider</span><span>{settings.aiProvider}</span></div>
									<select
										className="w-full rounded-md border border-slate-700 bg-slate-900/50 p-2 text-sm"
										value={settings.aiProvider}
										onChange={(e) => setSettings((p) => ({ ...p, aiProvider: e.target.value as AiProvider }))}
									>
										<option value="genetic">genetic (large neural net)</option>
									</select>
								</div>

								<div>
									<div className="flex justify-between text-sm mb-1"><span>AI Model</span><span>{settings.aiModel}</span></div>
									<select
										className="w-full rounded-md border border-slate-700 bg-slate-900/50 p-2 text-sm"
										value={settings.aiModel}
										onChange={(e) => setSettings((p) => ({ ...p, aiModel: e.target.value as AiModel }))}
									>
										<option value="large-nn-ga">large-nn-ga</option>
									</select>
								</div>

								{settings.aiProvider === "local" && (
									<div>
										<div className="text-sm mb-1">LM Studio base URL</div>
										<input
											className="w-full rounded-md border border-slate-700 bg-slate-900/50 p-2 text-sm"
											value={settings.localEndpointUrl}
											onChange={(e) => setSettings((p) => ({ ...p, localEndpointUrl: e.target.value }))}
											placeholder="/lmstudio (dev proxy) or http://127.0.0.1:1234"
										/>
										<div className="text-xs text-slate-300 mt-1">Uses `POST /api/v1/chat` and `POST /api/v1/models/load`.</div>
										<div className="text-xs text-slate-300 mt-1">If your server requires `messages`, it will fall back to `/v1/chat/completions`.</div>
										<div className="text-xs text-slate-300 mt-1">Expected JSON response: {"{left,right,forward,backward,speed,pick,throw,grow}"}.</div>
										<div className="text-sm mt-3 mb-1">Local model name</div>
										<input
											className="w-full rounded-md border border-slate-700 bg-slate-900/50 p-2 text-sm"
											value={settings.localModelName}
											onChange={(e) => setSettings((p) => ({ ...p, localModelName: e.target.value }))}
											placeholder='Example: "google/gemma-4-e2b"'
										/>
										<div className="text-xs text-slate-300 mt-1">List available models at `GET http://127.0.0.1:1234/api/v1/models`.</div>
									</div>
								)}

								{settings.aiProvider === "replicate" && (
									<>
										<div>
											<div className="text-sm mb-1">Replicate API token</div>
											<input
												className="w-full rounded-md border border-slate-700 bg-slate-900/50 p-2 text-sm"
												value={settings.replicateApiToken}
												onChange={(e) => setSettings((p) => ({ ...p, replicateApiToken: e.target.value }))}
												placeholder="Token ..."
										/>
										</div>
										<div>
											<div className="text-sm mb-1">Replicate model/version</div>
											<input
												className="w-full rounded-md border border-slate-700 bg-slate-900/50 p-2 text-sm"
												value={settings.replicateModel}
												onChange={(e) => setSettings((p) => ({ ...p, replicateModel: e.target.value }))}
												placeholder="<version-id or proxy>"
											/>
											<div className="text-xs text-slate-300 mt-1">This uses a minimal Replicate-style call; many setups use a proxy that returns the action directly.</div>
										</div>
									</>
								)}

								<div>
									<div className="flex justify-between text-sm mb-1"><span>Vision resolution</span><span>{Math.round(settings.visionResolution)}×{Math.round(settings.visionResolution)}</span></div>
									<Slider value={[settings.visionResolution]} min={16} max={96} step={8} onValueChange={(v: number[]) => setSettings((p) => ({ ...p, visionResolution: v[0] }))} />
								</div>

								<div>
									<div className="flex justify-between text-sm mb-1"><span>AI tick (Hz)</span><span>{settings.aiHz.toFixed(1)}</span></div>
									<Slider value={[settings.aiHz]} min={0.5} max={8} step={0.5} onValueChange={(v: number[]) => setSettings((p) => ({ ...p, aiHz: v[0] }))} />
								</div>

								<div>
									<div className="flex justify-between text-sm mb-1"><span>Generation duration (s)</span><span>{settings.generationSeconds.toFixed(0)}</span></div>
									<Slider value={[settings.generationSeconds]} min={5} max={120} step={1} onValueChange={(v: number[]) => setSettings((p) => ({ ...p, generationSeconds: v[0] }))} />
								</div>

								<div>
									<div className="flex justify-between text-sm mb-1"><span>Enemy speed</span><span>{settings.enemySpeed.toFixed(1)}</span></div>
									<Slider value={[settings.enemySpeed]} min={1} max={7} step={0.2} onValueChange={(v: number[]) => setSettings((p) => ({ ...p, enemySpeed: v[0] }))} />
								</div>
							</div>

							<div className="rounded-xl border border-slate-700 p-3 text-xs text-slate-300 leading-relaxed">
								Goal: reach the green exit ring. Stay alive: enemies damage on collision. The bot can optionally pick/throw rocks and grow organs (energy/health capacity) via AI actions.
							</div>
						</CardContent>
					)}
				</Card>
			</aside>
		</>
	);
}

