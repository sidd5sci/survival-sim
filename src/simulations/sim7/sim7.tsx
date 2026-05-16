import { useEffect, useRef, useState } from "react";
import * as BABYLON from "babylonjs";
import { Card, CardContent } from "../../components/ui/card";
import { Button } from "../../components/ui/button";
import { Slider } from "../../components/ui/slider";

type JointAxisLimit = { min: number; max: number };
type JointLimits = {
	x: JointAxisLimit;
	y: JointAxisLimit;
	z: JointAxisLimit;
};

type WalkerRig = {
	root: BABYLON.TransformNode;
	bodyMat: BABYLON.StandardMaterial;
	limbMat: BABYLON.StandardMaterial;
	pelvis: BABYLON.TransformNode;
	chest: BABYLON.TransformNode;
	hipsBar: BABYLON.Mesh;
	spineStick: BABYLON.Mesh;
	chestStick: BABYLON.Mesh;
	neckStick: BABYLON.Mesh;
	shoulderBar: BABYLON.Mesh;
	head: BABYLON.Mesh;
	leftHip: BABYLON.TransformNode;
	rightHip: BABYLON.TransformNode;
	leftKnee: BABYLON.TransformNode;
	rightKnee: BABYLON.TransformNode;
	leftAnkle: BABYLON.TransformNode;
	rightAnkle: BABYLON.TransformNode;
	leftShoulder: BABYLON.TransformNode;
	rightShoulder: BABYLON.TransformNode;
	leftElbow: BABYLON.TransformNode;
	rightElbow: BABYLON.TransformNode;
	leftThigh: BABYLON.Mesh;
	rightThigh: BABYLON.Mesh;
	leftShin: BABYLON.Mesh;
	rightShin: BABYLON.Mesh;
	leftFoot: BABYLON.Mesh;
	rightFoot: BABYLON.Mesh;
	leftToe: BABYLON.Mesh;
	rightToe: BABYLON.Mesh;
	leftUpperArm: BABYLON.Mesh;
	rightUpperArm: BABYLON.Mesh;
	leftForearm: BABYLON.Mesh;
	rightForearm: BABYLON.Mesh;
};

type WalkerState = {
	t: number;
	x: number;
	y: number;
	vx: number;
	vy: number;
	torsoAngle: number;
	torsoVel: number;
	jointAngles: Float32Array;
	jointVels: Float32Array;
	targetCache: Float32Array;
	phase: number;
	stepI: number;
	alive: boolean;
	grounded: boolean;
	airTime: number;
	aliveTime: number;
	maxX: number;
	rewardAcc: number;
	energy: number;
	lastX: number;
	leftContactPrev: number;
	rightContactPrev: number;
	contactSwitches: number;
	motorDriveAcc: number;
	fit: number;
};

type Genome = {
	weights: Float32Array;
	fitness: number;
};

type UiSnapshot = {
	aliveSecs: number;
	speed: number;
	status: string;
};

type NeuralLayerSample = {
	indices: number[];
	activations: number[];
};

type NeuralNetSnapshot = {
	selectedIndex: number;
	fitness: number;
	weightAbsMean: {
		w1: number;
		w2: number;
		w3: number;
	};
	inputs: NeuralLayerSample;
	hidden1: NeuralLayerSample;
	hidden2: NeuralLayerSample;
	outputs: Array<{ index: number; name: string; value: number }>;
	w1: number[][];
	w2: number[][];
	w3: number[][];
};

const POP_SIZE = 20;
const ELITE = 10;
const TOURN_K = 5;
const MUT_RATE = 0.08;
const MUT_STD = 0.15;
const CROSS_RATE = 0.5;

const N_IN = 20;
const N_H1 = 64;
const N_H2 = 64;
const N_OUT = 6;
const N_W = N_IN * N_H1 + N_H1 + N_H1 * N_H2 + N_H2 + N_H2 * N_OUT + N_OUT;

const EVAL_SECONDS = 12;
const DT = 1 / 60;
const GRAVITY = 9.81;
const ACTION_HZ = 30;
const ACTION_EVERY = Math.max(1, Math.round((1 / ACTION_HZ) / DT));
const TARGET_SMOOTH = 0.34;

const FALL_PITCH = 1.12;
const MAX_Z = 1.55;
const MAX_AIR_TIME = 1.0;
const FALLEN_PELVIS_Y = -0.68;
const MIN_RESET_SECONDS = 1.2;

const HIP_HEIGHT = 0.84;
const THIGH_LEN = 0.42;
const SHIN_LEN = 0.38;
const FOOT_TOE_LEN = 0.08;
const FOOT_CLEARANCE = 0.085;

const SURFACE_FRICTION = 2.2;
const AIR_FRICTION = 0.12;

const HIP_KP = 10.5;
const HIP_KD = 2.4;

const JOINT_LIMITS_1D: Array<[number, number]> = [
	[-0.55, 0.85],
	[0.05, 1.45],
	[-0.35, 0.32],
	[-0.55, 0.85],
	[0.05, 1.45],
	[-0.35, 0.32],
];

const ACTION_OUTPUT_NAMES = ["L Hip", "L Knee", "L Ankle", "R Hip", "R Knee", "R Ankle"];
const ACTION_OUTPUT_SHORT = ["LH", "LK", "LA", "RH", "RK", "RA"];

const HUMAN_LIMITS = {
	hip: {
		x: { min: -0.2, max: 0.2 },
		y: { min: -0.25, max: 0.25 },
		z: { min: -0.85, max: 1.05 },
	} as JointLimits,
	knee: {
		x: { min: 0, max: 0 },
		y: { min: 0, max: 0 },
		z: { min: -1.55, max: 0.05 },
	} as JointLimits,
	ankle: {
		x: { min: -0.15, max: 0.15 },
		y: { min: -0.15, max: 0.15 },
		z: { min: -0.45, max: 0.45 },
	} as JointLimits,
	shoulder: {
		x: { min: -0.35, max: 0.35 },
		y: { min: -0.35, max: 0.35 },
		z: { min: -1.2, max: 1.2 },
	} as JointLimits,
	elbow: {
		x: { min: 0, max: 0 },
		y: { min: -0.15, max: 0.15 },
		z: { min: -1.45, max: 0.05 },
	} as JointLimits,
};

const BODY_DIFFUSE_DEFAULT = BABYLON.Color3.FromHexString("#e2e8f0");
const BODY_EMISSIVE_DEFAULT = BABYLON.Color3.FromHexString("#0f172a");
const LIMB_DIFFUSE_DEFAULT = BABYLON.Color3.FromHexString("#93c5fd");
const LIMB_EMISSIVE_DEFAULT = BABYLON.Color3.FromHexString("#1e3a8a");
const BODY_DIFFUSE_SELECTED = BABYLON.Color3.FromHexString("#bfdbfe");
const BODY_EMISSIVE_SELECTED = BABYLON.Color3.FromHexString("#1d4ed8");
const LIMB_DIFFUSE_SELECTED = BABYLON.Color3.FromHexString("#60a5fa");
const LIMB_EMISSIVE_SELECTED = BABYLON.Color3.FromHexString("#1e40af");

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

function rand(min: number, max: number): number {
	return Math.random() * (max - min) + min;
}

function randn(): number {
	let u = 0;
	let v = 0;
	while (u === 0) u = Math.random();
	while (v === 0) v = Math.random();
	return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function setJointRotation(node: BABYLON.TransformNode, x: number, y: number, z: number, limits: JointLimits) {
	node.rotation.x = clamp(x, limits.x.min, limits.x.max);
	node.rotation.y = clamp(y, limits.y.min, limits.y.max);
	node.rotation.z = clamp(z, limits.z.min, limits.z.max);
}

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
	for (let i = 0; i < maxNodes; i += 1) out.push(Math.round((i * (total - 1)) / denom));
	return Array.from(new Set(out));
}

function sampleLayer(values: Float32Array, maxNodes: number): NeuralLayerSample {
	const indices = sampleEvenIndices(values.length, maxNodes);
	return { indices, activations: indices.map((idx) => values[idx] ?? 0) };
}

function sampleWeightsW1(net: ReturnType<typeof decodeNet>, inputIdx: number[], h1Idx: number[]): number[][] {
	return inputIdx.map((src) => h1Idx.map((dst) => net.w1[dst * N_IN + src] ?? 0));
}

function sampleWeightsW2(net: ReturnType<typeof decodeNet>, h1Idx: number[], h2Idx: number[]): number[][] {
	return h1Idx.map((src) => h2Idx.map((dst) => net.w2[dst * N_H1 + src] ?? 0));
}

function sampleWeightsW3(net: ReturnType<typeof decodeNet>, h2Idx: number[], outIdx: number[]): number[][] {
	return h2Idx.map((src) => outIdx.map((dst) => net.w3[dst * N_H2 + src] ?? 0));
}

function forwardTrace(net: ReturnType<typeof decodeNet>, obs: Float32Array): { h1: Float32Array; h2: Float32Array; out: Float32Array } {
	const h1 = new Float32Array(N_H1);
	for (let j = 0; j < N_H1; j += 1) {
		let sum = net.b1[j];
		const row = j * N_IN;
		for (let i = 0; i < N_IN; i += 1) sum += net.w1[row + i] * obs[i];
		h1[j] = Math.tanh(sum);
	}
	const h2 = new Float32Array(N_H2);
	for (let j = 0; j < N_H2; j += 1) {
		let sum = net.b2[j];
		const row = j * N_H1;
		for (let i = 0; i < N_H1; i += 1) sum += net.w2[row + i] * h1[i];
		h2[j] = Math.tanh(sum);
	}
	const out = new Float32Array(N_OUT);
	for (let j = 0; j < N_OUT; j += 1) {
		let sum = net.b3[j];
		const row = j * N_H2;
		for (let i = 0; i < N_H2; i += 1) sum += net.w3[row + i] * h2[i];
		out[j] = Math.tanh(sum);
	}
	return { h1, h2, out };
}

function randomGenome(): Genome {
	const w = new Float32Array(N_W);
	for (let i = 0; i < w.length; i += 1) w[i] = randn() * 0.32;
	return { weights: w, fitness: 0 };
}

function cloneGenome(g: Genome): Genome {
	return { weights: new Float32Array(g.weights), fitness: 0 };
}

function crossoverAndMutate(a: Genome, b: Genome): Genome {
	const child = new Float32Array(N_W);
	for (let i = 0; i < N_W; i += 1) {
		child[i] = Math.random() < CROSS_RATE ? a.weights[i] : b.weights[i];
		if (Math.random() < MUT_RATE) child[i] += randn() * MUT_STD;
	}
	return { weights: child, fitness: 0 };
}

function decodeNet(genome: Float32Array) {
	let idx = 0;
	const take = (n: number) => {
		const s = genome.subarray(idx, idx + n);
		idx += n;
		return s;
	};
	const w1 = take(N_IN * N_H1);
	const b1 = take(N_H1);
	const w2 = take(N_H1 * N_H2);
	const b2 = take(N_H2);
	const w3 = take(N_H2 * N_OUT);
	const b3 = take(N_OUT);
	return { w1, b1, w2, b2, w3, b3 };
}

function forward(net: ReturnType<typeof decodeNet>, obs: Float32Array): Float32Array {
	const h1 = new Float32Array(N_H1);
	for (let j = 0; j < N_H1; j += 1) {
		let sum = net.b1[j];
		const row = j * N_IN;
		for (let i = 0; i < N_IN; i += 1) sum += net.w1[row + i] * obs[i];
		h1[j] = Math.tanh(sum);
	}
	const h2 = new Float32Array(N_H2);
	for (let j = 0; j < N_H2; j += 1) {
		let sum = net.b2[j];
		const row = j * N_H1;
		for (let i = 0; i < N_H1; i += 1) sum += net.w2[row + i] * h1[i];
		h2[j] = Math.tanh(sum);
	}
	const out = new Float32Array(N_OUT);
	for (let j = 0; j < N_OUT; j += 1) {
		let sum = net.b3[j];
		const row = j * N_H2;
		for (let i = 0; i < N_H2; i += 1) sum += net.w3[row + i] * h2[i];
		out[j] = Math.tanh(sum);
	}
	return out;
}

function freshWalker(): WalkerState {
	return {
		t: 0,
		x: 0,
		y: 0,
		vx: 0,
		vy: 0,
		torsoAngle: 0,
		torsoVel: 0,
		jointAngles: new Float32Array([0.08, 0.34, -0.02, -0.08, 0.34, 0.02]),
		jointVels: new Float32Array(6),
		targetCache: new Float32Array([0.08, 0.34, -0.02, -0.08, 0.34, 0.02]),
		phase: rand(0, Math.PI * 2),
		stepI: 0,
		alive: true,
		grounded: true,
		airTime: 0,
		aliveTime: 0,
		maxX: 0,
		rewardAcc: 0,
		energy: 0,
		lastX: 0,
		leftContactPrev: 1,
		rightContactPrev: 1,
		contactSwitches: 0,
		motorDriveAcc: 0,
		fit: 0,
	};
}

function footContacts(s: WalkerState): { lc: number; rc: number; grounded: boolean } {
	const hipH = HIP_HEIGHT + s.y;
	const lHip = s.jointAngles[0];
	const lKnee = s.jointAngles[1];
	const lAnkle = s.jointAngles[2];
	const rHip = s.jointAngles[3];
	const rKnee = s.jointAngles[4];
	const rAnkle = s.jointAngles[5];
	const lReach = THIGH_LEN * Math.cos(lHip) + SHIN_LEN * Math.cos(lHip - lKnee) + FOOT_TOE_LEN * Math.cos(lHip - lKnee - lAnkle);
	const rReach = THIGH_LEN * Math.cos(rHip) + SHIN_LEN * Math.cos(rHip - rKnee) + FOOT_TOE_LEN * Math.cos(rHip - rKnee - rAnkle);
	const lFootH = hipH - lReach;
	const rFootH = hipH - rReach;
	const lc = lFootH <= FOOT_CLEARANCE ? 1 : 0;
	const rc = rFootH <= FOOT_CLEARANCE ? 1 : 0;
	return { lc, rc, grounded: lc > 0 || rc > 0 };
}

function buildObservation(s: WalkerState): Float32Array {
	const { lc, rc } = footContacts(s);
	const obs = new Float32Array(N_IN);
	obs[0] = 1.0 + s.y;
	obs[1] = s.torsoAngle;
	obs[2] = s.vx;
	obs[3] = s.vy;
	for (let i = 0; i < 6; i += 1) obs[4 + i] = s.jointAngles[i];
	for (let i = 0; i < 6; i += 1) obs[10 + i] = s.jointVels[i];
	obs[16] = lc;
	obs[17] = rc;
	obs[18] = Math.sin(s.phase);
	obs[19] = Math.cos(s.phase);
	return obs;
}

function stepWalker(state: WalkerState, net: ReturnType<typeof decodeNet>, dt: number): void {
	if (!state.alive) {
		state.vy -= GRAVITY * dt;
		state.y += state.vy * dt;
		if (state.y < FALLEN_PELVIS_Y) {
			state.y = FALLEN_PELVIS_Y;
			state.vy = 0;
			state.grounded = true;
		}
		state.vx *= Math.max(0, 1 - 6 * dt);
		state.x += state.vx * dt;
		state.torsoVel += (1.45 - state.torsoAngle) * 3.5 * dt;
		state.torsoVel *= Math.max(0, 1 - 3.2 * dt);
		state.torsoAngle += state.torsoVel * dt;
		state.t += dt;
		return;
	}

	state.stepI += 1;
	const obs = buildObservation(state);

	if (state.stepI % ACTION_EVERY === 1) {
		const raw = forward(net, obs);
		const p = state.phase;
		const tutor = new Float32Array([
			0.24 * Math.sin(p),
			0.55 + 0.32 * Math.max(0, Math.sin(p + 0.1)),
			-0.08 + 0.18 * Math.sin(p + 0.2),
			-0.24 * Math.sin(p),
			0.55 + 0.32 * Math.max(0, Math.sin(p + Math.PI + 0.1)),
			0.08 + 0.18 * Math.sin(p + Math.PI + 0.2),
		]);
		const tutorBlend = clamp(1 - state.aliveTime / 2.6, 0, 1) * 0.3;
		for (let i = 0; i < 6; i += 1) {
			const lo = JOINT_LIMITS_1D[i][0];
			const hi = JOINT_LIMITS_1D[i][1];
			const nnTarget = lo + 0.5 * (raw[i] + 1.0) * (hi - lo);
			const tutorTarget = clamp(tutor[i], lo, hi);
			const target = nnTarget * (1 - tutorBlend) + tutorTarget * tutorBlend;
			state.targetCache[i] = (1 - TARGET_SMOOTH) * state.targetCache[i] + TARGET_SMOOTH * target;
		}
	}

	for (let i = 0; i < 6; i += 1) {
		const err = state.targetCache[i] - state.jointAngles[i];
		const acc = HIP_KP * err - HIP_KD * state.jointVels[i];
		state.jointVels[i] += acc * dt;
		state.jointVels[i] *= 0.985;
		state.jointAngles[i] += state.jointVels[i] * dt;
		const lo = JOINT_LIMITS_1D[i][0];
		const hi = JOINT_LIMITS_1D[i][1];
		if (state.jointAngles[i] < lo) {
			state.jointAngles[i] = lo;
			state.jointVels[i] = 0;
		}
		if (state.jointAngles[i] > hi) {
			state.jointAngles[i] = hi;
			state.jointVels[i] = 0;
		}
	}

	const contacts = footContacts(state);
	state.grounded = contacts.grounded;
	state.airTime = contacts.grounded ? 0 : state.airTime + dt;
	if (contacts.lc !== state.leftContactPrev) state.contactSwitches += 1;
	if (contacts.rc !== state.rightContactPrev) state.contactSwitches += 1;
	state.leftContactPrev = contacts.lc;
	state.rightContactPrev = contacts.rc;
	state.phase += dt * 1.8;

	const stride = (state.jointVels[3] - state.jointVels[0]) * 0.28;
	state.motorDriveAcc += Math.abs(stride) * dt;
	if (contacts.grounded) state.vx += stride * dt;
	const friction = contacts.grounded ? SURFACE_FRICTION : AIR_FRICTION;
	state.vx -= state.vx * friction * dt;
	state.vx = clamp(state.vx, -1.2, 5.2);
	state.x += state.vx * dt;

	const desiredTorso = clamp((state.jointAngles[3] - state.jointAngles[0]) * 0.35, -0.6, 0.6);
	const torsoAcc = (desiredTorso - state.torsoAngle) * 7.4 - state.torsoVel * 2.7;
	state.torsoVel += torsoAcc * dt;
	state.torsoAngle += state.torsoVel * dt;

	state.vy -= GRAVITY * dt;
	if (contacts.grounded && state.vy < 0) {
		state.y = 0;
		state.vy = 0;
	} else {
		state.y += state.vy * dt;
		if (state.y < 0) {
			state.y = 0;
			state.vy = 0;
		}
	}

	const dx = state.x - state.lastX;
	state.lastX = state.x;
	const forwardStep = clamp(dx, -0.02, 0.025);
	const upright = Math.max(0, 1 - Math.abs(state.torsoAngle) / 1.5);
	const energy = Math.abs(state.jointVels[0]) + Math.abs(state.jointVels[1]) + Math.abs(state.jointVels[2]) + Math.abs(state.jointVels[3]) + Math.abs(state.jointVels[4]) + Math.abs(state.jointVels[5]);
	const symPen = Math.abs(state.jointAngles[0] + state.jointAngles[3]) + 0.4 * Math.abs(state.jointAngles[1] - state.jointAngles[4]);
	const alternating = Math.abs(state.jointVels[0] - state.jointVels[3]);
	const contactBonus = contacts.grounded ? 0.007 : 0;
	state.rewardAcc +=
		6.5 * Math.max(0, forwardStep) * upright +
		0.022 * upright +
		0.0015 * alternating +
		contactBonus -
		0.0022 * energy -
		0.003 * symPen -
		0.02 * Math.min(1, state.airTime / MAX_AIR_TIME);
	state.energy += energy * 0.001 * dt;

	state.t += dt;
	state.aliveTime += dt;
	state.maxX = Math.max(state.maxX, state.x);

	const worldZ = HIP_HEIGHT + state.y;
	if (Math.abs(state.torsoAngle) > FALL_PITCH || worldZ > MAX_Z || state.airTime > MAX_AIR_TIME) {
		state.alive = false;
	}
}

function scoreWalkerState(s: WalkerState): number {
	const upright = Math.max(0, 1 - Math.abs(s.torsoAngle) / 1.5);
	const distance = Math.max(0, s.maxX);
	const survival = s.aliveTime / EVAL_SECONDS;
	const fallPenalty = s.alive ? 0 : 0.35;
	const fit =
		2.2 * distance +
		0.45 * s.rewardAcc +
		1.2 * survival * upright +
		0.045 * s.contactSwitches +
		0.08 * s.motorDriveAcc -
		fallPenalty;
	return fit;
}

function createWalkerRig(scene: BABYLON.Scene, idx: number): WalkerRig {
	const root = new BABYLON.TransformNode(`sim7-root-${idx}`, scene);

	const bodyMat = new BABYLON.StandardMaterial(`sim7-body-mat-${idx}`, scene);
	bodyMat.diffuseColor = BODY_DIFFUSE_DEFAULT.clone();
	bodyMat.emissiveColor = BODY_EMISSIVE_DEFAULT.clone();
	bodyMat.specularColor = BABYLON.Color3.Black();
	const limbMat = new BABYLON.StandardMaterial(`sim7-limb-mat-${idx}`, scene);
	limbMat.diffuseColor = LIMB_DIFFUSE_DEFAULT.clone();
	limbMat.emissiveColor = LIMB_EMISSIVE_DEFAULT.clone();
	limbMat.specularColor = BABYLON.Color3.Black();

	const pelvis = new BABYLON.TransformNode(`sim7-pelvis-${idx}`, scene);
	pelvis.parent = root;
	pelvis.position.y = HIP_HEIGHT;

	const hipHalfWidth = 0.16;
	const shoulderHalfWidth = 0.25;
	const chestHeight = 0.38;

	const hipsBar = BABYLON.MeshBuilder.CreateCapsule(
		`sim7-hips-bar-${idx}`,
		{ radius: 0.04, height: hipHalfWidth * 2.2, tessellation: 10 },
		scene,
	);
	hipsBar.parent = pelvis;
	hipsBar.rotation.x = Math.PI / 2;
	hipsBar.material = bodyMat;

	const spineStick = BABYLON.MeshBuilder.CreateCapsule(
		`sim7-spine-stick-${idx}`,
		{ radius: 0.045, height: chestHeight, tessellation: 10 },
		scene,
	);
	spineStick.parent = pelvis;
	spineStick.position.y = chestHeight * 0.5;
	spineStick.material = bodyMat;

	const chest = new BABYLON.TransformNode(`sim7-chest-${idx}`, scene);
	chest.parent = pelvis;
	chest.position.y = chestHeight;

	const chestStick = BABYLON.MeshBuilder.CreateCapsule(
		`sim7-chest-stick-${idx}`,
		{ radius: 0.055, height: 0.34, tessellation: 10 },
		scene,
	);
	chestStick.parent = chest;
	chestStick.position.y = 0.18;
	chestStick.material = bodyMat;

	const shoulderBar = BABYLON.MeshBuilder.CreateCapsule(
		`sim7-shoulder-bar-${idx}`,
		{ radius: 0.035, height: shoulderHalfWidth * 2.2, tessellation: 10 },
		scene,
	);
	shoulderBar.parent = chest;
	shoulderBar.position.y = 0.36;
	shoulderBar.rotation.x = Math.PI / 2;
	shoulderBar.material = bodyMat;

	const neckStick = BABYLON.MeshBuilder.CreateCapsule(`sim7-neck-stick-${idx}`, { radius: 0.03, height: 0.14, tessellation: 10 }, scene);
	neckStick.parent = chest;
	neckStick.position.y = 0.52;
	neckStick.material = bodyMat;

	const head = BABYLON.MeshBuilder.CreateSphere(`sim7-head-${idx}`, { diameter: 0.28, segments: 16 }, scene);
	head.parent = chest;
	head.position.y = 0.69;
	head.material = bodyMat;

	const leftHip = new BABYLON.TransformNode(`sim7-left-hip-${idx}`, scene);
	leftHip.parent = pelvis;
	leftHip.position.set(0, 0, hipHalfWidth);

	const rightHip = new BABYLON.TransformNode(`sim7-right-hip-${idx}`, scene);
	rightHip.parent = pelvis;
	rightHip.position.set(0, 0, -hipHalfWidth);

	const leftThigh = BABYLON.MeshBuilder.CreateCapsule(`sim7-left-thigh-${idx}`, { radius: 0.06, height: THIGH_LEN, tessellation: 10 }, scene);
	leftThigh.parent = leftHip;
	leftThigh.position.y = -THIGH_LEN * 0.5;
	leftThigh.material = limbMat;

	const rightThigh = BABYLON.MeshBuilder.CreateCapsule(`sim7-right-thigh-${idx}`, { radius: 0.06, height: THIGH_LEN, tessellation: 10 }, scene);
	rightThigh.parent = rightHip;
	rightThigh.position.y = -THIGH_LEN * 0.5;
	rightThigh.material = limbMat;

	const leftKnee = new BABYLON.TransformNode(`sim7-left-knee-${idx}`, scene);
	leftKnee.parent = leftHip;
	leftKnee.position.y = -THIGH_LEN;

	const rightKnee = new BABYLON.TransformNode(`sim7-right-knee-${idx}`, scene);
	rightKnee.parent = rightHip;
	rightKnee.position.y = -THIGH_LEN;

	const leftShin = BABYLON.MeshBuilder.CreateCapsule(`sim7-left-shin-${idx}`, { radius: 0.052, height: SHIN_LEN, tessellation: 10 }, scene);
	leftShin.parent = leftKnee;
	leftShin.position.y = -SHIN_LEN * 0.5;
	leftShin.material = limbMat;

	const rightShin = BABYLON.MeshBuilder.CreateCapsule(`sim7-right-shin-${idx}`, { radius: 0.052, height: SHIN_LEN, tessellation: 10 }, scene);
	rightShin.parent = rightKnee;
	rightShin.position.y = -SHIN_LEN * 0.5;
	rightShin.material = limbMat;

	const leftAnkle = new BABYLON.TransformNode(`sim7-left-ankle-${idx}`, scene);
	leftAnkle.parent = leftKnee;
	leftAnkle.position.y = -SHIN_LEN;

	const rightAnkle = new BABYLON.TransformNode(`sim7-right-ankle-${idx}`, scene);
	rightAnkle.parent = rightKnee;
	rightAnkle.position.y = -SHIN_LEN;

	const leftFoot = BABYLON.MeshBuilder.CreateCapsule(`sim7-left-foot-${idx}`, { radius: 0.032, height: 0.26, tessellation: 10 }, scene);
	leftFoot.parent = leftAnkle;
	leftFoot.position.set(0.1, -0.035, 0);
	leftFoot.rotation.z = Math.PI / 2;
	leftFoot.material = limbMat;

	const rightFoot = BABYLON.MeshBuilder.CreateCapsule(`sim7-right-foot-${idx}`, { radius: 0.032, height: 0.26, tessellation: 10 }, scene);
	rightFoot.parent = rightAnkle;
	rightFoot.position.set(0.1, -0.035, 0);
	rightFoot.rotation.z = Math.PI / 2;
	rightFoot.material = limbMat;

	const leftToe = BABYLON.MeshBuilder.CreateSphere(`sim7-left-toe-${idx}`, { diameter: 0.055, segments: 10 }, scene);
	leftToe.parent = leftAnkle;
	leftToe.position.set(0.23, -0.035, 0);
	leftToe.material = limbMat;

	const rightToe = BABYLON.MeshBuilder.CreateSphere(`sim7-right-toe-${idx}`, { diameter: 0.055, segments: 10 }, scene);
	rightToe.parent = rightAnkle;
	rightToe.position.set(0.23, -0.035, 0);
	rightToe.material = limbMat;

	const leftShoulder = new BABYLON.TransformNode(`sim7-left-shoulder-${idx}`, scene);
	leftShoulder.parent = chest;
	leftShoulder.position.set(0, 0.36, shoulderHalfWidth);

	const rightShoulder = new BABYLON.TransformNode(`sim7-right-shoulder-${idx}`, scene);
	rightShoulder.parent = chest;
	rightShoulder.position.set(0, 0.36, -shoulderHalfWidth);

	const upperArmLen = 0.28;
	const forearmLen = 0.26;

	const leftUpperArm = BABYLON.MeshBuilder.CreateCapsule(`sim7-left-upper-arm-${idx}`, { radius: 0.038, height: upperArmLen, tessellation: 10 }, scene);
	leftUpperArm.parent = leftShoulder;
	leftUpperArm.position.y = -upperArmLen * 0.5;
	leftUpperArm.material = limbMat;

	const rightUpperArm = BABYLON.MeshBuilder.CreateCapsule(`sim7-right-upper-arm-${idx}`, { radius: 0.038, height: upperArmLen, tessellation: 10 }, scene);
	rightUpperArm.parent = rightShoulder;
	rightUpperArm.position.y = -upperArmLen * 0.5;
	rightUpperArm.material = limbMat;

	const leftElbow = new BABYLON.TransformNode(`sim7-left-elbow-${idx}`, scene);
	leftElbow.parent = leftShoulder;
	leftElbow.position.y = -upperArmLen;

	const rightElbow = new BABYLON.TransformNode(`sim7-right-elbow-${idx}`, scene);
	rightElbow.parent = rightShoulder;
	rightElbow.position.y = -upperArmLen;

	const leftForearm = BABYLON.MeshBuilder.CreateCapsule(`sim7-left-forearm-${idx}`, { radius: 0.032, height: forearmLen, tessellation: 10 }, scene);
	leftForearm.parent = leftElbow;
	leftForearm.position.y = -forearmLen * 0.5;
	leftForearm.material = limbMat;

	const rightForearm = BABYLON.MeshBuilder.CreateCapsule(`sim7-right-forearm-${idx}`, { radius: 0.032, height: forearmLen, tessellation: 10 }, scene);
	rightForearm.parent = rightElbow;
	rightForearm.position.y = -forearmLen * 0.5;
	rightForearm.material = limbMat;

	return {
		root,
		bodyMat,
		limbMat,
		pelvis,
		chest,
		hipsBar,
		spineStick,
		chestStick,
		neckStick,
		shoulderBar,
		head,
		leftHip,
		rightHip,
		leftKnee,
		rightKnee,
		leftAnkle,
		rightAnkle,
		leftShoulder,
		rightShoulder,
		leftElbow,
		rightElbow,
		leftThigh,
		rightThigh,
		leftShin,
		rightShin,
		leftFoot,
		rightFoot,
		leftToe,
		rightToe,
		leftUpperArm,
		rightUpperArm,
		leftForearm,
		rightForearm,
	};
}

function poseWalkerRig(rig: WalkerRig, s: WalkerState, laneZ: number): void {
	const a = s.jointAngles;
	const torsoPitch = clamp(s.torsoAngle, -0.95, 0.95);

	// World: X = forward, Y = up, Z = lane/left-right.
	rig.root.position.x = s.x;
	rig.root.position.y = s.y;
	rig.root.position.z = laneZ;
	rig.root.rotation.set(0, 0, 0);

	rig.chest.rotation.set(0, 0, torsoPitch);
	rig.pelvis.rotation.set(0, 0, clamp(torsoPitch * 0.25, -0.25, 0.25));

	if (!s.alive) {
		rig.root.rotation.set(0.22, 0, 0);
		rig.pelvis.rotation.set(0.35, 0, clamp(torsoPitch * 0.45 + 0.5, -0.25, 1.2));
		rig.chest.rotation.set(0.95, 0, clamp(torsoPitch * 0.5 + 0.85, -0.2, 1.4));
		setJointRotation(rig.leftHip, 0, 0, 0.35, HUMAN_LIMITS.hip);
		setJointRotation(rig.rightHip, 0, 0, 0.35, HUMAN_LIMITS.hip);
		setJointRotation(rig.leftKnee, 0, 0, -1.2, HUMAN_LIMITS.knee);
		setJointRotation(rig.rightKnee, 0, 0, -1.2, HUMAN_LIMITS.knee);
		setJointRotation(rig.leftAnkle, 0, 0, -0.15, HUMAN_LIMITS.ankle);
		setJointRotation(rig.rightAnkle, 0, 0, -0.15, HUMAN_LIMITS.ankle);
		setJointRotation(rig.leftShoulder, 0, 0, -0.7, HUMAN_LIMITS.shoulder);
		setJointRotation(rig.rightShoulder, 0, 0, 0.7, HUMAN_LIMITS.shoulder);
		setJointRotation(rig.leftElbow, 0, 0, -0.75, HUMAN_LIMITS.elbow);
		setJointRotation(rig.rightElbow, 0, 0, -0.75, HUMAN_LIMITS.elbow);
		return;
	}

	setJointRotation(rig.leftHip, 0, 0, a[0] - 0.04, HUMAN_LIMITS.hip);
	setJointRotation(rig.rightHip, 0, 0, a[3] + 0.04, HUMAN_LIMITS.hip);
	setJointRotation(rig.leftKnee, 0, 0, -a[1], HUMAN_LIMITS.knee);
	setJointRotation(rig.rightKnee, 0, 0, -a[4], HUMAN_LIMITS.knee);
	setJointRotation(rig.leftAnkle, 0, 0, a[2], HUMAN_LIMITS.ankle);
	setJointRotation(rig.rightAnkle, 0, 0, a[5], HUMAN_LIMITS.ankle);

	const sw = 0.32 * Math.sin(s.phase);
	setJointRotation(rig.leftShoulder, 0, 0, -sw - 0.08, HUMAN_LIMITS.shoulder);
	setJointRotation(rig.rightShoulder, 0, 0, sw + 0.08, HUMAN_LIMITS.shoulder);
	setJointRotation(rig.leftElbow, 0, 0, -0.35, HUMAN_LIMITS.elbow);
	setJointRotation(rig.rightElbow, 0, 0, -0.35, HUMAN_LIMITS.elbow);

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
				<circle key={`in-${i}`} cx={inputX} cy={y} r={actRadius(3.8, snap.inputs.activations[i])} fill={actColor(snap.inputs.activations[i], "#94a3b8")} opacity={actAlpha(snap.inputs.activations[i])} />
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

export default function Sim7HumanWalk() {
	const canvasRef = useRef<HTMLCanvasElement | null>(null);
	const runningRef = useRef(true);
	const followCamRef = useRef(true);
	const camDistanceRef = useRef(9.5);
	const generationElapsedRef = useRef(0);
	const generationRef = useRef(1);

	const populationRef = useRef<Genome[]>(Array.from({ length: POP_SIZE }, () => randomGenome()));
	const netsRef = useRef<ReturnType<typeof decodeNet>[]>(populationRef.current.map((g) => decodeNet(g.weights)));
	const statesRef = useRef<WalkerState[]>(Array.from({ length: POP_SIZE }, () => freshWalker()));
	const bestIndexRef = useRef(0);
	const selectedWalkerRef = useRef(0);

	const [running, setRunning] = useState(true);
	const [controlsCollapsed, setControlsCollapsed] = useState(false);
	const [followCam, setFollowCam] = useState(true);
	const [camDistance, setCamDistance] = useState(9.5);
	const [selectedWalker, setSelectedWalker] = useState(0);
	const [generation, setGeneration] = useState(1);
	const [bestFitness, setBestFitness] = useState(0);
	const [bestDistance, setBestDistance] = useState(0);
	const [ui, setUi] = useState<UiSnapshot>({ aliveSecs: 0, speed: 0, status: "Training" });
	const [neural, setNeural] = useState<NeuralNetSnapshot | null>(null);

	useEffect(() => {
		generationRef.current = generation;
	}, [generation]);

	useEffect(() => {
		runningRef.current = running;
	}, [running]);

	useEffect(() => {
		followCamRef.current = followCam;
	}, [followCam]);

	useEffect(() => {
		camDistanceRef.current = camDistance;
	}, [camDistance]);

	useEffect(() => {
		if (!canvasRef.current) return;

		const engine = new BABYLON.Engine(canvasRef.current, true, { preserveDrawingBuffer: true, stencil: true });
		const scene = new BABYLON.Scene(engine);
		scene.clearColor = BABYLON.Color4.FromHexString("#020617FF");

		const camera = new BABYLON.ArcRotateCamera("sim7-cam", -Math.PI / 2.6, 1.1, 9.5, new BABYLON.Vector3(0, 1.1, 0), scene);
		camera.attachControl(canvasRef.current, true);
		camera.lowerRadiusLimit = 5.5;
		camera.upperRadiusLimit = 120;

		const hemi = new BABYLON.HemisphericLight("sim7-hemi", new BABYLON.Vector3(0, 1, 0), scene);
		hemi.intensity = 0.95;
		hemi.specular = BABYLON.Color3.Black();
		const sun = new BABYLON.DirectionalLight("sim7-sun", new BABYLON.Vector3(-0.6, -1, -0.3), scene);
		sun.position = new BABYLON.Vector3(18, 20, 10);
		sun.intensity = 0.9;
		sun.specular = BABYLON.Color3.Black();

		const ground = BABYLON.MeshBuilder.CreateGround("sim7-ground", { width: 420, height: 140 }, scene);
		const gMat = new BABYLON.StandardMaterial("sim7-ground-mat", scene);
		gMat.diffuseColor = BABYLON.Color3.FromHexString("#0b2a22");
		gMat.emissiveColor = BABYLON.Color3.FromHexString("#052018");
		gMat.specularColor = BABYLON.Color3.Black();
		ground.material = gMat;

		for (let i = -70; i <= 70; i += 1) {
			const x = i * 3;
			const line = BABYLON.MeshBuilder.CreateLines(
				`sim7-marker-${i}`,
				{ points: [new BABYLON.Vector3(x, 0.012, -69), new BABYLON.Vector3(x, 0.012, 69)] },
				scene,
			);
			line.color = BABYLON.Color3.FromHexString(i % 5 === 0 ? "#334155" : "#1e293b");
		}

		const laneGap = 2.4;
		const laneZs = Array.from({ length: POP_SIZE }, (_, i) => (i - (POP_SIZE - 1) / 2) * laneGap);
		const rigs = Array.from({ length: POP_SIZE }, (_, i) => createWalkerRig(scene, i));

		for (let i = 0; i < POP_SIZE; i += 1) {
			const selectWalker = () => {
				selectedWalkerRef.current = i;
				setSelectedWalker(i);
			};
			const pickMeshes: BABYLON.AbstractMesh[] = [
				rigs[i].head,
				rigs[i].chestStick,
				rigs[i].hipsBar,
				rigs[i].leftThigh,
				rigs[i].rightThigh,
				rigs[i].leftShin,
				rigs[i].rightShin,
				rigs[i].leftFoot,
				rigs[i].rightFoot,
			];
			for (const mesh of pickMeshes) {
				mesh.isPickable = true;
				mesh.actionManager = mesh.actionManager ?? new BABYLON.ActionManager(scene);
				mesh.actionManager.registerAction(new BABYLON.ExecuteCodeAction(BABYLON.ActionManager.OnPickTrigger, selectWalker));
			}
		}

		const evolve = () => {
			const scored = populationRef.current
				.map((g, i) => {
					const f = scoreWalkerState(statesRef.current[i]);
					g.fitness = f;
					statesRef.current[i].fit = f;
					return { g, i, fit: f };
				})
				.sort((a, b) => b.fit - a.fit);

			const best = scored[0];
			bestIndexRef.current = best?.i ?? 0;
			setBestFitness(best?.fit ?? 0);
			setBestDistance(statesRef.current[bestIndexRef.current]?.maxX ?? 0);

			const elites = scored.slice(0, ELITE).map((x) => x.g);
			const next: Genome[] = elites.map((e) => cloneGenome(e));

			const tournamentPick = () => {
				let bestCand: Genome | null = null;
				for (let i = 0; i < TOURN_K; i += 1) {
					const c = populationRef.current[Math.floor(Math.random() * POP_SIZE)];
					if (!bestCand || c.fitness > bestCand.fitness) bestCand = c;
				}
				return bestCand ?? populationRef.current[0];
			};

			while (next.length < POP_SIZE) {
				const pa = tournamentPick();
				const pb = tournamentPick();
				next.push(crossoverAndMutate(pa, pb));
			}

			populationRef.current = next;
			netsRef.current = populationRef.current.map((g) => decodeNet(g.weights));
			statesRef.current = Array.from({ length: POP_SIZE }, () => freshWalker());
			generationElapsedRef.current = 0;
			generationRef.current += 1;
			setGeneration(generationRef.current);
		};

		let lastUiAt = 0;
		let prevMs = performance.now();

		engine.runRenderLoop(() => {
			const now = performance.now();
			const dt = Math.min(0.05, (now - prevMs) / 1000);
			prevMs = now;
			const selectedIdx = clamp(Math.round(selectedWalkerRef.current), 0, POP_SIZE - 1);

			if (runningRef.current) {
				for (let i = 0; i < POP_SIZE; i += 1) stepWalker(statesRef.current[i], netsRef.current[i], dt);
				generationElapsedRef.current += dt;
				const allDead = statesRef.current.every((s) => !s.alive);
				if (generationElapsedRef.current >= EVAL_SECONDS || (allDead && generationElapsedRef.current >= MIN_RESET_SECONDS)) evolve();
			}

			for (let i = 0; i < POP_SIZE; i += 1) {
				poseWalkerRig(rigs[i], statesRef.current[i], laneZs[i]);
				const isSelected = i === selectedIdx;
				rigs[i].bodyMat.diffuseColor.copyFrom(isSelected ? BODY_DIFFUSE_SELECTED : BODY_DIFFUSE_DEFAULT);
				rigs[i].bodyMat.emissiveColor.copyFrom(isSelected ? BODY_EMISSIVE_SELECTED : BODY_EMISSIVE_DEFAULT);
				rigs[i].limbMat.diffuseColor.copyFrom(isSelected ? LIMB_DIFFUSE_SELECTED : LIMB_DIFFUSE_DEFAULT);
				rigs[i].limbMat.emissiveColor.copyFrom(isSelected ? LIMB_EMISSIVE_SELECTED : LIMB_EMISSIVE_DEFAULT);
				rigs[i].head.scaling.setAll(isSelected ? 1.25 : 1);
				rigs[i].leftToe.scaling.setAll(isSelected ? 1.18 : 1);
				rigs[i].rightToe.scaling.setAll(isSelected ? 1.18 : 1);
			}

			const s = statesRef.current[selectedIdx] ?? statesRef.current[0] ?? freshWalker();
			if (followCamRef.current) {
				camera.target.x = BABYLON.Scalar.Lerp(camera.target.x, s.x + 2.5, 0.08);
				camera.target.y = BABYLON.Scalar.Lerp(camera.target.y, 1.0, 0.08);
				camera.target.z = BABYLON.Scalar.Lerp(camera.target.z, 0, 0.08);
				camera.radius = BABYLON.Scalar.Lerp(camera.radius, Math.max(25, camDistanceRef.current * 3.2), 0.08);
			}

			if (now - lastUiAt > 120) {
				lastUiAt = now;
				const aliveCount = statesRef.current.filter((x) => x.alive).length;
				const selectedState = statesRef.current[selectedIdx] ?? statesRef.current[0];
				const selectedNet = netsRef.current[selectedIdx] ?? netsRef.current[0];
				if (selectedState && selectedNet) {
					const obs = buildObservation(selectedState);
					const trace = forwardTrace(selectedNet, obs);
					const sampledInput = sampleLayer(obs, 12);
					const sampledH1 = sampleLayer(trace.h1, 10);
					const sampledH2 = sampleLayer(trace.h2, 8);
					const outIdx = Array.from({ length: N_OUT }, (_, i) => i);
					setNeural({
						selectedIndex: selectedIdx,
						fitness: selectedState.fit,
						weightAbsMean: {
							w1: meanAbs(selectedNet.w1),
							w2: meanAbs(selectedNet.w2),
							w3: meanAbs(selectedNet.w3),
						},
						inputs: sampledInput,
						hidden1: sampledH1,
						hidden2: sampledH2,
						outputs: ACTION_OUTPUT_NAMES.map((name, i) => ({ index: i, name, value: trace.out[i] ?? 0 })),
						w1: sampleWeightsW1(selectedNet, sampledInput.indices, sampledH1.indices),
						w2: sampleWeightsW2(selectedNet, sampledH1.indices, sampledH2.indices),
						w3: sampleWeightsW3(selectedNet, sampledH2.indices, outIdx),
					});
				}
				setUi({
					aliveSecs: s.aliveTime,
					speed: s.vx,
					status: !runningRef.current ? "Paused" : `Training G${generationRef.current} · alive ${aliveCount}/${POP_SIZE}`,
				});
			}
			scene.render();
		});

		const onResize = () => engine.resize();
		window.addEventListener("resize", onResize);

		return () => {
			window.removeEventListener("resize", onResize);
			engine.dispose();
		};
	}, []);

	const handleReset = () => {
		populationRef.current = Array.from({ length: POP_SIZE }, () => randomGenome());
		netsRef.current = populationRef.current.map((g) => decodeNet(g.weights));
		statesRef.current = Array.from({ length: POP_SIZE }, () => freshWalker());
		bestIndexRef.current = 0;
		selectedWalkerRef.current = 0;
		setSelectedWalker(0);
		generationElapsedRef.current = 0;
		generationRef.current = 1;
		setGeneration(1);
		setBestFitness(0);
		setBestDistance(0);
		setUi({ aliveSecs: 0, speed: 0, status: runningRef.current ? "Training" : "Paused" });
	};

	return (
		<>
			<canvas ref={canvasRef} className="fixed inset-0 z-0 block h-screen w-screen" style={{ width: "100vw", height: "100vh" }} />

			<aside className="selected-panel">
				<Card className="bg-transparent border-0 shadow-none">
					<div className="controls-title" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
						<span>Sim 7 Overview</span>
						<span className="text-xs text-slate-300">3D</span>
					</div>
					<CardContent className="space-y-2">
						<div className="rounded-xl border border-slate-700 p-2 text-xs text-left space-y-2">
							<div className="grid grid-cols-2 gap-2">
								<div className="rounded-md border border-slate-700 bg-slate-950/50 p-2">
									<div className="text-slate-400">Generation</div>
									<div className="font-mono text-slate-100 mt-1">{generation}</div>
								</div>
								<div className="rounded-md border border-slate-700 bg-slate-950/50 p-2">
									<div className="text-slate-400">Status</div>
									<div className="font-mono text-slate-100 mt-1">{ui.status}</div>
								</div>
							</div>
							<table className="w-full table-fixed border-collapse text-left">
								<tbody>
									<tr><th className="w-[42%] pr-2 py-[2px] text-slate-400 font-normal">Selected Walker</th><td className="py-[2px] text-slate-100 font-mono">#{selectedWalker + 1}</td></tr>
									<tr><th className="w-[42%] pr-2 py-[2px] text-slate-400 font-normal">Best Fitness</th><td className="py-[2px] text-slate-100 font-mono">{bestFitness.toFixed(2)}</td></tr>
									<tr><th className="w-[42%] pr-2 py-[2px] text-slate-400 font-normal">Best Distance</th><td className="py-[2px] text-slate-100 font-mono">{bestDistance.toFixed(2)} m</td></tr>
									<tr><th className="w-[42%] pr-2 py-[2px] text-slate-400 font-normal">Alive Time</th><td className="py-[2px] text-slate-100 font-mono">{ui.aliveSecs.toFixed(1)} s</td></tr>
									<tr><th className="w-[42%] pr-2 py-[2px] text-slate-400 font-normal">Current Speed</th><td className="py-[2px] text-slate-100 font-mono">{ui.speed.toFixed(2)} m/s</td></tr>
								</tbody>
							</table>
							{neural && (
								<div className="rounded-md border border-slate-700 bg-slate-950/50 p-2 mt-2 space-y-2">
									<div className="flex items-center justify-between">
										<div className="text-slate-300">Selected Neural Net</div>
										<div className="font-mono text-slate-200">Fit {neural.fitness.toFixed(2)}</div>
									</div>
									{/* <div className="grid grid-cols-3 gap-2 text-[11px]">
										<div className="rounded border border-slate-700 p-1"><span className="text-slate-400">|W1|</span> <span className="font-mono">{neural.weightAbsMean.w1.toFixed(3)}</span></div>
										<div className="rounded border border-slate-700 p-1"><span className="text-slate-400">|W2|</span> <span className="font-mono">{neural.weightAbsMean.w2.toFixed(3)}</span></div>
										<div className="rounded border border-slate-700 p-1"><span className="text-slate-400">|W3|</span> <span className="font-mono">{neural.weightAbsMean.w3.toFixed(3)}</span></div>
									</div> */}
									<DenseNeuralNetGraph snap={neural} />
								</div>
							)}
						</div>
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
								<div className="font-semibold">Walk Training</div>
								<div className="grid grid-cols-2 gap-2">
									<Button onClick={() => setRunning((r) => !r)}>{running ? "Pause" : "Resume"}</Button>
									<Button variant="secondary" onClick={handleReset}>New Population</Button>
								</div>
							</div>

							<div className="rounded-xl border border-slate-700 p-3 text-sm space-y-3">
								<div className="font-semibold">Camera</div>
								<div className="flex items-center justify-between text-sm">
									<span>Follow walker</span>
									<input type="checkbox" checked={followCam} onChange={(e) => setFollowCam(e.target.checked)} />
								</div>
								<div>
									<div className="flex justify-between text-sm mb-1"><span>Camera distance</span><span>{camDistance.toFixed(1)}</span></div>
									<Slider value={[camDistance]} min={5.5} max={18} step={0.1} onValueChange={(v: number[]) => setCamDistance(v[0])} />
								</div>
							</div>
						</CardContent>
					)}
				</Card>
			</aside>
		</>
	);
}
