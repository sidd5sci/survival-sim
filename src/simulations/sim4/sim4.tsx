import { useEffect, useRef, useState } from "react";
import * as BABYLON from "babylonjs";
import { Card, CardContent } from "../../components/ui/card";
import { Button } from "../../components/ui/button";
import { Slider } from "../../components/ui/slider";
import { ensureLmStudioModelReady, lmStudioChatText, normalizeLmStudioBaseUrl } from "../../services/lmstudio";
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

type AiProvider = "local" | "replicate" | "none";
type AiModel = "gpt-5.2" | "gemini-3" | "gemma-4";

type Sim4Settings = {
	aiProvider: AiProvider;
	aiModel: AiModel;
	localEndpointUrl: string;
	localModelName: string;
	replicateApiToken: string;
	replicateModel: string;
	visionResolution: number;
	aiHz: number;
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

const WORLD_SIZE = 120;
const HALF = WORLD_SIZE / 2;
const BOT_RADIUS = 0.6;
const ENEMY_RADIUS = 0.6;
const ROCK_RADIUS = 0.28;

const EXIT_X = HALF - 6;
const EXIT_Z = HALF - 6;
const EXIT_RADIUS = 3.0;

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
};

function round3(v: number): number {
	return Math.round(v * 1000) / 1000;
}

function extractJsonObject(text: string): any {
	// Try fenced ```json blocks first
	const fenced = text.match(/```json\s*([\s\S]*?)\s*```/i);
	if (fenced?.[1]) {
		try {
			return JSON.parse(fenced[1]);
		} catch {
			// fallthrough
		}
	}

	// Then try first {...last} heuristic
	const first = text.indexOf("{");
	const last = text.lastIndexOf("}");
	if (first >= 0 && last > first) {
		const slice = text.slice(first, last + 1);
		return JSON.parse(slice);
	}

	// Lastly, attempt direct parse
	return JSON.parse(text);
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
	};
}

function defaultFallbackAction(bot: Bot): AiAction {
	// Simple fallback: move toward the exit.
	const toExit = normalize(EXIT_X - bot.x, EXIT_Z - bot.z);
	const heading = normalize(bot.headingX, bot.headingZ);
	const right = { x: heading.z, z: -heading.x };
	const forward = clamp(toExit.x * heading.x + toExit.z * heading.z, -1, 1);
	const sideways = clamp(toExit.x * right.x + toExit.z * right.z, -1, 1);
	return {
		left: sideways < 0 ? Math.min(1, -sideways) : 0,
		right: sideways > 0 ? Math.min(1, sideways) : 0,
		forward: Math.max(0, forward),
		backward: Math.max(0, -forward),
		speed: 0.8,
		pick: false,
		throw: false,
		grow: bot.energy < bot.maxEnergy * 0.25,
	};
}
async function callLocalAi(lmStudioBaseUrl: string, model: string, compactState: any, visionDataUrl?: string | null): Promise<AiAction> {
	const modelKey = model.trim();
	if (!modelKey) throw new Error("Missing model key");

	const systemPrompt =
		"Return ONLY one minified JSON object. No reasoning. No prose. No markdown.\n" +
		"Keys: left,right,forward,backward in [-1..1], speed in [0..1], optional booleans pick,throw,grow.\n" +
		"Actions: pick picks nearest rock if close and hands free; throw throws held rock forward; grow increases capacity if enough energy.\n" +
		"Goal: survive, avoid enemies, reach the green exit ring.\n" +
		"Example: {\"left\":0,\"right\":0,\"forward\":1,\"backward\":0,\"speed\":0.8}";

	const userText = JSON.stringify(compactState);
	const content = await lmStudioChatText({
		lmStudioBaseUrl,
		model: modelKey,
		systemPrompt,
		userText,
		visionDataUrl: visionDataUrl ?? null,
		maxOutputTokens: 90,
		temperature: 0,
	});

	let actionObj: any = null;
	try {
		actionObj = extractJsonObject(String(content));
	} catch {
		actionObj = null;
	}
	return sanitizeAction(actionObj);
}

async function callReplicateLike(apiToken: string, model: string, payload: any): Promise<AiAction> {
	const json = await replicatePredict(apiToken, model, payload);
	const output = json?.output ?? json;
	return sanitizeAction(output);
}

function BabylonWorld({
	settings,
	paused,
	onUi,
	onVisionPreview,
}: {
	settings: Sim4Settings;
	paused: boolean;
	onUi: (snap: UiSnapshot) => void;
	onVisionPreview: (dataUrl: string) => void;
}) {
	const canvasRef = useRef<HTMLCanvasElement | null>(null);
	const settingsRef = useRef(settings);
	const pausedRef = useRef(paused);
	const onVisionPreviewRef = useRef(onVisionPreview);

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
		const sun = new BABYLON.DirectionalLight("sun", new BABYLON.Vector3(-0.7, -1, -0.3), scene);
		sun.position = new BABYLON.Vector3(50, 80, 10);
		sun.intensity = 1.05;

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

		// Exit marker
		const exit = BABYLON.MeshBuilder.CreateTorus("exit", { diameter: EXIT_RADIUS * 2.2, thickness: 0.22, tessellation: 48 }, scene);
		exit.position.set(EXIT_X, 0.12, EXIT_Z);
		exit.rotation.x = Math.PI / 2;
		exit.isPickable = false;
		const exitMat = new BABYLON.StandardMaterial("exitMat", scene);
		exitMat.emissiveColor = BABYLON.Color3.FromHexString("#22c55e");
		exitMat.diffuseColor = BABYLON.Color3.FromHexString("#15803d");
		exitMat.alpha = 0.8;
		exit.material = exitMat;

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

		// Bot rig (simple articulated limbs)
		const botRoot = new BABYLON.TransformNode("botRoot", scene);
		const torso = BABYLON.MeshBuilder.CreateBox("botTorso", { width: 1.4, depth: 0.9, height: 1.6 }, scene);
		torso.parent = botRoot;
		torso.position.y = 1.6;
		const head = BABYLON.MeshBuilder.CreateSphere("botHead", { diameter: 0.9, segments: 16 }, scene);
		head.parent = botRoot;
		head.position.y = 2.55;

		const limbMat = new BABYLON.StandardMaterial("botMat", scene);
		limbMat.diffuseColor = BABYLON.Color3.FromHexString("#e2e8f0");
		limbMat.emissiveColor = BABYLON.Color3.FromHexString("#0f172a");
		torso.material = limbMat;
		head.material = limbMat;

		const makeLimb = (name: string, length: number, radius: number) => {
			const pivot = new BABYLON.TransformNode(`${name}-pivot`, scene);
			pivot.parent = botRoot;
			const mesh = BABYLON.MeshBuilder.CreateCylinder(name, { height: length, diameter: radius * 2, tessellation: 10 }, scene);
			mesh.parent = pivot;
			mesh.position.y = -length / 2;
			mesh.material = limbMat;
			return { pivot, mesh };
		};

		const leftArm = makeLimb("leftArm", 1.25, 0.16);
		leftArm.pivot.position.set(-0.95, 2.15, 0);
		const rightArm = makeLimb("rightArm", 1.25, 0.16);
		rightArm.pivot.position.set(0.95, 2.15, 0);
		const leftLeg = makeLimb("leftLeg", 1.45, 0.18);
		leftLeg.pivot.position.set(-0.45, 1.05, 0);
		const rightLeg = makeLimb("rightLeg", 1.45, 0.18);
		rightLeg.pivot.position.set(0.45, 1.05, 0);

		// Hand attach point (for rocks)
		const rightHand = new BABYLON.TransformNode("rightHand", scene);
		rightHand.parent = rightArm.mesh;
		rightHand.position.set(0, -0.7, 0.15);

		// Eye camera + render target for vision
		const eyeCam = new BABYLON.FreeCamera("eyeCam", new BABYLON.Vector3(0, 2.55, 0.5), scene);
		eyeCam.parent = botRoot;
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
		const bot: Bot = {
			x: -HALF + 6,
			z: -HALF + 6,
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

		const enemies: Enemy[] = [];
		const enemyMeshes = new Map<string, BABYLON.Mesh>();
		const enemyMat = new BABYLON.StandardMaterial("enemyMat", scene);
		enemyMat.diffuseColor = BABYLON.Color3.FromHexString("#fb7185");
		enemyMat.emissiveColor = BABYLON.Color3.FromHexString("#450a0a");

		const rocks: Rock[] = [];
		const rockMeshes = new Map<string, BABYLON.Mesh>();
		const rockMat = new BABYLON.StandardMaterial("rockMat", scene);
		rockMat.diffuseColor = BABYLON.Color3.FromHexString("#94a3b8");

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
					mesh.parent = rightHand;
					mesh.position.set(0, 0, 0);
				} else {
					mesh.parent = null;
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

		const spawnEnemyAt = (x: number, z: number) => {
			enemies.push({ id: randomId("enemy"), x, z, vx: 0, vz: 0, alive: true });
		};

		// Click on ground to add enemies
		scene.onPointerObservable.add((eventInfo) => {
			if (eventInfo.type !== BABYLON.PointerEventTypes.POINTERPICK) return;
			const pick = eventInfo.pickInfo;
			if (!pick?.hit || !pick.pickedPoint) return;
			if (pick.pickedMesh?.name !== "ground") return;
			const p = pick.pickedPoint;
			spawnEnemyAt(clamp(p.x, -HALF + 2, HALF - 2), clamp(p.z, -HALF + 2, HALF - 2));
		});

		// Vision rendering should include relevant meshes
		visionTarget.renderList = [ground, ...Array.from(buildingMeshes.values()), torso, head];
		// Enemies/rocks are added dynamically via sync functions; we will refresh renderList periodically.
		let lastRenderListRefresh = 0;

		let lastAiAt = 0;
		let pendingAi = false;
		let currentAction: AiAction = defaultFallbackAction(bot);

		const captureVision = async (res: number) => {
			// Ensure target size matches settings.
			if (visionTarget.getSize().width !== res) {
				visionTarget.resize({ width: res, height: res });
			}
			visionTarget.render(true);
			const pixels = (await visionTarget.readPixels()) as Uint8Array | null;
			if (!pixels) return null;

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
			if (!ctx) return { dataUrl: "" };
			const img = new ImageData(flipped, res, res);
			ctx.putImageData(img, 0, 0);
			const dataUrl = offscreen.toDataURL("image/png");
			return { dataUrl };
		};

		const computeCompactState = () => {
			const basis = computeBotHeadingBasis();
			const toExitX = EXIT_X - bot.x;
			const toExitZ = EXIT_Z - bot.z;
			const exitDist = Math.sqrt(toExitX * toExitX + toExitZ * toExitZ);
			const toExitDir = normalize(toExitX, toExitZ);
			const exitForward = toExitDir.x * basis.h.x + toExitDir.z * basis.h.z;
			const exitRight = toExitDir.x * basis.r.x + toExitDir.z * basis.r.z;

			let nearestEnemyDist = Infinity;
			let nearestEnemyForward = 0;
			let nearestEnemyRight = 0;
			for (const e of enemies) {
				if (!e.alive) continue;
				const dx = e.x - bot.x;
				const dz = e.z - bot.z;
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
				const dx = r.x - bot.x;
				const dz = r.z - bot.z;
				const d = Math.sqrt(dx * dx + dz * dz);
				if (d < nearestRockDist) {
					nearestRockDist = d;
					const dir = normalize(dx, dz);
					nearestRockForward = dir.x * basis.h.x + dir.z * basis.h.z;
					nearestRockRight = dir.x * basis.r.x + dir.z * basis.r.z;
				}
			}

			const heldRock = rocks.some((r) => r.active && r.heldByBot);
			const botSpeed = Math.sqrt(bot.vx * bot.vx + bot.vz * bot.vz);
			const underThreat = Number.isFinite(nearestEnemyDist) && nearestEnemyDist < 4.2;

			return {
				health: round3(bot.health),
				energy: round3(bot.energy),
				organs: bot.organs,
				botSpeed: round3(botSpeed),
				exitDist: round3(exitDist),
				exitForward: round3(exitForward),
				exitRight: round3(exitRight),
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

		const computeBotHeadingBasis = () => {
			const h = normalize(bot.headingX, bot.headingZ);
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
			bot.x = -HALF + 6;
			bot.z = -HALF + 6;
			bot.vx = 0;
			bot.vz = 0;
			bot.headingX = 0;
			bot.headingZ = 1;
			bot.health = 100;
			bot.maxHealth = 100;
			bot.energy = 100;
			bot.maxEnergy = 120;
			bot.organs = 0;
			bot.escaped = false;
			enemies.length = 0;
			for (const e of enemyMeshes.values()) e.dispose();
			enemyMeshes.clear();
			for (const r of rocks) {
				r.x = rand(-HALF + 10, HALF - 10);
				r.z = rand(-HALF + 10, HALF - 10);
				r.vx = 0;
				r.vz = 0;
				r.heldByBot = false;
				r.active = true;
			}
			currentAction = defaultFallbackAction(bot);
		};

		// Expose reset via window for now (controls panel in React triggers via custom event)
		const resetHandler = () => resetWorld();
		window.addEventListener("sim4-reset", resetHandler as any);

		let lastUiAt = 0;
		let lastT = performance.now();

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
				visionTarget.renderList = [ground, ...Array.from(buildingMeshes.values()), torso, head, ...Array.from(enemyMeshes.values()), ...Array.from(rockMeshes.values())];
			}

			const s = settingsRef.current;
			const dead = bot.health <= 0;
			if (!pausedRef.current && !dead && !bot.escaped) {
				// AI tick
				const aiPeriodMs = 1000 / Math.max(0.2, s.aiHz);
				if (!pendingAi && now - lastAiAt >= aiPeriodMs) {
					lastAiAt = now;
					pendingAi = true;
					(void (async () => {
						try {
							const cap = await captureVision(Math.round(clamp(s.visionResolution, 16, 96)));
							if (cap?.dataUrl) onVisionPreviewRef.current(cap.dataUrl);
							const compactState = computeCompactState();

							let action: AiAction;
							if (s.aiProvider === "local" && s.localEndpointUrl.trim()) {
								if (!localBootstrap.ready) {
									action = defaultFallbackAction(bot);
								} else {
									action = await callLocalAi(
										s.localEndpointUrl.trim(),
										s.localModelName.trim() || "google/gemma-4-e2b",
									compactState,
									cap?.dataUrl ?? null,
								);
								}
							} else if (s.aiProvider === "replicate" && s.replicateApiToken.trim() && s.replicateModel.trim()) {
								action = await callReplicateLike(s.replicateApiToken.trim(), s.replicateModel.trim(), compactState);
							} else {
								action = defaultFallbackAction(bot);
							}
							currentAction = sanitizeAction(action);
						} catch {
							currentAction = defaultFallbackAction(bot);
						} finally {
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
							const cap = await captureVision(Math.round(clamp(settingsRef.current.visionResolution, 16, 96)));
							if (cap?.dataUrl) onVisionPreviewRef.current(cap.dataUrl);
						} finally {
							previewState.pending = false;
						}
					})());
				}

				// Bot movement
				const basis = computeBotHeadingBasis();
				const ax = (currentAction.right - currentAction.left) * s.botAccel;
				const az = (currentAction.forward - currentAction.backward) * s.botAccel;
				let steerX = basis.r.x * ax + basis.h.x * az;
				let steerZ = basis.r.z * ax + basis.h.z * az;
				const steerMag = Math.sqrt(steerX * steerX + steerZ * steerZ);
				if (steerMag > 1e-6) {
					steerX /= steerMag;
					steerZ /= steerMag;
				}
				const desiredMaxSpeed = s.botMaxSpeed * clamp(currentAction.speed, 0, 1);

				let vx = bot.vx;
				let vz = bot.vz;
				let x = bot.x;
				let z = bot.z;

				// Sub-step to prevent tunneling through buildings.
				const estSpeed = Math.sqrt(vx * vx + vz * vz);
				const maxStepDist = Math.max(BOT_RADIUS * 0.75, 0.14);
				const steps = clamp(Math.ceil((estSpeed * dt) / maxStepDist), 1, 18);
				const dtStep = dt / steps;
				const dragFactor = Math.pow(0.90, 1 / steps);

				for (let i = 0; i < steps; i += 1) {
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

				// Energy drain
				bot.energy = clamp(bot.energy - s.energyDrainPerSec * (0.3 + vmag) * dt, 0, bot.maxEnergy);
				if (bot.energy <= 0) bot.health = clamp(bot.health - 10 * dt, 0, bot.maxHealth);

				// Actions: grow organ, pick, throw
				if (currentAction.grow && bot.organs < 4 && bot.energy > bot.maxEnergy * 0.25) {
					bot.organs += 1;
					bot.maxEnergy += 18;
					bot.maxHealth += 12;
					bot.energy = clamp(bot.energy - 12, 0, bot.maxEnergy);
					const organ = BABYLON.MeshBuilder.CreateSphere(`organ-${bot.organs}`, { diameter: 0.35, segments: 10 }, scene);
					organ.parent = torso;
					organ.position.set(rand(-0.6, 0.6), rand(-0.5, 0.5), rand(-0.25, 0.25));
					organ.material = limbMat;
					organ.isPickable = false;
				}

				if (currentAction.pick) {
					const held = rocks.some((r) => r.heldByBot && r.active);
					if (!held) {
						let best: Rock | null = null;
						let bestD = Number.POSITIVE_INFINITY;
						for (const r of rocks) {
							if (!r.active || r.heldByBot) continue;
							const d = dist2({ x: bot.x, z: bot.z }, { x: r.x, z: r.z });
							if (d < bestD) {
								bestD = d;
								best = r;
							}
						}
						if (best && bestD < 2.2 * 2.2) {
							best.heldByBot = true;
							best.vx = 0;
							best.vz = 0;
						}
					}
				}

				if (currentAction.throw) {
					const heldRock = rocks.find((r) => r.active && r.heldByBot);
					if (heldRock) {
						heldRock.heldByBot = false;
						heldRock.x = bot.x + bot.headingX * 1.2;
						heldRock.z = bot.z + bot.headingZ * 1.2;
						heldRock.vx = bot.headingX * 16;
						heldRock.vz = bot.headingZ * 16;
						bot.energy = clamp(bot.energy - 3.5, 0, bot.maxEnergy);
					}
				}

				// Enemies chase and damage on collision
				for (const e of enemies) {
					if (!e.alive) continue;
					const toBot = normalize(bot.x - e.x, bot.z - e.z);
					let ex = e.x;
					let ez = e.z;
					let evx = (e.vx + toBot.x * s.enemySpeed * 2.2 * dt) * 0.86;
					let evz = (e.vz + toBot.z * s.enemySpeed * 2.2 * dt) * 0.86;
					const esp = Math.sqrt(evx * evx + evz * evz) || 1;
					if (esp > s.enemySpeed) {
						evx = (evx / esp) * s.enemySpeed;
						evz = (evz / esp) * s.enemySpeed;
					}

					// Sub-step enemy vs buildings
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

					const hit = dist2({ x: bot.x, z: bot.z }, { x: e.x, z: e.z }) < (BOT_RADIUS + ENEMY_RADIUS) ** 2;
					if (hit) {
						bot.health = clamp(bot.health - s.enemyDamagePerSec * dt, 0, bot.maxHealth);
					}
				}

				// Rocks move + collide with enemies
				for (const r of rocks) {
					if (!r.active || r.heldByBot) continue;
					r.x = clamp(r.x + r.vx * dt, -HALF + ROCK_RADIUS, HALF - ROCK_RADIUS);
					r.z = clamp(r.z + r.vz * dt, -HALF + ROCK_RADIUS, HALF - ROCK_RADIUS);
					r.vx *= 0.93;
					r.vz *= 0.93;
					// Stop if slow
					if (Math.abs(r.vx) + Math.abs(r.vz) < 0.05) {
						r.vx = 0;
						r.vz = 0;
					}

					for (const e of enemies) {
						if (!e.alive) continue;
						if (dist2({ x: r.x, z: r.z }, { x: e.x, z: e.z }) < (ROCK_RADIUS + ENEMY_RADIUS) ** 2) {
							e.alive = false;
							r.active = false;
							bot.energy = clamp(bot.energy + 6, 0, bot.maxEnergy);
						}
					}
				}

				// Escape check
				if (dist2({ x: bot.x, z: bot.z }, { x: EXIT_X, z: EXIT_Z }) < EXIT_RADIUS * EXIT_RADIUS) {
					bot.escaped = true;
				}
			}

			// Update bot transforms + animation
			botRoot.position.set(bot.x, 0, bot.z);
			botRoot.rotation.y = Math.atan2(bot.headingX, bot.headingZ);
			const moveSpeed = Math.sqrt(bot.vx * bot.vx + bot.vz * bot.vz);
			const t = now * 0.005;
			const swing = Math.min(0.75, moveSpeed / 3.5);
			leftLeg.pivot.rotation.x = Math.sin(t) * 0.8 * swing;
			rightLeg.pivot.rotation.x = -Math.sin(t) * 0.8 * swing;
			leftArm.pivot.rotation.x = -Math.sin(t) * 0.6 * swing;
			rightArm.pivot.rotation.x = Math.sin(t) * 0.6 * swing;

			syncEnemies();
			syncRocks();

			// UI snapshot
			if (now - lastUiAt > 160) {
				lastUiAt = now;
				const aliveEnemies = enemies.filter((e) => e.alive).length;
				const activeRocks = rocks.filter((r) => r.active).length;
				const localHint = settingsRef.current.aiProvider === "local" && !localBootstrap.ready ? ` (${localBootstrap.status})` : "";
				const status = bot.escaped
					? "Escaped!"
					: bot.health <= 0
						? "Dead"
						: pausedRef.current
							? "Paused"
							: settingsRef.current.aiProvider === "none"
								? "No AI configured (fallback running)"
								: `Running${localHint}`;

				onUi({
					health: bot.health,
					energy: bot.energy,
					enemiesAlive: aliveEnemies,
					rocksActive: activeRocks,
					organs: bot.organs,
					escaped: bot.escaped,
					dead: bot.health <= 0,
					status,
				});
			}

			scene.render();
		});

		const onResize = () => engine.resize();
		window.addEventListener("resize", onResize);

		return () => {
			window.removeEventListener("resize", onResize);
			window.removeEventListener("sim4-reset", resetHandler as any);
			engine.dispose();
		};
	}, []);

	return <canvas ref={canvasRef} className="fixed inset-0 block h-screen w-screen" />;
}

export default function CityEscapeSim3D() {
	const defaultLocalUrl = import.meta.env.DEV ? "/lmstudio" : "http://127.0.0.1:1234";
	const [settings, setSettings] = useState<Sim4Settings>({
		aiProvider: "local",
		aiModel: "gemma-4",
		localEndpointUrl: defaultLocalUrl,
		localModelName: "google/gemma-4-e2b",
		replicateApiToken: "",
		replicateModel: "",
		visionResolution: 32,
		aiHz: 2,
		enemySpeed: 3.2,
		enemyDamagePerSec: 14,
		botMaxSpeed: 6,
		botAccel: 10,
		energyDrainPerSec: 0.55,
	});
	const [paused, setPaused] = useState(false);
	const [controlsCollapsed, setControlsCollapsed] = useState(false);
	const [visionPreviewUrl, setVisionPreviewUrl] = useState<string>("");
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
	};

	return (
		<>
			<BabylonWorld settings={settings} paused={paused} onUi={setUi} onVisionPreview={setVisionPreviewUrl} />
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
								<Button variant="secondary" onClick={restart}>
									Restart
								</Button>
							</div>

							<div className="rounded-xl border border-slate-700 p-3 text-sm space-y-1">
								<div>Status: <strong>{ui.status}</strong></div>
								<div>Health: <strong>{ui.health.toFixed(0)}</strong></div>
								<div>Energy: <strong>{ui.energy.toFixed(0)}</strong></div>
								<div>Organs: <strong>{ui.organs}</strong></div>
								<div>Enemies alive: <strong>{ui.enemiesAlive}</strong></div>
								<div>Rocks active: <strong>{ui.rocksActive}</strong></div>
								<div className="text-xs text-slate-300">Tip: click on the ground to add enemies.</div>
							</div>

							<div className="rounded-xl border border-slate-700 p-3 text-sm space-y-2">
								<div className="font-semibold">Bot vision</div>
								{visionPreviewUrl ? (
									<img
										src={visionPreviewUrl}
										alt="Bot vision"
										className="w-full rounded-md border border-slate-700 bg-slate-950/40"
										style={{ imageRendering: "pixelated", aspectRatio: "1 / 1" }}
									/>
								) : (
									<div className="text-xs text-slate-300">Waiting for first frame…</div>
								)}
								<div className="text-xs text-slate-300">Resolution: {Math.round(settings.visionResolution)}×{Math.round(settings.visionResolution)}</div>
							</div>

							<div className="space-y-3">
								<div>
									<div className="flex justify-between text-sm mb-1"><span>AI Provider</span><span>{settings.aiProvider}</span></div>
									<select
										className="w-full rounded-md border border-slate-700 bg-slate-900/50 p-2 text-sm"
										value={settings.aiProvider}
										onChange={(e) => setSettings((p) => ({ ...p, aiProvider: e.target.value as AiProvider }))}
									>
										<option value="none">none (fallback)</option>
										<option value="local">local</option>
										<option value="replicate">replicate</option>
									</select>
								</div>

								<div>
									<div className="flex justify-between text-sm mb-1"><span>AI Model</span><span>{settings.aiModel}</span></div>
									<select
										className="w-full rounded-md border border-slate-700 bg-slate-900/50 p-2 text-sm"
										value={settings.aiModel}
										onChange={(e) => setSettings((p) => ({ ...p, aiModel: e.target.value as AiModel }))}
									>
										<option value="gpt-5.2">gpt-5.2</option>
										<option value="gemini-3">gemini-3</option>
										<option value="gemma-4">gemma-4</option>
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

