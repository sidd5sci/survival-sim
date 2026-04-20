export type EnsureModelResult =
	| { state: "ready"; message: string }
	| { state: "downloading"; message: string }
	| { state: "error"; message: string };

export function normalizeLmStudioBaseUrl(input: string): string {
	const trimmed = input.trim().replace(/\/+$/, "");
	if (!trimmed) return "";
	// Allow using the Vite dev proxy.
	if (trimmed === "/lmstudio") return trimmed;
	// Accept either full base URL or mistakenly pasted endpoint URLs.
	if (trimmed.endsWith("/api/v1/chat")) return trimmed.slice(0, -"/api/v1/chat".length);
	if (trimmed.endsWith("/api/v1/models")) return trimmed.slice(0, -"/api/v1/models".length);
	if (trimmed.endsWith("/v1/chat/completions")) return trimmed.slice(0, -"/v1/chat/completions".length);
	return trimmed;
}

const loadCache = new Map<string, { loadedUntilMs: number }>();
const downloadJobs = new Map<string, { jobId: string; lastStatusAtMs: number }>();

async function listModels(baseUrl: string): Promise<any> {
	const res = await fetch(`${baseUrl}/api/v1/models`, {
		method: "GET",
		headers: { "content-type": "application/json" },
	});
	if (!res.ok) throw new Error(`LM Studio list models HTTP ${res.status}`);
	return await res.json();
}

async function downloadModel(baseUrl: string, modelKey: string): Promise<{ status: string; jobId?: string }> {
	const res = await fetch(`${baseUrl}/api/v1/models/download`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ model: modelKey }),
	});
	if (!res.ok) throw new Error(`LM Studio download model HTTP ${res.status}`);
	const json = await res.json();
	return { status: String(json?.status ?? ""), jobId: typeof json?.job_id === "string" ? json.job_id : undefined };
}

async function downloadStatus(baseUrl: string, jobId: string): Promise<{ status: string }> {
	const res = await fetch(`${baseUrl}/api/v1/models/download/status/${encodeURIComponent(jobId)}`, {
		method: "GET",
		headers: { "content-type": "application/json" },
	});
	if (!res.ok) throw new Error(`LM Studio download status HTTP ${res.status}`);
	const json = await res.json();
	return { status: String(json?.status ?? "") };
}

async function loadModel(baseUrl: string, modelKey: string): Promise<void> {
	const res = await fetch(`${baseUrl}/api/v1/models/load`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ model: modelKey }),
	});
	if (!res.ok) throw new Error(`LM Studio load model HTTP ${res.status}`);
	await res.json();
}

export async function ensureLmStudioModelReady(lmStudioBaseUrl: string, modelKey: string): Promise<EnsureModelResult> {
	const baseUrl = normalizeLmStudioBaseUrl(lmStudioBaseUrl);
	const model = modelKey.trim();
	if (!baseUrl) return { state: "error", message: "Missing LM Studio base URL" };
	if (!model) return { state: "error", message: "Missing model key" };

	const cacheKey = `${baseUrl}::${model}`;
	const cached = loadCache.get(cacheKey);
	if (cached && Date.now() < cached.loadedUntilMs) return { state: "ready", message: "Model ready" };

	const list = await listModels(baseUrl);
	const models: any[] = Array.isArray(list?.models) ? list.models : [];
	const found = models.find((m) => m?.key === model);

	if (!found) {
		const existingJob = downloadJobs.get(cacheKey);
		if (!existingJob) {
			const started = await downloadModel(baseUrl, model);
			if (started.jobId) downloadJobs.set(cacheKey, { jobId: started.jobId, lastStatusAtMs: 0 });
			return { state: "downloading", message: `Downloading model (${started.status || "started"})` };
		}

		if (Date.now() - existingJob.lastStatusAtMs > 1000) {
			const st = await downloadStatus(baseUrl, existingJob.jobId);
			existingJob.lastStatusAtMs = Date.now();
			downloadJobs.set(cacheKey, existingJob);
			if (st.status === "completed" || st.status === "already_downloaded") {
				downloadJobs.delete(cacheKey);
				return { state: "downloading", message: "Download completed, preparing…" };
			}
			if (st.status === "failed") return { state: "error", message: "Model download failed" };
			return { state: "downloading", message: `Downloading model (${st.status})` };
		}
		return { state: "downloading", message: "Downloading model" };
	}

	const loadedAlready = Array.isArray(found?.loaded_instances) && found.loaded_instances.length > 0;
	if (loadedAlready) {
		loadCache.set(cacheKey, { loadedUntilMs: Number.POSITIVE_INFINITY });
		return { state: "ready", message: "Model ready" };
	}

	await loadModel(baseUrl, model);
	loadCache.set(cacheKey, { loadedUntilMs: Number.POSITIVE_INFINITY });
	return { state: "ready", message: "Model ready" };
}

export type LmStudioChatRequest = {
	lmStudioBaseUrl: string;
	model: string;
	systemPrompt: string;
	userText: string;
	visionDataUrl?: string | null;
	maxOutputTokens?: number;
	temperature?: number;
};

export async function lmStudioChatText(req: LmStudioChatRequest): Promise<string> {
	const baseUrl = normalizeLmStudioBaseUrl(req.lmStudioBaseUrl);
	if (!baseUrl) throw new Error("Missing LM Studio base URL");
	const modelKey = req.model.trim();
	if (!modelKey) throw new Error("Missing model key");

	const max_output_tokens = req.maxOutputTokens ?? 140;
	const temperature = req.temperature ?? 0;

	// LM Studio REST payload format can vary by server/version.
	// This server variant expects discriminated items: {type:"text", content:"..."} and {type:"image", data_url:"..."}.
	// We try the discriminated multimodal shape first to ensure images are actually sent.
	const inputDiscriminated: any = req.visionDataUrl
		? [
				{ type: "text", content: req.userText },
				{ type: "image", data_url: req.visionDataUrl },
			]
		: [{ type: "text", content: req.userText }];

	// Older/alternate variants may accept "message objects" with content + images.
	const inputMessages: any = req.visionDataUrl
		? [{ role: "user", content: req.userText, images: [req.visionDataUrl] }]
		: [{ role: "user", content: req.userText }];

	const doChat = async (input: any, includeReasoningOff: boolean) => {
		return await fetch(`${baseUrl}/api/v1/chat`, {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify({
				model: modelKey,
				system_prompt: req.systemPrompt,
				input,
				temperature,
				max_output_tokens,
				...(includeReasoningOff ? { reasoning: "off" } : {}),
				store: false,
			}),
		});
	};

	let res = await doChat(inputDiscriminated, true);
	let errText = "";
	if (!res.ok) {
		try {
			errText = await res.text();
		} catch {
			errText = "";
		}

		// If server rejects the `reasoning` field, retry without it.
		const reasoningKeyRejected = /unrecognized key\(s\) in object:\s*'reasoning'/i.test(errText);
		if (reasoningKeyRejected) {
			res = await doChat(inputDiscriminated, false);
			if (!res.ok) {
				try {
					errText = await res.text();
				} catch {
					errText = "";
				}
			} else {
				errText = "";
			}
		}

		// If server doesn't accept discriminated items, try message-object variant.
		const expectsMessagesShape = /input\.?0\.?content/i.test(errText) || /unrecognized key\(s\) in object:\s*'text'/i.test(errText);
		const expectsDiscriminated =
			/invalid discriminator value/i.test(errText) ||
			/expected\s*["']text["']\s*\|\s*["']image["']/i.test(errText) ||
			/unrecognized key\(s\) in object:\s*'content'/i.test(errText);
		if (expectsMessagesShape) {
			res = await doChat(inputMessages, false);
			if (!res.ok) {
				try {
					errText = await res.text();
				} catch {
					errText = "";
				}
			} else {
				errText = "";
			}
		} else if (expectsDiscriminated) {
			// Already tried discriminated; keep errText for OpenAI fallback below.
		}
	}

	if (res.ok) {
		const json = await res.json();
		const outItems: any[] = Array.isArray(json?.output) ? json.output : [];
		const messages = outItems
			.filter((it) => it?.type === "message" && typeof it?.content === "string")
			.map((it) => String(it.content));
		if (messages.length) return messages[messages.length - 1];

		const reasoning = outItems
			.filter((it) => it?.type === "reasoning" && typeof it?.content === "string")
			.map((it) => String(it.content));
		return reasoning.length ? reasoning[reasoning.length - 1] : "";
	}

	if (!errText) {
		try {
			errText = await res.text();
		} catch {
			errText = "";
		}
	}

	const shouldTryOpenAi = /messages\s*field\s*is\s*required/i.test(errText) || res.status === 404 || res.status === 405;
	if (!shouldTryOpenAi) throw new Error(`LM Studio chat HTTP ${res.status}: ${errText}`);

	const userContent: any = req.visionDataUrl
		? [
				{ type: "text", text: req.userText },
				{ type: "image_url", image_url: { url: req.visionDataUrl } },
			]
		: req.userText;

	const res2 = await fetch(`${baseUrl}/v1/chat/completions`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({
			model: modelKey,
			temperature,
			max_tokens: max_output_tokens,
			messages: [
				{ role: "system", content: req.systemPrompt },
				{ role: "user", content: userContent },
			],
		}),
	});
	if (!res2.ok) throw new Error(`Local AI HTTP ${res2.status}`);
	const json2 = await res2.json();
	return json2?.choices?.[0]?.message?.content ?? "";
}
