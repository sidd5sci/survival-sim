export async function replicatePredict(apiToken: string, modelVersion: string, input: any): Promise<any> {
	// Minimal, generic “Replicate-style” call. Many users run their own proxy with this shape.
	// If you use Replicate directly, you can point `modelVersion` at your model version id.
	const res = await fetch("https://api.replicate.com/v1/predictions", {
		method: "POST",
		headers: {
			"content-type": "application/json",
			Authorization: `Token ${apiToken}`,
		},
		body: JSON.stringify({ version: modelVersion, input }),
	});
	if (!res.ok) throw new Error(`Replicate HTTP ${res.status}`);
	return await res.json();
}
