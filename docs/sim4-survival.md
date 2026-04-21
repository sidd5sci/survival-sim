# Simulation 4 — City Escape AI 3D

## What it is
An MVP “city escape” simulation in BabylonJS.

- The world is a simple **procedurally generated 3D city** (a grid of box “buildings”).
- There is **one articulated AI bot** (torso/head + arms/legs with simple joint rotation).
- There are **many enemies** you can spawn on the ground; enemies chase the bot.
- The bot has an **eye camera** that produces a low-res 2D image which is:
	- converted into a **grayscale matrix** (0..1), and
	- sent to an AI model endpoint (local or Replicate-style).

Goal: **escape** by reaching the green exit ring.

## What you see
- Green exit ring near the far corner.
- Buildings (box obstacles) forming a city-like maze.
- Bot (white) with limbs.
- Enemies (red spheres).
- Rocks (small gray spheres) used as throwables.

## Interaction
- **Click the ground** to spawn an enemy at the clicked location.
- Orbit/zoom/pan the camera with the mouse.

## Controls (left panel)

### Game state
- **Pause / Resume**
- **Restart**

### Bot status
- Health, Energy, Organs
- Enemies alive, Rocks active

### Bot vision (live feed)
- Shows a **live pixelated preview** of what the bot’s eye camera sees.
- Resolution matches the “Vision resolution” slider.

### AI settings
- **AI Provider**:
	- `none (fallback)`: uses a built-in fallback controller (moves toward exit).
	- `local`: calls a local OpenAI-compatible endpoint (LM Studio).
	- `replicate`: calls Replicate-style predictions endpoint.
- **AI Model** (label only in this MVP): `gpt-5.2`, `gemini-3`, `gemma-4`
- **Vision resolution**: low-res square image size (e.g. 32×32)
- **AI tick (Hz)**: how frequently the bot queries the model.

## Local AI (LM Studio) integration
This sim supports LM Studio’s OpenAI-compatible server.

LM Studio endpoints (the sim treats your setting as a **base URL**):
- `GET  <base>/api/v1/models`
- `POST <base>/api/v1/models/load`
- `POST <base>/v1/responses` (preferred for tool calling)
- Fallbacks: `POST <base>/api/v1/chat` and/or `POST <base>/v1/chat/completions`

In the UI:
- Set **AI Provider** to `local`
- Set **LM Studio base URL** to:
	- `/lmstudio` (dev proxy; avoids browser CORS)
	- or `http://127.0.0.1:1234` (direct)
- Set **Local model name** to one of the IDs returned by `GET /api/v1/models` (default is `google/gemma-4-e2b`).

### Expected model response
The sim will try to use tool calling to obtain structured action arguments. If it falls back to text, the model must reply with **ONLY** a JSON object:

```json
{
	"left": 0,
	"right": 0,
	"forward": 1,
	"backward": 0,
	"speed": 0.8,
	"pick": false,
	"throw": false,
	"grow": false
}
```

Notes:
- Movement uses `left/right/forward/backward/speed`.
- Optional actions:
	- `pick`: pick up a nearby rock
	- `throw`: throw the held rock forward
	- `grow`: grow a new “organ” (increases max health/energy)

## Core rules (MVP)

### Survival
- Enemies damage the bot **on collision**.
- Bot has **health** and **energy**.
- Energy drains while moving; if energy hits 0, health starts dropping.

### Escape
- The bot “escapes” when it reaches the green exit ring.

### City collisions
- Buildings block movement. Movement uses sub-stepping + penetration resolution to reduce tunneling.
- The world boundary has visible walls (so the vision model can see the edge) and edges behave like obstacles.

### Rocks and combat
- Rocks exist in the world.
- Bot can pick a rock if close enough.
- Bot can throw a held rock.
- If a rock hits an enemy, the enemy dies (rock disappears).

### Organ growth
- When `grow` triggers and energy is sufficient, bot grows an organ:
	- increases max energy + max health
	- consumes some energy

## Current limitations
- Enemies are spheres (not skinned animals yet).
- “Organ growth” is visualized as small attached spheres (not a full procedural anatomy system).
- Replicate integration is intentionally generic and may need a proxy depending on your model setup.

## Files
- Implementation: `src/simulations/sim4/sim4.tsx`