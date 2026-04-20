# Simulation 1 — 3D Survival Sandbox

## What it is
A BabylonJS sandbox where you can place **obstacles**, **enemies**, and **food** on a bounded ground plane and watch a single bot navigate while enemies chase.

## What you see
- A square world with a boundary outline.
- Static box obstacles.
- Food pickups.
- Enemies.
- A single controllable “bot” entity.

## Controls (left panel)
- **Placement**: choose what you place with clicks.
  - `Obstacle` / `Enemy` / `Food`
- **Pause / Resume**: stops/continues the simulation stepping.
- **Reset Bot**: respawns the bot and clears its internal state.
- **Random World**: randomizes world contents.
- **Clear World**: removes placed items.
- **Bot Speed**: adjusts how fast the bot moves.
- **Enemy Speed**: adjusts how fast enemies move.
- **Auto-spawn food**: periodically creates food.

## Interaction
- Click on the ground to place the currently selected item.
- Use the mouse on the canvas to orbit/zoom/pan (ArcRotateCamera).

## Core rules (high level)
- The bot moves and reacts to the environment every tick.
- Enemies move toward the bot.
- Food can be eaten/collected by the bot.
- Obstacles block motion (collision/avoidance).

## Notes
- This sim is primarily a sandbox/debug playground: you directly manipulate the world and observe behaviors.
