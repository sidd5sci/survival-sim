# Simulation 4 — Maze Neural Vision GA 3D

## What it is
A genetic-algorithm simulation where each creature is controlled by a small neural network (DNA). Compared to Sim 3, this version generates a **maze** instead of scattered blocks.

Goals for evolution:
- Eat fruits
- Avoid obstacles/walls
- Maintain energy

## Maze world
- Obstacles are generated as maze walls (grid-based perfect maze).
- Fruits are placed at many reachable maze cell-centers.
- A clear area is kept near the start (no walls/blocks close to spawn).

Because it is a perfect maze, there is a path through corridors between maze cells; fruits are placed in those open cells.

## Controls
- **Pause / Resume**
- **Restart**
- **Vision Radius**
- **Iteration Duration**
- **Paths** (trails)
- **Learning** toggle + Learning Rate
- GA sliders (mutation / crossover / elite / population)

## Shortcuts
- Press **F** to drop a fruit at the current mouse pointer position (on ground).
- Press **O** to drop an obstacle at the current mouse pointer position (on ground).

Placement rules:
- Obstacles won’t be placed inside the start clear-zone.
- Foods/obstacles won’t be placed overlapping existing objects.

## Neural net
Inputs include food/obstacle signals + velocity + energy + fitness. Outputs are movement actions (L/R/F/B + speed).

## Fitness
Fitness is clamped to 0–100 and starts at 100, then penalties reduce it:
- Collisions
- Low energy
- Slow time-to-first-fruit
- Laziness penalty (per second) while staying within 10 units of spawn

## Files
- Implementation: `src/simulations/sim4/sim4.tsx`
