# Simulation 3 — Neural Vision GA 3D

## What it is
A 3D genetic simulation where each creature is controlled by a **small neural network** (stored in DNA). Creatures have limited **vision radius** and a **120° forward field-of-view**. They evolve over generations to maximize fitness.

This sim adds:
- A neural network per creature (DNA includes topology + masks + weights/biases)
- A limited perception model (“vision”)
- Energy (creatures can die)
- Optional on-policy learning (Lamarckian write-back into DNA)

## What you see
- World boundary + ground.
- Obstacles (solid boxes).
- Food spheres.
- Triangle creatures that rotate to face their heading.

Notes:
- Obstacles are distributed across the full world (all directions).
- A small spawn area near the start is kept clear (no obstacles within ~6 units).
- The world boundary has visible walls and creatures treat the boundary as an obstacle.

## Interaction
- **Click a creature** to select it.
- When selected:
  - A ring shows the current vision radius.
  - A 120° “view segment” shows the creature’s forward FOV.

## Controls (left panel)
- **Pause / Resume**
- **Restart**
- **Vision Radius**: adjusts how far creatures can sense food/obstacles.
- **Iteration Duration**: generation length in seconds (default 180s / 3 min).
- **Paths**: show/hide trails (for all creatures).
- **Learning** toggle + **Learning Rate** (optional Lamarckian learning)
- GA parameters:
  - Mutation Rate / Strength
  - Topology Mutation
  - Crossover Rate
  - Elite Count
  - Population (restart)
- **Selected creature** dropdown (top performers + current selection)
  - Shows hidden topology (layers and neurons)
  - Shows energy status
  - Shows a neural network graph

## Creature perception (inputs)
Input layer is fixed (`INPUTS = 10`). Inputs represent **nearest visible** food and obstacle (within radius AND within 120° FOV), plus health-awareness signals:
1. `Fx`: Food direction **right/left** in creature-local space
2. `Fz`: Food direction **forward** in creature-local space
3. `Fd`: Food proximity (0..1)
4. `Ox`: Obstacle/wall direction **right/left** in creature-local space
5. `Oz`: Obstacle/wall direction **forward** in creature-local space
6. `Od`: Obstacle/wall proximity (0..1)
7. `Vx`: Current velocity x normalized (-1..1)
8. `Vz`: Current velocity z normalized (-1..1)
9. `En`: Energy normalized (0..1)
10. `Fit`: Fitness normalized (0..1)

Visibility filtering:
- A target is considered visible only if it is within radius AND within the 120° cone in front of the creature.

Boundary walls:
- Walls are treated as a virtual obstacle.
- Wall sensing uses nearest-wall distance, so walls remain “visible” even while sliding along them.

## Neural policy (outputs)
Output layer is fixed (`OUTPUTS = 5`). Outputs are action neurons:
- `left`, `right`, `forward`, `backward`, `speed`

These are combined into steering:
- `x = right - left`
- `z = forward - backward`
- `speed` scales acceleration and the effective max speed.

## DNA / topology
DNA (`BrainGenome`) includes:
- `hiddenLayers`: number of hidden layers (can increase during crossover/mutation; capped)
- `hiddenSize`: neurons per hidden layer
- Connection masks + weights:
  - input → hidden[0]
  - hidden[l-1] → hidden[l]
  - hidden[last] → output

The graph shown in the UI renders all hidden layers and their inter-layer connections.

## Energy + death
- Each creature starts with energy 100.
- Energy drains over time:
  - drains faster at higher speed
  - drains faster under higher acceleration
- Eating food increases energy.
- If energy reaches 0:
- If energy reaches 0 the creature effectively stops (no movement).

## Collision behavior
- Creatures are prevented from passing through obstacles by sub-stepped integration and penetration resolution against obstacle circles (obstacle size + agent radius) in the XZ plane.

Collision energy loss:
- On collision with an obstacle or wall, a creature loses a small fixed amount of energy (0.5).

Boundary behavior:
- The world boundary behaves like a wall. Hitting/clamping at the boundary counts as a collision.
- For perception, the boundary is treated as a “virtual obstacle” so creatures learn to avoid edges.

## Fitness (high level)
Fitness is strictly **0..100** and starts at **100**.

Penalties reduce fitness:
- More collisions
- Lower energy
- Slower time-to-first-fruit

Rewards increase fitness (but clamped to 100):
- More fruits eaten
- Faster eating rate
- Higher energy (above the start energy)

## Notes
- The GA runs for a fixed generation duration and then breeds a new population.
- Optional learning can nudge weights during a generation; that change is written back to DNA.

### Parent selection (breeding)
For crossover, parents are chosen only from the **top 10** creatures by fitness.
Elites are copied into the next generation (elite count is configurable).

### Crossover-time mutation
When crossover happens, the child genome also receives a small additional “jitter” on some weights/biases.
This reduces the chance that crossover produces an offspring identical to a parent.

## Genome export/import
- You can export the **best 10** genomes as a JSON file.
- You can load a previously exported JSON file to restart the sim seeded with those genomes.
- The loader is compatible with older exports (it pads missing inputs if the input count changed).
