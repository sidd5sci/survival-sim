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
Input layer is fixed (`INPUTS = 8`). Inputs represent **nearest visible** food and obstacle (within radius AND within 120° FOV):
1. Food direction **right/left** in creature-local space
2. Food direction **forward** in creature-local space
3. Food proximity (0..1)
4. Obstacle direction **right/left** in creature-local space
5. Obstacle direction **forward** in creature-local space
6. Obstacle proximity (0..1)
7. Current velocity x normalized
8. Current velocity z normalized

Visibility filtering:
- A target is considered visible only if it is within radius AND within the 120° cone in front of the creature.

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
  - creature dies
  - stops moving
  - receives a fitness penalty

## Collision behavior
- Creatures are prevented from passing through obstacles by sub-stepped integration and penetration resolution against obstacle circles (obstacle size + agent radius) in the XZ plane.

Boundary behavior:
- The world boundary behaves like a wall. Hitting/clamping at the boundary counts as a collision.
- For perception, the boundary is treated as a “virtual obstacle” so creatures learn to avoid edges.

## Fitness (high level)
Fitness rewards:
- Food eaten
- Progress toward food
- Faster arrival

Fitness penalties:
- Collisions
- Death

## Notes
- The GA runs for a fixed generation duration and then breeds a new population.
- Optional learning can nudge weights during a generation; that change is written back to DNA.

### Parent selection (breeding)
Parents for crossover are selected with a weighted score that rewards:
- High fitness
- High remaining energy
- Eating the first food quickly (lower `firstFoodTime`)

This makes the next generation converge toward creatures that are both effective and efficient.

### Crossover-time mutation
When crossover happens, the child genome also receives a small additional “jitter” on some weights/biases.
This reduces the chance that crossover produces an offspring identical to a parent.
