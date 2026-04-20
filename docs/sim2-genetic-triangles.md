# Simulation 2 — Genetic Triangles (GA) 3D

## What it is
A 3D genetic-algorithm simulation where a population of triangle agents evolves over generations to improve a fitness objective (reach/eat food while avoiding obstacles).

## What you see
- A bounded ground plane (world boundary).
- Obstacles (boxes).
- Food targets.
- A population of moving triangle agents.

## Controls (left panel)
- **Pause / Resume**: stops/continues stepping.
- **Restart**: resets the world back to generation 1.
- **Mutation Rate**: probability of changing DNA values.
- **Mutation Strength**: magnitude of weight/parameter perturbations.
- **Elite Count**: how many top performers are copied into the next generation.
- **Crossover Rate**: probability that a child mixes DNA from two parents.
- **Population (restart)**: population size for the next restart.

## Evolution loop
- The sim runs a fixed-length generation (timer).
- At the end of the generation:
  - Agents are ranked by fitness.
  - Top **elites** are copied.
  - Remaining children are produced via selection + crossover + mutation.

## Agent behavior (high level)
- Each agent moves each tick according to its genome-driven policy.
- Agents collide with obstacles and are penalized.
- Agents get rewarded for reaching/consuming food and making progress.

## Notes
- This simulation focuses on *population-level evolution* of behavior.
- Fitness is computed continuously during the generation and used for selection.
