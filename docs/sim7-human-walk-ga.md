# Sim 7: Human Walk GA

Sim 7 trains a human-like stick figure to walk forward using a genetic algorithm.

## Core Idea

- Each genome stores gait parameters (hip amplitudes, knee amplitudes, phase, frequency, torso lean, stride gain).
- The simulation evaluates each genome over a fixed time window.
- Fitness rewards forward distance and staying upright, while penalizing unstable posture and excessive control effort.
- Top genomes are kept and mixed/mutated to produce the next generation.

## Controls

- Pause / Resume: stop or continue training.
- New Population: reset the population and restart learning from scratch.

## Metrics

- Generation: current evolution generation.
- Best Fitness: best score in the current generation.
- Best Distance: farthest distance reached by the best genome.
- Current Speed: velocity of the currently displayed best walker.
