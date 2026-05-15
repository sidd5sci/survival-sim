import { Link, Navigate, Route, Routes, useNavigate, useParams } from 'react-router-dom'
import SurvivalSim3D from './simulations/sim1/sim.tsx'
import GeneticTrianglesSim3D from './simulations/sim2/sim2.tsx'
import NeuralVisionSim3D from './simulations/sim3/sim3.tsx'
import MazeNeuralVisionSim3D from './simulations/sim4/sim4.tsx'
import CityEscapeSim3D from './simulations/sim15/sim15.tsx'
import CityEscapeGeneticSim3D from './simulations/sim6/sim6.tsx'
import SplatWebSim3D from './simulations/splatsWeb/splatsWeb.tsx'
import './App.css'

const simulations = [
  {
    id: 'survival-3d',
    title: '3D Survival Sandbox',
    description: 'AI bot survival simulation with obstacles, enemies, and food behavior.',
    status: 'Ready',
		docsPath: '/docs/sim1-survival-sandbox.md',
  },
  {
    id: 'genetic-triangles-3d',
    title: 'Genetic algo 3D',
    description: '100 triangle agents evolve DNA over 5-minute generations to reach and eat distant food while avoiding obstacles.',
    status: 'Ready',
		docsPath: '/docs/sim2-genetic-triangles.md',
  },
  {
    id: 'neural-vision-ga-3d',
    title: 'Neural Vision GA 3D',
    description: 'Agents with 5-unit vision evolve tiny neural networks where mutations can change weights, biases, neurons, and connections.',
    status: 'Ready',
		docsPath: '/docs/sim3-neural-vision-ga.md',
  },
  {
    id: 'maze-neural-vision-ga-3d',
    title: 'Maze Neural Vision GA 3D',
    description: 'Like Sim 3, but the world is a maze and fruits are spread across reachable corridors. Includes quick placement shortcuts (F/O).',
    status: 'Ready',
		docsPath: '/docs/sim4-maze-neural-vision-ga.md',
  },
  {
    id: 'city-escape-ai-3d',
    title: 'City Escape AI 3D',
    description: 'One articulated bot must escape a procedural city while enemies chase; vision is sent as a low-res matrix to a configurable AI endpoint.',
    status: 'MVP',
		docsPath: '/docs/sim15-city-escape-ai.md',
  },
  {
    id: 'city-escape-genetic-3d',
    title: 'City Escape Genetic NN 3D',
    description: 'Large neural network trained with genetic evolution from bot vision + instruction features; outputs direct bot actions in real time.',
    status: 'New',
		docsPath: '/docs/sim15-city-escape-genetic.md',
  },
  {
    id: 'splats-webgl-face',
    title: 'Gaussian Splat WebGL',
    description: 'Structured facial topology splat renderer (WebGL) with separate face/body shading and readable motion silhouette.',
    status: 'New',
		docsPath: '/docs/sim4-survival.md',
  },
]

function Home() {
  return (
    <main className="home-page">
      <section className="home-hero">
        <p className="eyebrow">Simulation Hub</p>
        <h1>Choose a Simulation</h1>
        <p className="subtitle">Launch any simulation from this list. Add new ones to this page as your library grows.</p>
    <div className="home-actions">
      <a
        className="docs-btn"
        href="https://github.com/sidd5sci/survival-sim"
        target="_blank"
        rel="noreferrer"
      >
        GitHub Repo
      </a>
      <a
        className="launch-btn"
        href="https://paypal.me/learnkevin?locale.x=en_GB&country.x=IN"
        target="_blank"
        rel="noreferrer"
      >
        Buy me a coffee
      </a>
    </div>
      </section>

      <section className="sim-grid" aria-label="Simulation list">
        {simulations.map((sim) => (
          <article key={sim.id} className="sim-card">
            <div className="sim-card-head">
              <h2>{sim.title}</h2>
              <span className="pill">{sim.status}</span>
            </div>
            <p>{sim.description}</p>
			<div className="sim-card-actions">
				<Link className="launch-btn" to={`/sim/${sim.id}`}>
					Open Simulation
				</Link>
				<a className="docs-btn" href={sim.docsPath} target="_blank" rel="noreferrer">
					Read docs
				</a>
			</div>
          </article>
        ))}
      </section>
    </main>
  )
}

function SimulationView() {
  const navigate = useNavigate()
  const { simulationId } = useParams()

  let simulation = <SurvivalSim3D />
  if (simulationId === 'genetic-triangles-3d') simulation = <GeneticTrianglesSim3D />
  if (simulationId === 'neural-vision-ga-3d') simulation = <NeuralVisionSim3D />
  if (simulationId === 'maze-neural-vision-ga-3d') simulation = <MazeNeuralVisionSim3D />
  if (simulationId === 'city-escape-ai-3d') simulation = <CityEscapeSim3D />
  if (simulationId === 'city-escape-genetic-3d') simulation = <CityEscapeGeneticSim3D />
  if (simulationId === 'splats-webgl-face') simulation = <SplatWebSim3D />

  const validIds = new Set(simulations.map((s) => s.id))
  if (!simulationId || !validIds.has(simulationId)) {
    return <Navigate to="/" replace />
  }

  return (
    <>
      <button className="back-btn" onClick={() => navigate('/')}>
        Back to Home
      </button>
      {simulation}
    </>
  )
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/sim/:simulationId" element={<SimulationView />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
