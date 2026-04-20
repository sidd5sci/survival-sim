import { Link, Navigate, Route, Routes, useNavigate, useParams } from 'react-router-dom'
import SurvivalSim3D from './simulations/sim1/sim.tsx'
import GeneticTrianglesSim3D from './simulations/sim2/sim2.tsx'
import NeuralVisionSim3D from './simulations/sim3/sim3.tsx'
import './App.css'

const simulations = [
  {
    id: 'survival-3d',
    title: '3D Survival Sandbox',
    description: 'AI bot survival simulation with obstacles, enemies, and food behavior.',
    status: 'Ready',
  },
  {
    id: 'genetic-triangles-3d',
    title: 'Genetic algo 3D',
    description: '100 triangle agents evolve DNA over 5-minute generations to reach and eat distant food while avoiding obstacles.',
    status: 'Ready',
  },
  {
    id: 'neural-vision-ga-3d',
    title: 'Neural Vision GA 3D',
    description: 'Agents with 5-unit vision evolve tiny neural networks where mutations can change weights, biases, neurons, and connections.',
    status: 'Ready',
  },
]

function Home() {
  return (
    <main className="home-page">
      <section className="home-hero">
        <p className="eyebrow">Simulation Hub</p>
        <h1>Choose a Simulation</h1>
        <p className="subtitle">Launch any simulation from this list. Add new ones to this page as your library grows.</p>
      </section>

      <section className="sim-grid" aria-label="Simulation list">
        {simulations.map((sim) => (
          <article key={sim.id} className="sim-card">
            <div className="sim-card-head">
              <h2>{sim.title}</h2>
              <span className="pill">{sim.status}</span>
            </div>
            <p>{sim.description}</p>
            <Link className="launch-btn" to={`/sim/${sim.id}`}>
              Open Simulation
            </Link>
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
