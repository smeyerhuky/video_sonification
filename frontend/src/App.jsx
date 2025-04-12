import './App.css'

// main entry point for the application
import Main from './components/Main'

/**
 * App Component
 *
 * Main application component that renders the Natural Language UI Control System.
 * This system allows users to control data visualizations using natural language commands.
 */
function App() {
  return (
    <div className="app-container">
      <Main />
    </div>
  )
}

export default App
