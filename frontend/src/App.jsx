import { useEffect, useState } from 'react'

import HowToUse from './components/HowToUse'
import ImageUpload, { validateFile } from './components/ImageUpload'
import ModelInfo from './components/ModelInfo'
import PredictionResult from './components/PredictionResult'
import ProjectAbout from './components/ProjectAbout'
import { fetchModelInfo, predict } from './services/api'
import './styles/app.css'

export default function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [modelInfo, setModelInfo] = useState(null)
  const [backendWaking, setBackendWaking] = useState(false)

  useEffect(() => {
    // This request doubles as the wake-up call. The free tier stops the backend
    // after 15 minutes idle, so loading the page starts it warming while the
    // visitor is still choosing a file.
    let cancelled = false
    const wakingTimer = setTimeout(() => {
      if (!cancelled) setBackendWaking(true)
    }, 2500)

    fetchModelInfo()
      .then((info) => {
        if (!cancelled) setModelInfo(info)
      })
      .catch(() => {
        if (!cancelled) setModelInfo(null)
      })
      .finally(() => {
        clearTimeout(wakingTimer)
        if (!cancelled) setBackendWaking(false)
      })

    return () => {
      cancelled = true
      clearTimeout(wakingTimer)
    }
  }, [])

  // Object URLs are not garbage collected on their own.
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      return undefined
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  function handleSelect(picked) {
    const problem = validateFile(picked)
    if (problem) {
      setError(problem)
      return
    }
    setError(null)
    setResult(null)
    setFile(picked)
  }

  function handleClear() {
    setFile(null)
    setResult(null)
    setError(null)
  }

  async function handlePredict() {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      setResult(await predict(file))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <header className="site-header">
        <h1>Plant Disease Classification</h1>
        <p>Upload an apple leaf image to check whether the model reads it as healthy or diseased.</p>
      </header>

      {backendWaking && (
        <div className="card">
          <div className="notice">
            Waking the prediction service. It sleeps after 15 minutes of inactivity on the
            free hosting tier and takes about a minute to come back.
          </div>
        </div>
      )}

      <HowToUse />

      <div className="card">
        <h2>Upload</h2>
        <ImageUpload
          file={file}
          previewUrl={previewUrl}
          onSelect={handleSelect}
          onClear={handleClear}
          disabled={loading}
        />
        <div className="actions">
          <button className="primary" onClick={handlePredict} disabled={!file || loading}>
            {loading ? 'Classifying...' : 'Classify leaf'}
          </button>
        </div>
      </div>

      {error && (
        <div className="card">
          <div className="error">{error}</div>
        </div>
      )}

      {result && <PredictionResult result={result} />}

      <ModelInfo info={modelInfo} />

      <ProjectAbout />

      <div className="disclaimer">
        <p>
          This is an experimental classifier built for a technical assessment. It was trained
          only on apple leaf photographs taken under controlled laboratory conditions, so its
          output is a model prediction and not a diagnosis.
        </p>
        <p>
          Results on other crops, on field photographs, or on leaves with cluttered backgrounds
          are not supported by the evaluation and should not be relied on.
        </p>
      </div>
    </div>
  )
}
