// The deployed backend URL is injected at build time. Falls back to the local
// uvicorn port so `npm run dev` works with no configuration.
const API_BASE = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')

// The free hosting tier stops the backend after 15 minutes without traffic and
// takes roughly a minute to start it again. That is longer than fetch's default
// patience on a slow connection, so requests carry an explicit timeout and the
// first one after a cold start is allowed to take much longer than the rest.
const PREDICT_TIMEOUT_MS = 90_000
const INFO_TIMEOUT_MS = 75_000

async function fetchWithTimeout(url, options = {}, timeoutMs = 30_000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(url, { ...options, signal: controller.signal })
  } finally {
    clearTimeout(timer)
  }
}

async function readError(response, fallback) {
  // FastAPI puts validation and HTTPException messages in `detail`.
  try {
    const body = await response.json()
    if (typeof body.detail === 'string') return body.detail
  } catch {
    // Non-JSON error body; fall through to the generic message.
  }
  return fallback
}

export async function predict(file) {
  const form = new FormData()
  form.append('file', file)

  let response
  try {
    response = await fetchWithTimeout(
      `${API_BASE}/predict`,
      { method: 'POST', body: form },
      PREDICT_TIMEOUT_MS,
    )
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new Error('The prediction service did not respond in time. It may still be starting up - please try again.')
    }
    throw new Error('Could not reach the prediction service. Check that the backend is running.')
  }

  if (!response.ok) {
    // 503 is what the API returns while the model is still loading.
    if (response.status === 503) {
      throw new Error('The model is still loading. Give it a few seconds and try again.')
    }
    throw new Error(await readError(response, `Request failed (${response.status}).`))
  }
  return response.json()
}

export async function fetchModelInfo() {
  const response = await fetchWithTimeout(`${API_BASE}/model-info`, {}, INFO_TIMEOUT_MS)
  if (!response.ok) throw new Error('Model information is unavailable.')
  return response.json()
}

export { API_BASE }
