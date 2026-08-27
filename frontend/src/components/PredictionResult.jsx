function percent(value) {
  return `${(value * 100).toFixed(1)}%`
}

export default function PredictionResult({ result }) {
  const { predicted_label, confidence, low_confidence, top_predictions, message } = result

  return (
    <div className="card">
      <h2>Result</h2>

      {low_confidence && (
        <div className="notice">
          Confidence is below the {percent(result.confidence_threshold)} threshold, so treat
          this result as inconclusive rather than as an answer.
        </div>
      )}

      <p className={`result-label ${predicted_label}`}>{predicted_label}</p>
      <p className="result-message">{message}</p>

      {top_predictions.map((item) => (
        <div className="confidence-row" key={item.label}>
          <span className="name">{item.label}</span>
          <span className="bar">
            <span style={{ width: percent(item.confidence) }} />
          </span>
          <span className="value">{percent(item.confidence)}</span>
        </div>
      ))}

      <p className="result-message" style={{ marginTop: 14, marginBottom: 0 }}>
        Top prediction {percent(confidence)} &middot; inference {result.inference_ms} ms
      </p>
    </div>
  )
}
