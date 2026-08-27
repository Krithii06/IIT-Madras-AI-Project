export default function ModelInfo({ info }) {
  if (!info) return null

  return (
    <div className="card">
      <h2>Model</h2>
      <dl className="info-grid">
        <div>
          <dt>Architecture</dt>
          <dd>{info.architecture}</dd>
        </div>
        <div>
          <dt>Classes</dt>
          <dd>{info.classes.join(', ')}</dd>
        </div>
        <div>
          <dt>Input size</dt>
          <dd>
            {info.input_size}&times;{info.input_size} px
          </dd>
        </div>
        <div>
          <dt>Confidence threshold</dt>
          <dd>{(info.confidence_threshold * 100).toFixed(0)}%</dd>
        </div>
        <div style={{ gridColumn: '1 / -1' }}>
          <dt>Training data</dt>
          <dd>
            {info.dataset} &mdash; {info.source_classes.length} source classes grouped into{' '}
            {info.num_classes}
          </dd>
        </div>
      </dl>
    </div>
  )
}
