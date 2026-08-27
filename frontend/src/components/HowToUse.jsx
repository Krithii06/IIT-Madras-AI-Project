// Sits above the upload area so a first-time visitor knows what the tool expects
// before they pick a file, rather than after a confusing result.
const STEPS = [
  'Choose a leaf photograph — drop it on the upload area or click to browse. JPG, PNG or WebP, up to 8 MB.',
  'Check the preview. One apple leaf filling most of the frame, on a plain background, works best.',
  'Press "Classify leaf". The first request after a quiet period can take about a minute while the free backend wakes up.',
  'Read the result: the predicted label, how confident the model is, and the score for both classes.',
]

export default function HowToUse() {
  return (
    <div className="card">
      <h2>How to use this</h2>
      <ol className="steps">
        {STEPS.map((step) => (
          <li key={step}>{step}</li>
        ))}
      </ol>
      <p className="note">
        The model was trained on apple leaves only. Anything else — another crop, a field
        photo with a busy background, or a picture that is not a leaf at all — will still
        produce a confident-looking answer, and that answer means nothing. Confidence below
        70% is flagged as inconclusive, but a high score is not a guarantee either.
      </p>
    </div>
  )
}
