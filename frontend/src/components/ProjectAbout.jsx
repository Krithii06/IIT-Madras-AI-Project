const RESUME_URL = '/Krithick_Balaji_Ramesh_Resume.pdf'
const PORTFOLIO_URL = 'https://ramesh-profile.vercel.app'

// Grouped by pipeline stage rather than by vendor, so the list reads as an
// explanation of how the thing was built instead of a badge collection.
const STACK = [
  {
    stage: 'Dataset',
    items: 'PlantVillage (apple subset, 3,171 images) · Hugging Face Hub · Pillow · NumPy · pandas',
  },
  {
    stage: 'Data integrity',
    items: 'MD5 exact-duplicate detection · 64-bit dHash near-duplicate detection · leaf-group audit',
  },
  {
    stage: 'Splitting',
    items: 'scikit-learn StratifiedGroupKFold grouped on leaf identity · fixed seed · disjointness asserted in tests',
  },
  {
    stage: 'Model',
    items: 'PyTorch · torchvision · MobileNetV2 / EfficientNet-B0 / ResNet18, ImageNet-pretrained',
  },
  {
    stage: 'Training',
    items: 'Two-stage transfer learning (frozen head, then fine-tune) · AdamW · cosine annealing · early stopping on macro F1',
  },
  {
    stage: 'Evaluation',
    items: 'scikit-learn metrics · confusion matrix · per-disease breakdown · leaf-level error analysis · Matplotlib',
  },
  {
    stage: 'Serving',
    items: 'ONNX Runtime for CPU inference — no PyTorch in the deployed image',
  },
  {
    stage: 'Backend',
    items: 'FastAPI · uvicorn · Pydantic · Docker',
  },
  {
    stage: 'Frontend',
    items: 'React 18 · Vite',
  },
  {
    stage: 'Testing & hosting',
    items: 'pytest · Playwright · Render (API) · Vercel (web app)',
  },
]

const TECHNIQUES = [
  ['Leaf-grouped splitting', 'The 3,171 images come from only 505 physical leaves. Splitting on images would put photographs of the same leaf on both sides of the split, so the split is grouped by leaf instead.'],
  ['Transfer learning', 'ImageNet-pretrained backbone with a replaced classifier head, trained in two stages so the new head does not disturb the pretrained features.'],
  ['Symptom-preserving augmentation', 'Flips, small rotations and mild brightness/contrast jitter — but hue is left almost untouched, because leaf colour is the diagnosis.'],
  ['Macro-F1 checkpointing', 'The saved model is the one with the best validation macro F1, not the best accuracy, so neither class can be quietly ignored.'],
  ['Structural leakage measurement', 'Rather than inferring leakage from an accuracy gap, the pipeline counts how many held-out images share a leaf with training: 0% here against 100% under a naive random split.'],
]

export default function ProjectAbout() {
  return (
    <>
      <div className="card">
        <h2>Tools and techniques</h2>

        <dl className="stack-list">
          {STACK.map(({ stage, items }) => (
            <div key={stage}>
              <dt>{stage}</dt>
              <dd>{items}</dd>
            </div>
          ))}
        </dl>

        <h3 className="subhead">Techniques worth calling out</h3>
        <ul className="technique-list">
          {TECHNIQUES.map(([name, why]) => (
            <li key={name}>
              <strong>{name}.</strong> {why}
            </li>
          ))}
        </ul>
      </div>

      <div className="card">
        <h2>Built by</h2>
        <p className="author-name">Krithick Balaji Ramesh</p>
        <p className="author-role">AI Research Intern, GEETHIK Technologies</p>
        <p className="author-role">B.Tech Computer Science &amp; Engineering, SRM Institute of Science &amp; Technology</p>
        <div className="author-links">
          <a href={PORTFOLIO_URL} target="_blank" rel="noopener noreferrer">
            Portfolio
          </a>
          <a href={RESUME_URL} target="_blank" rel="noopener noreferrer">
            Résumé (PDF)
          </a>
        </div>
      </div>
    </>
  )
}
