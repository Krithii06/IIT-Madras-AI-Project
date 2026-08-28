# Plant Disease Classification

**Live app: <https://plant-disease-classification-rosy.vercel.app>**

A binary healthy/diseased classifier for apple leaf images, served through a Python
API and a small React front end. Built as an internship technical assessment.

The interesting part of this project is not the accuracy number. PlantVillage
photographs each physical leaf several times, and in the subset used here **3,171
images come from only 505 distinct leaves**. Split those images at random and
near-identical photographs of the same leaf end up on both sides, which produces a
test score the model has not earned. Most of the design below follows from taking
that seriously.

## Problem statement

From the task brief: take a plant leaf image, classify it into a health/disease
category, and show the prediction through a web interface. The brief specifies
PlantVillage as the dataset, names MobileNetV2 / ResNet18 / EfficientNet as
candidate architectures, asks for accuracy, precision, recall and F1 plus a
confusion matrix, and requires free hosting with a public URL.

### An ambiguity in the brief, and how it was resolved

The brief asks for two different things in two places:

> *Dataset section:* "select 3–5 suitable classes from the dataset. At least one
> selected class should represent a healthy condition."

> *Tasks section:* "Use the PlantVillage Dataset, but select only two classes:
> (Healthy, Diseased)" … "Display results (healthy/diseased)."

Rather than pick one and quietly drop the other, this project satisfies both:
**four source classes are selected** (three apple diseases plus apple healthy —
3–5 classes with at least one healthy), and they are **mapped onto a binary
healthy/diseased target** for the deployed model, which is what the UI requirement
describes. Evaluation additionally reports performance broken down by the four
source classes, which is where the useful error analysis is.

## Dataset

[PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage) — 54,305 images,
38 classes, 14 crops, CC BY-SA 3.0. Introduced in Mohanty, Hughes & Salathé (2016).

The images are not committed to this repository. See [data/README.md](data/README.md)
for the source and for how to rebuild the local copy.

### Selected subset

The four apple classes:

| Source class | Images | Binary label |
|---|---:|---|
| `Apple___Apple_scab` | 630 | diseased |
| `Apple___Black_rot` | 621 | diseased |
| `Apple___Cedar_apple_rust` | 275 | diseased |
| `Apple___healthy` | 1,645 | healthy |
| **Total** | **3,171** | 1,645 healthy / 1,526 diseased |

Apple was chosen for three reasons:

1. It is exactly four classes with one healthy class, matching the brief.
2. The binary target is nearly balanced (51.9% healthy), so accuracy is a
   meaningful headline number and no class weighting is needed.
3. **Every apple image resolves to a real entry in the dataset's leaf-grouping
   map** — 0 fallbacks out of 3,171. Apple is the only crop where that is true,
   and it is what makes the leak-free split verifiable rather than assumed.

### What was actually checked

Rather than assume the dataset is clean, it was profiled:

| Property | Result |
|---|---|
| Resolution | 3,171/3,171 at 256×256 |
| Colour mode / format | 3,171/3,171 RGB JPEG |
| Corrupt or unreadable files | 0 |
| Exact duplicates (md5) | 7 pairs — all within a single leaf group |
| Near duplicates (dHash ≤ 3) | 7 pairs — all within a single leaf group |
| Distinct physical leaves | 505 (mean 6.28 images per leaf) |
| Images resolving to a real leaf id | 3,171 (0 fallbacks) |

Image resolution is reported as a sentence rather than a histogram because every
file is the same size — a chart there would answer nothing.

## Data splitting, and why it is grouped

The test set is PlantVillage's own held-out split, untouched. Validation is carved
out of the official *training* portion with `StratifiedGroupKFold` grouped on
`leaf_id` and stratified on the four source classes, so the rarest disease (cedar
apple rust, 275 images) stays represented.

| Split | Images | Leaves | Healthy | Diseased |
|---|---:|---:|---:|---:|
| train | 2,131 | 340 | 1,131 | 1,000 |
| val | 429 | 68 | 228 | 201 |
| test | 611 | 97 | 286 | 325 |

Splitting is seeded and reproducible, and `tests/test_data.py` asserts that no leaf
appears in more than one split.

### The cost of getting this wrong

`python -m src.evaluation.leakage` measures the leak structurally, without training
anything:

| Split strategy | Validation images sharing a leaf with training |
|---|---:|
| Leaf-grouped (used here) | **0 of 429 — 0.0%** |
| Naive random image split | **426 of 426 — 100.0%** |

Under a naive split every single validation image has a photograph of the same
physical leaf in the training set, and two are byte-identical duplicates. The
official PlantVillage train/test split was audited the same way and is genuinely
leaf-clean (0 of 611 test images share a leaf with training), which is why it is
trusted as the test set here.

This structural measure is used rather than an accuracy gap on purpose: the binary
apple task saturates (see Results), so a difference in accuracy would understate
the problem. The leak is real whether or not this particular model is weak enough
to reveal it.

## Preprocessing and augmentation

Training augmentation:

```
RandomResizedCrop(160, scale=(0.7, 1.0))
RandomHorizontalFlip + RandomVerticalFlip
RandomRotation(20)
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02)
Normalize(ImageNet mean/std)
```

Two choices worth explaining:

- **Both flips are enabled.** These are detached leaves photographed on a plain
  background, so there is no canonical orientation — mirroring is a realistic
  variation, not a distortion.
- **Hue jitter is almost frozen at 0.02.** Leaf colour *is* the diagnosis here:
  chlorotic yellowing, rust orange, necrotic brown. A wide hue shift would destroy
  the signal the model has to learn. Brightness and contrast are allowed to move
  more, since those vary with photography rather than with pathology.

Validation, test and production preprocessing are identical and fully
deterministic — `Resize(182) → CenterCrop(160) → Normalize` — so evaluation is
repeatable and the deployed API returns the same answer for the same upload.

### Why 160×160 rather than 224

Measured on the 4-core CPU this project trains and deploys on, a MobileNetV2
fine-tune step costs **2.95 s at 224 against 1.90 s at 160**. The source images are
256×256 close-ups of a single leaf filling the frame, so the extra resolution buys
little, and the free hosting tier is CPU-only with 0.1 CPU allocated.

A full 224-versus-160 comparison was not run — the two were not trained to
completion side by side, so no accuracy trade-off is claimed here. What can be said
is that the 160px model reaches a perfect validation macro F1, which leaves 224 no
room to look better on this validation set. Whether it would generalise better is
untested.

## Model

Three ImageNet-pretrained backbones were trained under an identical budget:
MobileNetV2, EfficientNet-B0 and ResNet18. Only the classifier head is replaced.

Training is two-stage transfer learning: the new head is trained first with the
backbone frozen (2 epochs, lr 1e-3), then the whole network is fine-tuned at a
lower rate (lr 1e-4, cosine annealing). Freezing first stops the randomly
initialised head from pushing large gradients into pretrained features.

- Loss: cross-entropy (unweighted — the binary target is 53/47)
- Optimiser: AdamW, weight decay 1e-4
- Batch size 32, 6 epochs, early stopping on validation macro F1 (patience 3)
- Best checkpoint selected on **validation macro F1**, not accuracy, so neither
  class can be ignored
- Seed 42 throughout

## Results

Raw numbers in [results/metrics/](results/metrics/); regenerate with
`python -m src.evaluation.summarize`.

All four runs reach **1.0000 validation macro F1**, so validation does not separate
them. The deployed model is chosen on cost, measured under the same single-thread
setting the API uses:

| Run | Architecture | Split | Test acc. | Macro F1 | Params | ONNX | CPU/image | Train |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `mobilenet_leaf` | MobileNetV2 | leaf | **0.9951** | 0.9951 | 2.23 M | **9.1 MB** | **10.7 ms** | 18.6 min |
| `efficientnet_leaf` | EfficientNet-B0 | leaf | 0.9918 | 0.9918 | 4.01 M | 16.6 MB | 23.7 ms | 49.3 min |
| `resnet18_leaf` | ResNet18 | leaf | 0.9902 | 0.9901 | 11.18 M | 44.8 MB | 40.1 ms | 18.8 min |
| `mobilenet_random` | MobileNetV2 | random | 0.9967 | 0.9967 | 2.23 M | 9.1 MB | 10.7 ms | 90.1 min |

MobileNetV2 is both the most accurate on the held-out set and the cheapest to serve —
a fifth of ResNet18's exported size and a quarter of its latency — so the choice is
uncontroversial.

### The naive-split run scores highest, and that is not a finding

`mobilenet_random` tops the table at 0.9967. It is trained on a split where 100% of
validation images share a leaf with training, so it is worth being explicit: **this is
not evidence that leakage helps.** The test set is leaf-clean for every run, so leakage
never touched the reported number. What it corrupts is the *validation* signal used to
select a checkpoint — and with all four runs pegged at 1.0000, there is no headroom for
that corruption to show. The gap is two misclassified images against three, out of 611.
That is noise, and it is exactly why the leak is quantified structurally rather than
inferred from a score.

### Deployed model, on the held-out test split

611 images from 97 leaves, none sharing a leaf with training.

| Metric | Value |  | Class | Precision | Recall | F1 | n |
|---|---:|---|---|---:|---:|---:|---:|
| Accuracy | 0.9951 |  | healthy | 0.9896 | 1.0000 | 0.9948 | 286 |
| Macro F1 | 0.9951 |  | diseased | 1.0000 | 0.9908 | 0.9954 | 325 |
| Weighted F1 | 0.9951 |  | | | | | |

Confusion matrix (rows actual): `[[286, 0], [3, 322]]` — every error is a diseased leaf
called healthy, which for a disease detector is the worse direction.

### By original disease

| Source class | Images | Correct | Recall | Mean confidence |
|---|---:|---:|---:|---:|
| `Apple___Apple_scab` | 127 | 124 | 0.9764 | 0.9938 |
| `Apple___Black_rot` | 146 | 146 | 1.0000 | 0.9921 |
| `Apple___Cedar_apple_rust` | 52 | 52 | 1.0000 | 0.9999 |
| `Apple___healthy` | 286 | 286 | 1.0000 | 0.9932 |

Apple scab absorbs the entire error budget. Black rot and cedar apple rust produce large
high-contrast lesions; early scab can be a few small dark spots on otherwise green
tissue.

### Errors are correlated, and confidence does not catch them

All 3 errors are photographs of **one physical leaf**. At leaf level the model misses 1
of 97 — **leaf-level accuracy 0.9897**, the more honest denominator here. Mean confidence
on the wrong predictions is 0.9901 (max 0.9940), so the 0.70 threshold would have flagged
**0 of 3**. The model is confidently wrong, and showing a confidence score does not
protect the user against that.

A plausible cause was tested and rejected: because the lesions sit near the leaf margin,
the centre crop might have been discarding them. Re-running all five of that leaf's test
images with no cropping changed **none** of the five predictions.

## Running it locally

Requires Python 3.11+ and Node 18+.

```bash
git clone <repo-url>
cd plant-disease-classification
pip install -r requirements-dev.txt
```

`requirements-dev.txt` is the full environment: training, evaluation, figures and
tests. The `requirements.txt` beside it holds only the three runtime packages the
deployed serverless function installs — Vercel resolves Python dependencies from the
repository root, so that file has to stay small or `torch` ends up in the function.

### 1. Build the dataset

```bash
python -m src.data.prepare --download
```

Downloads the 2.2 GB archive, extracts the four apple classes and writes
`data/manifest.csv`. The download is cached by `huggingface_hub`, so re-running is
cheap. If you already have the archive, point at it instead and skip the download:

```bash
python -m src.data.prepare --zip /path/to/data.zip
```

The script prints the integrity report — resolutions, colour modes, duplicate
counts, per-class leaf counts — before writing anything.

### 2. Train

```bash
python run_experiments.py                    # all four runs
python run_experiments.py --only mobilenet_leaf
```

Each run writes `models/<run-name>/` containing `best_model.pt`, `class_mapping.json`,
`train_config.json` and `history.json`.

### 3. Evaluate

```bash
python -m src.evaluation.evaluate --run-name mobilenet_leaf --split test
python -m src.evaluation.leakage
python -m src.evaluation.summarize
```

### 4. Export for serving

```bash
python -m src.inference.export --run-name mobilenet_leaf
```

Writes `models/export/` with `model.onnx`, `class_mapping.json` and `preprocess.json`.
The export is checked against the torch model on random input and refuses to ship if
the logits diverge by more than 1e-3.

### 5. Run the API

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Interactive docs at <http://localhost:8000/docs>.

### 6. Run the front end

```bash
cd frontend
npm install
npm run dev
```

Opens on <http://localhost:5173> and talks to `http://localhost:8000` by default.

The page is a single scroll: a short **how to use this** guide, the upload area and
predict button, the result with both class scores, a **model** panel fed live from
`/model-info`, a **tools and techniques** reference describing how the system was
built, a **built by** card, and the disclaimer. The résumé PDF it links to lives in
`frontend/public/` and is served as a static asset.

### Tests

```bash
pytest -q
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Service description and endpoint list |
| `GET` | `/health` | Liveness probe; reports whether the model loaded |
| `GET` | `/model-info` | Architecture, classes, input size, threshold |
| `POST` | `/predict` | Classify an uploaded leaf image |

`POST /predict` takes `multipart/form-data` with a single `file` field.

```bash
curl -F "file=@leaf.jpg" http://localhost:8000/predict
```

```json
{
  "predicted_label": "diseased",
  "confidence": 0.9871,
  "low_confidence": false,
  "confidence_threshold": 0.7,
  "top_predictions": [
    {"label": "diseased", "confidence": 0.9871},
    {"label": "healthy", "confidence": 0.0129}
  ],
  "message": "The model finds visual patterns consistent with a diseased leaf.",
  "inference_ms": 24.3
}
```

Error responses use FastAPI's `{"detail": "..."}` shape:

| Status | When |
|---|---|
| 400 | Empty, corrupt or undecodable image |
| 413 | Over the 8 MB upload limit, or an unreasonable pixel count |
| 415 | Content type is not JPEG, PNG or WebP |
| 422 | No `file` field in the request |
| 503 | Model failed to load |

### Confidence

The API returns the softmax probability of the predicted class and flags anything
below **0.70** as low confidence. That threshold is a documented default, not a
tuned value, and it is stored in `models/export/preprocess.json` rather than
hard-coded. A softmax score is a model output, not a calibrated probability of
being right — the wording in the UI is hedged accordingly.

## Environment variables

| Variable | Used by | Default | Purpose |
|---|---|---|---|
| `ALLOWED_ORIGINS` | backend | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated CORS origins |
| `MODEL_DIR` | backend | `models/export` | Location of the exported bundle |
| `PORT` | backend | `7860` | Injected by the host |
| `VITE_API_URL` | frontend | `http://localhost:8000` | Backend base URL, read at **build** time |

No secrets are required. `.env` files are gitignored; `frontend/.env.example`
documents the shape.

## Deployment

**Live: <https://plant-disease-classification-rosy.vercel.app>**

Everything runs in one Vercel project on the free tier — the React build as static
files, and the inference API as Python serverless functions in `frontend/api/`. Both
answer on the same origin, so `/predict` is a same-origin call and **there is no CORS
configuration at all**.

| | |
|---|---|
| Live app | <https://plant-disease-classification-rosy.vercel.app> |
| Health | `/health` → `{"status":"ok","model_loaded":true}` |
| Warm inference (measured) | 8.8–22.8 ms |
| Sleeps when idle | no |

The original design was a two-host split — Docker on Render for the API, Vercel for
the front end. `backend/Dockerfile` and `render.yaml` are still here and still work;
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) documents that route. Single-platform won on
every axis, so it is what ships.

Three things about the Vercel build that cost four failed attempts, recorded because
none of them are obvious:

- It resolves Python dependencies from the **repository root**, not the configured
  root directory. `frontend/requirements.txt` was never read.
- It selects its own interpreter (CPython 3.14) and ignores `.python-version` in both
  locations. `onnxruntime` must therefore be at **1.24.3 or newer**, the first release
  with cp314 wheels — a test enforces this.
- The root `requirements.txt` must stay runtime-only. `torch` would push the function
  past the 250 MB limit; the development stack lives in `requirements-dev.txt`.

`frontend/api/` contains copies of the predictor and the model bundle, because Vercel
builds with `frontend/` as its root and cannot reach `src/` or `models/` above it.
`python -m src.inference.sync_vercel` refreshes them and `tests/test_vercel_bundle.py`
hashes them against their sources, so the copies cannot silently go stale.

Hugging Face Spaces was the original target and the Dockerfile still works there,
but as of 2026 the Hub documentation states that Gradio and Docker Spaces "require
a paid plan to create". Only static Spaces remain free, and a static Space cannot
run the model.

Two consequences of the free tier are visible in the code:

- The API serves **ONNX through onnxruntime and does not depend on torch**, which
  keeps the image near 150 MB instead of close to a gigabyte and fits the 512 MB
  instance.
- Render stops a free service after 15 minutes idle and takes about a minute to
  restart it. The front end requests `/model-info` on load, which doubles as a
  wake-up call, shows a notice while it waits, and allows a 90-second timeout.

## Limitations

Worth being direct about these.

- **The benchmark is close to saturated.** Validation macro F1 reaches 1.0000 by
  the third epoch. That says more about the dataset than about the model.
- **PlantVillage is a laboratory dataset.** Leaves are detached and photographed
  against uniform backgrounds under controlled lighting. Published work has
  repeatedly found that models trained on it degrade sharply on field photographs.
  Nothing here contradicts that, and nothing here measures it.
- **Apple only.** The model has never seen another crop. It will still return a
  confident healthy/diseased answer for a tomato leaf, a hand, or a photograph of
  a wall, because a two-class softmax has nowhere else to go. There is no
  out-of-distribution detection.
- **Binary output hides which disease it is.** That is what the brief asked for,
  but scab, black rot and cedar apple rust are clinically different problems.
- **Confidence is uncalibrated.** The 0.70 threshold is a reasonable default, not
  a validated operating point, and no calibration curve was produced.
- **505 leaves is a small effective sample.** The image count flatters the real
  amount of independent evidence behind these numbers.
- **Model selection used a saturated validation set.** With several epochs tied at
  1.0000 macro F1, "best epoch" is the first to reach the ceiling rather than a
  meaningfully better checkpoint.

## Possible improvements

- Field imagery — PlantDoc or similar — to measure the laboratory-to-field gap
  instead of leaving it as a caveat.
- A confidence calibration step (temperature scaling) and a rejection option, so
  out-of-distribution uploads can be declined rather than confidently labelled.
- Keep the four-way disease head alongside the binary one; the information is
  already in the labels.
- Grad-CAM overlays, to check the model is looking at lesions rather than at
  background or leaf outline.
- Group-aware cross-validation over all 505 leaves instead of a single split, for
  error bars rather than point estimates.

## Repository layout

```
├── src/
│   ├── config.py            paths, class selection, split and image settings
│   ├── data/                download, manifest building, leaf-grouped split
│   ├── preprocessing/       train and eval transforms
│   ├── training/            model factory and the training loop
│   ├── evaluation/          metrics, figures, leakage audit, summary tables
│   └── inference/           ONNX export and the torch-free predictor
├── backend/                 FastAPI service + Dockerfile
├── frontend/                React (Vite) single page app
├── tests/                   split integrity, inference contract, API behaviour
├── results/                 metrics, confusion matrices, figures
├── docs/DEPLOYMENT.md       step-by-step hosting instructions
├── render.yaml              Render blueprint for the backend
└── run_experiments.py       runs the four reported training jobs
```

## Licence

Code is MIT (see [LICENSE](LICENSE)). The PlantVillage dataset is CC BY-SA 3.0 and
is not redistributed here.
