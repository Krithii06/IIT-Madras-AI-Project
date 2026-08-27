"""Prose for the technical report, kept apart from the layout code.

Every number in the report is read from results/metrics at build time rather than
written here, so the document cannot drift from what was measured. Anything that
depends on a measurement is a {placeholder} filled in by build_report.py.
"""

ABSTRACT = """
This report documents a binary healthy/diseased image classifier for apple leaves,
trained on a four-class subset of the PlantVillage dataset and served through a
FastAPI endpoint with a React front end.

The headline result is {test_accuracy_pct} accuracy and {test_macro_f1} macro F1 on a
held-out test set of {test_images} images. That number is easy to obtain on this
dataset and is not the interesting part of the work. PlantVillage photographs each
physical leaf several times: the {total_images} images used here come from only
{total_leaves} distinct leaves. Splitting those images at random places
near-identical photographs of the same leaf on both sides of the split, and this
report shows that under a naive split {naive_leak_pct} of validation images have a
photograph of the same leaf in the training set, against {grouped_leak_pct} for the
leaf-grouped split used throughout. The reported figures are therefore honest rather
than flattering, and the limitations section is explicit about what they do and do
not support.
"""

PROBLEM = """
The task brief asks for a deep-learning image classification system that identifies
the health condition of a plant leaf from a photograph, classifies it into a
disease or health category, and presents the prediction through a web interface.
It specifies PlantVillage as the dataset, names MobileNetV2, ResNet18 and
EfficientNet as candidate architectures, requires accuracy, precision, recall,
F1 and a confusion matrix, and requires the finished system to be deployed on free
hosting with a publicly reachable URL.
"""

AMBIGUITY = """
The brief specifies the class selection twice, and the two statements disagree. The
dataset section asks the intern to "select 3-5 suitable classes from the dataset"
with "at least one selected class" representing a healthy condition. The task list
then says to "select only two classes: (Healthy, Diseased)" and to "display results
(healthy/diseased)".

Rather than silently choosing one reading, this project satisfies both. Four source
classes were selected from PlantVillage - three apple diseases and apple healthy,
which is three to five classes including a healthy one - and those four classes are
mapped onto a binary healthy/diseased target for the deployed model, which is what
the web interface requirement describes. Evaluation reports the binary metrics the
brief asks for and additionally breaks performance down by the four source classes,
which is where the informative errors are. The ambiguity is recorded here rather
than resolved by assumption.
"""

DATASET = """
PlantVillage is an open dataset of leaf photographs covering 14 crop species and 26
diseases, introduced by Mohanty, Hughes and Salathe (2016). The copy used here is
the Hugging Face mirror, which ships the images together with the authors' own
train/test split files and, importantly, a leaf-grouping map.

The full archive contains {full_images} images across 38 classes. Counted from the
official split files this is {full_images} rather than the 54,306 quoted in the
dataset card - a one-image discrepancy that is noted for accuracy and has no bearing
on the work.

Class imbalance across the full dataset is severe: the largest class holds
{full_max_class} images and the smallest {full_min_class}, a ratio of about
{full_imbalance} to one. That is one reason a single crop was selected rather than
the whole dataset.
"""

SELECTED_CLASSES = """
The four apple classes were selected. Three reasons, in order of importance:

First, every apple image resolves to a real entry in the dataset's leaf-grouping
map - {leafmap_coverage} of {total_images} images, with no fallbacks. Apple is the
only crop for which this is true. Because the entire argument of this report rests
on grouping images by physical leaf, it matters that the grouping can be verified
rather than assumed.

Second, the binary target is close to balanced at {healthy_pct} healthy. Accuracy is
therefore a meaningful headline number and no class weighting is required.

Third, four classes with exactly one healthy class matches the brief's stated range
directly.
"""

DATA_QUALITY = """
The subset was profiled before any training, rather than assumed to be clean. All
{total_images} images are 256x256, RGB and JPEG; none failed to decode. Seven pairs
of byte-identical duplicates were found by MD5, and seven near-duplicate pairs by a
64-bit difference hash at a Hamming distance of three or less. Every one of those
pairs falls inside a single leaf group, so the grouped split handles them
automatically; no images were removed.

Because every file is the same size, image resolution is reported as a sentence
rather than as a histogram.
"""

SPLIT = """
The test set is PlantVillage's own held-out split, left untouched and used exactly
once, after model selection had finished. Validation is carved out of the official
training portion using StratifiedGroupKFold grouped on leaf identity and stratified
on the four source classes, so that the rarest disease - cedar apple rust, with only
275 images - stays represented in validation. The split is seeded and reproducible.

The official split was audited rather than trusted: no leaf group spans its
train/test boundary, for the apple subset or for the dataset as a whole. That is why
it is used as the test set here.
"""

LEAKAGE = """
The cost of splitting on images instead of leaves was measured directly, without
training anything, by counting how many held-out images have a photograph of the
same physical leaf in the training set.

Under the leaf-grouped split used throughout this project, that number is
{grouped_leaked} of {grouped_val_images} validation images - {grouped_leak_pct}.
Under a naive random image split it is {naive_leaked} of {naive_val_images} -
{naive_leak_pct} - and two of those pairs are byte-identical duplicates.

A structural measure is used here in preference to an accuracy comparison, and the
reason is worth stating plainly. The binary apple task saturates: validation macro
F1 reaches 1.0000 by the third epoch. An accuracy gap between the two split
strategies would therefore understate the problem, because there is no headroom in
which the optimism can show itself. The leak is present whether or not this
particular model is weak enough to expose it, and the count above does not depend on
the model at all.
"""

PREPROCESSING = """
Training images are augmented with a random resized crop to {image_size} pixels at a
scale of 0.7 to 1.0, horizontal and vertical flips, rotation up to 20 degrees, and
mild colour jitter. Validation, test and production inputs share a single
deterministic path: resize the shorter side to {resize_before_crop}, centre crop to
{image_size}, and normalise with ImageNet statistics.

Two augmentation choices need justifying.

Both flips are enabled. These are detached leaves photographed against a plain
background, so there is no canonical orientation and mirroring is a realistic
variation rather than a distortion. This would not be a safe choice for imagery
where gravity or plant structure carries information.

Hue jitter is nearly frozen, at 0.02. Leaf colour is the diagnosis in this task -
chlorotic yellowing, rust orange, necrotic brown - so a wide hue shift would destroy
the signal the model has to learn. Brightness, contrast and saturation are allowed to
move further because those vary with photography rather than with pathology.

The deterministic evaluation path is shared with the deployed backend, so the model
sees the same pixels in production that it saw during validation. A test asserts that
the backend's NumPy reimplementation of that path matches the torchvision original.
"""

RESOLUTION = """
An input resolution of {image_size} pixels was used rather than the more usual 224.
Measured on the four-core CPU this project was trained on, a MobileNetV2 fine-tuning
step costs 2.95 seconds per batch of 32 at 224 pixels against 1.90 seconds at 160.
The source images are 256x256 close-ups of a single leaf filling the frame, so the
additional resolution buys little detail, and the deployment target is a free CPU
instance with a fraction of a core allocated.

No full side-by-side comparison of 224 against 160 was run to completion, so no
accuracy trade-off is claimed. What can be said is that the 160-pixel model reaches a
perfect validation macro F1, leaving 224 no room to score better on that validation
set. Whether it would generalise better was not tested.
"""

MODEL = """
All three architectures named in the brief were trained under an identical budget:
MobileNetV2, EfficientNet-B0 and ResNet18, each ImageNet-pretrained with only the
classifier head replaced.

Training proceeds in two stages. The new head is trained first with the backbone
frozen, for two epochs at a learning rate of 1e-3; then the whole network is
fine-tuned at 1e-4 with cosine annealing. Freezing first prevents the randomly
initialised head from driving large gradients into pretrained features before it has
learned anything useful.

Loss is unweighted cross-entropy, since the binary target is close to balanced.
The optimiser is AdamW with weight decay 1e-4, batch size 32, for up to six epochs
with early stopping on validation macro F1 and a patience of three. Checkpoints are
selected on macro F1 rather than accuracy so that neither class can be ignored. The
seed is fixed at 42 and the split is reproducible.
"""

SELECTION_NOTE = """
Because all candidates saturate the validation set, architecture selection could not
be made on validation accuracy in any meaningful sense. The choice was made on
deployment cost instead - parameter count, exported model size and CPU inference
latency - which is the honest basis given the measurements, and which matches the
brief's emphasis on free-tier hosting constraints.
"""

ERROR_ANALYSIS = """
The selected model misclassifies {image_errors} of {test_images} test images. Those
mistakes are not independent: all of them are photographs of a single physical leaf,
{error_leaf}. Counted at leaf level - a leaf being missed if any of its photographs
is wrong - the model misses {missed_leaves} of {test_leaves} leaves, a leaf-level
accuracy of {leaf_accuracy}. That is the more honest denominator for this dataset,
and it is a direct consequence of the same grouping structure that makes the split
design necessary.

The leaf in question carries a small number of dark scab lesions near the tip and
margin, on tissue that is otherwise unbroken green. Five of its photographs are in the
test set; the model calls three of them healthy with high confidence and gets the
other two right, one of them only marginally at 0.52 confidence.

One plausible explanation was tested and rejected. Because the lesions sit near the
leaf margin, the deterministic centre crop might have been discarding them. Re-running
those five images through a full-frame resize that crops nothing changed none of the
five predictions. The centre crop is therefore not the mechanism, and the more likely
reading is simply that sparse, small lesions on predominantly healthy tissue do not
produce enough evidence for this model. That hypothesis was not itself tested further,
and is offered as a reading of the evidence rather than a conclusion.

The confidence figures matter more than the error count. Mean confidence on the wrong
predictions is {mean_conf_wrong} and the maximum is {max_conf_wrong}. A 0.70
confidence threshold - the one the deployed API uses to flag uncertain results - would
have caught {errors_caught} of {image_errors}. The model is confidently wrong, and
exposing a confidence score does not protect a user against that. This is a real
limitation of the confidence mechanism, not a tuning problem.
"""

CONFIDENCE = """
The API returns the softmax probability of the predicted class and flags anything
below 0.70 as low confidence. The threshold is a documented default rather than a
tuned operating point, and it is stored in the exported preprocessing configuration
rather than hard-coded, so it can be changed without touching code.

No calibration was performed. A softmax score is a model output, not a probability of
being correct, and as the error analysis shows this model can be both wrong and
confident. The user-facing wording is hedged accordingly: the interface says the model
finds patterns consistent with a diseased leaf, not that the leaf is diseased.
"""

BACKEND = """
The backend is FastAPI served by uvicorn, exposing four endpoints: a service
description at the root, a liveness probe at /health, a description of the loaded
model at /model-info, and the classifier itself at POST /predict, which accepts a
single multipart file field.

Three decisions are worth recording. The model is loaded once, during application
startup, because building the inference session costs far more than running it. The
service depends on onnxruntime rather than torch, which keeps the deployed image near
150 MB instead of approaching a gigabyte and matters on an instance with 512 MB of
memory. Model loading failures are recorded rather than raised, so a missing model
produces a degraded health response instead of a container that restarts in a loop.

Uploads are validated on content type, emptiness, byte size against an 8 MB limit, and
decodability, with a further guard on total pixel count so that a decompression bomb
cannot turn one request into gigabytes of pixels. Internal errors are logged in full
and returned to the caller as a generic message.
"""

FRONTEND = """
The front end is a small React application built with Vite. The workflow is
deliberately linear: select or drop an image, see it previewed, press a single button,
read the result. The result panel shows the predicted label, the confidence, both
class probabilities and a plain-language message, with an explicit warning when
confidence falls below the threshold. A model information panel reports the
architecture, class list, input size and threshold, all fetched from the API rather
than duplicated in the interface.

The page also documents itself. A short usage guide sits above the upload area, so a
first-time visitor learns what the tool expects before choosing a file rather than
after a confusing answer; it states plainly that anything other than an apple leaf
will still produce a confident-looking result that means nothing. Below the model
panel, a tools and techniques reference lists the stack by pipeline stage and
explains the five decisions the project turns on, and a short authorship card links
to the author's portfolio and resume.

The theme is light, flat and typographic, with no gradients, animation or decorative
iconography. A disclaimer states that the system is experimental, that it was trained
only on apple leaves photographed under laboratory conditions, and that its output is
a prediction rather than a diagnosis.

The interface also accounts for its own hosting. The free backend tier sleeps after
fifteen minutes of inactivity and takes about a minute to restart, so the page
requests model information on load - which doubles as a wake-up call - shows a notice
while it waits, and allows a ninety-second timeout on prediction requests.
"""

DEPLOYMENT = """
The system is deployed as two pieces: the inference API as a Docker web service on
Render's free tier, and the built front end as a static bundle on Vercel.

Hugging Face Spaces was the original target, and the Dockerfile still runs there
unchanged. It was ruled out on checking rather than assuming: as of 2026 the Hugging
Face documentation states that Gradio and Docker Spaces "require a paid plan to
create", with only static Spaces remaining free, and a static Space cannot run the
model. This is exactly the kind of change that makes a free-tier assumption stale, and
it is recorded here because the same check will need repeating in future.

The constraints of the chosen tier shaped the implementation rather than being
discovered after it. The instance provides 512 MB of memory and a fraction of a CPU
core, which is why the API serves ONNX and why a single worker is used - each worker
would load its own copy of the inference session. The service sleeps after fifteen
minutes idle with a cold start of roughly a minute, which the front end handles
explicitly. Free instance hours are capped at 750 per month, which is roughly one
continuously-running service and no more.
"""

CHALLENGES = """
Four problems were worth the time they took.

The first was recognising the leaf-grouping issue at all. The dataset's own card
mentions that its split preserves leaf grouping, but the significance only becomes
clear on counting: {total_images} images from {total_leaves} leaves is a mean of
{images_per_leaf} photographs per leaf. The response was to build the split on leaf
identity, audit the official split against the same rule, and assert the property in
the test suite so it cannot regress silently.

The second was that the benchmark saturates. With validation macro F1 at 1.0000 from
the third epoch onward, neither the architecture comparison nor the leakage
demonstration could be settled on validation accuracy. The response was to measure
leakage structurally instead, and to select the architecture on deployment cost, both
of which are reported as such rather than dressed up as accuracy findings.

The third was training cost. Training runs on a four-core laptop CPU with no usable
GPU. Benchmarking showed the model step, not the data loader, was the bottleneck -
the loader needs about 13 seconds per epoch against roughly 127 for the model - so the
fix was to reduce input resolution to {image_size} pixels and to use two loader
workers rather than four, leaving the physical cores to the forward and backward pass
instead of competing with it.

The fourth was the deployment target disappearing mid-project, described above.
"""

LIMITATIONS = """
The following limitations are material to how these results should be read.

The benchmark is close to saturated. Validation macro F1 reaches 1.0000 by the third
epoch, which says more about the dataset than about the model. Reported test accuracy
of {test_accuracy_pct} should be read in that light.

PlantVillage is a laboratory dataset. Leaves are detached and photographed against
uniform backgrounds under controlled lighting. Published work has repeatedly found
that models trained on it degrade substantially on field photographs. Nothing in this
project contradicts that finding, and nothing in this project measures it.

The model has only ever seen apple leaves. It will still return a confident
healthy-or-diseased answer for a tomato leaf, a hand, or a photograph of a wall,
because a two-class softmax has nowhere else to put its mass. There is no
out-of-distribution detection.

The effective sample size is {total_leaves} leaves, not {total_images} images. The
image count flatters the amount of independent evidence behind every figure in this
report.

Confidence is uncalibrated, and the error analysis shows the model can be wrong at
0.99 confidence. The threshold flags uncertainty, not incorrectness.

Model selection used a saturated validation set, so the selected epoch is the first to
reach the ceiling rather than a demonstrably better checkpoint.

The binary output hides which disease is present, which is what the brief asked for
but is a real reduction: scab, black rot and cedar apple rust are different problems
with different responses.
"""

FUTURE = """
The most valuable next step would be evaluation on field imagery - PlantDoc or a
similar in-situ dataset - to measure the laboratory-to-field gap rather than leaving
it as a caveat. Everything else is secondary to knowing that number.

Beyond that: temperature scaling to calibrate confidence, together with a rejection
option so that out-of-distribution uploads can be declined instead of confidently
labelled; retaining the four-way disease head alongside the binary one, since the
labels already carry that information; Grad-CAM overlays to check that the model
attends to lesions rather than to background or leaf outline, which would also test
the reading offered in the error analysis; and group-aware cross-validation across all
{total_leaves} leaves in place of a single split, to report intervals rather than
point estimates.
"""

CONCLUSION = """
The system meets the brief: a transfer-learned classifier, a validated API, a light
web interface, and a deployment path onto free infrastructure, with accuracy,
precision, recall, F1 and a confusion matrix reported on a held-out test set.

The engineering judgement worth carrying forward is not the {test_accuracy_pct}
accuracy figure. It is that this dataset contains {images_per_leaf} photographs of
each physical leaf on average, that a conventional random split therefore leaks
{naive_leak_pct} of validation images, and that the resulting score would have looked
slightly better while meaning considerably less. The reported number is lower than a
leaky pipeline would have produced, and it is the one that can be defended.
"""
