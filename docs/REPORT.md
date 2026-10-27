# Campus Waste Segregation Assistant

*Neural Network & Deep Learning (PCCCS582) - End-term Project #5 - Report draft*

> Draft generated from the experiment outputs in `reports/`. Sections marked **[TODO]** need
> input that only the team can supply (campus photos, names, references in the required style).

## 1. Abstract

Wrong bin choices contaminate recycling streams on campus. We built an assistant that
classifies a photo of a single waste item into nine categories and turns the prediction into
a disposal action: which bin, how to prepare the item, and what not to do. A MobileNetV2
pre-trained on ImageNet and fine-tuned in two phases reaches **94.9% accuracy and 92.7%
macro-F1** on a held-out test split of the Kaggle Garbage Classification dataset. A 4-block
CNN trained from scratch reaches 67.5% accuracy. A temperature-calibrated confidence threshold
(tau = 0.55) separates confident, uncertain and unrecognised predictions; the uncertain state
falls back to the safe rule "when in doubt, keep it out of recycling". On TrashNet, a dataset
from a different photographer, accuracy drops to 89.7%. The drop comes mostly from a
label-definition mismatch, which we analyse. The system runs on CPU in about 16 ms per image
(2.5 ms with ONNX Runtime) inside a Streamlit web app with upload, webcam, batch, analytics and
Grad-CAM explanations.

## 2. Motivation and related work

- **TrashNet** (Yang and Thung, 2016) introduced a 6-class, 2.5k-image waste dataset and showed
  that CNNs beat hand-crafted features (SIFT + SVM) for material classification.
- **Transfer learning:** features learned on ImageNet transfer well to small target datasets;
  re-training only the classifier ("feature extraction") or also the top layers
  ("fine-tuning") is standard when labelled data is limited.
- **MobileNetV2** (Sandler et al., 2018) uses inverted residual blocks with depthwise-separable
  convolutions, giving ImageNet-level features at about 3.4M parameters, suitable for CPU
  deployment.
- **Calibration:** modern networks are often miscalibrated; temperature scaling (Guo et al.,
  2017) fixes this with one parameter fitted on validation data.
- **Grad-CAM** (Selvaraju et al., 2017) highlights the image regions that drive a class score.

**[TODO]** Add citations in the department's required format.

Our contribution beyond classification is the **recommendation layer**: a config-driven engine
that maps each class to campus-specific bins and instructions, with a confidence-aware
fallback.

## 3. Dataset, class design and cleaning

**Source.** Kaggle "Garbage Classification" (mostafaabla), 12 folders, 15,515 images.

**Class design.** Nine final classes, defined in `configs/class_map.yaml`:

| Final class | Source folders | Reason |
|---|---|---|
| glass | brown-glass, green-glass, white-glass | Same bin, so colour is irrelevant to the decision |
| textile | clothes, shoes | Same disposal route (donation or dry waste) |
| organic | biological | Renamed to campus wording |
| general | trash | Renamed to campus wording |
| paper, cardboard, plastic, metal, battery | one folder each | |

**De-duplication.** We computed a 64-bit perceptual hash (pHash) of every image and grouped
images within Hamming distance 4 using union-find, before splitting.

| Step | Images |
|---|---|
| Raw | 15,515 |
| Near-duplicate copies removed | 157 (from 162 groups) |
| Removed because copies carried different labels | 62 |
| Clean | 15,296 |

Without this step, a copy of a test image could appear in training and inflate test accuracy.
After splitting, we re-checked every pair: **0 near-duplicate pairs cross the train/val/test
boundary**.

**Split and imbalance.** Stratified 70/15/15 split (seed 42) recorded in
`data/splits/manifest.csv`. Textile has 7,265 clean images against 660 for general, so the
training split is capped at 1,200 images per class (4,045 textile and glass images marked
`unused`). The loss also uses inverse-frequency class weights. Validation and test keep the
natural distribution, so **macro-F1** (the unweighted mean of per-class F1) is the headline
metric; accuracy alone would be dominated by textile.

Figures: `reports/eda/class_counts.png`, `reports/eda/split_counts.png`,
`reports/eda/samples.png`.

## 4. Methodology

**Preprocessing.** Images are resized once so the short side is 288 px. Training uses
RandomResizedCrop(224, scale 0.6-1.0), horizontal flip, rotation up to 20 degrees, colour
jitter and random erasing (p = 0.2). Evaluation uses Resize(256) and CenterCrop(224). All
inputs are normalised with ImageNet statistics.

**Baseline CNN (M0).** Four blocks of Conv3x3 -> BatchNorm -> ReLU -> MaxPool(2) with 32, 64,
128 and 256 filters, then global average pooling, Dropout(0.4) and Linear(256, 9), for 0.39M
parameters. Convolutions share weights across positions, so the network learns local patterns
(edges, textures) wherever they appear. Pooling builds translation tolerance and a growing
receptive field. BatchNorm stabilises training by normalising activations, and ReLU provides
the non-linearity.

**MobileNetV2 (M1).** ImageNet weights, with the classifier replaced by Dropout(0.3) and
Linear(1280, 9).

- *Phase A (head only):* backbone frozen, AdamW, lr 1e-3, 5 epochs. Frozen BatchNorm layers stay
  in eval mode so their ImageNet statistics are not overwritten.
- *Phase B (fine-tune):* the last 5 feature blocks are unfrozen (1.69M trainable parameters), lr
  1e-4 with cosine decay, up to 15 epochs.

Training the new head first prevents large gradients from a randomly initialised classifier
from destroying the pre-trained features. The lower learning rate in Phase B then adapts the
high-level features to waste materials.

**Loss and regularisation.** Softmax cross-entropy with class weights and label smoothing 0.1;
AdamW with weight decay 1e-4; dropout; augmentation; early stopping on validation macro-F1
(patience 4); mixed precision on GPU; batch size 32; seed 42.

> **Deviation from the plan.** With patience 4, the from-scratch CNN stopped at epoch 9 (val
> macro-F1 0.493) while its training loss was still falling, because its validation score is
> noisy at lr 1e-3. We raised its patience to 10. It then trained all 30 epochs and reached
> 0.623. Leaving it under-trained would have exaggerated the benefit of transfer learning.

**Comparisons.** ResNet18 with the same two-phase recipe (M2); MobileNetV2 without augmentation;
MobileNetV2 head-only for 20 epochs.

## 5. Disposal recommendation engine and confidence handling

`src/waste/disposal.py` holds the logic; `configs/disposal_map.json` holds the content (bin
name and colour, preparation steps, do-not list, impact note per class, and the fallback
texts). The map is validated at startup: a missing class or malformed entry raises a clear
error before the app serves anything.

| State | Condition (calibrated top probability p) | Output |
|---|---|---|
| Confident | p >= tau | Full recommendation for the predicted class |
| Uncertain | 0.35 <= p < tau | Top-2 candidates with their bins and the safe-default rule |
| Unrecognised | p < 0.35 | Retake-photo guidance only |

**Calibration.** Label smoothing makes the network under-confident: before calibration its
expected calibration error (ECE) on validation is 0.132. A fitted temperature T = 0.50
(softmax(z / T)) reduces ECE to 0.008, so the probabilities can be read as real likelihoods.

**Choosing tau.** On the validation set we swept tau from 0 to 0.99 and plotted the share of
images answered confidently (coverage) against accuracy on those images. We chose the lowest
tau whose accuracy reaches 97%: **tau = 0.55, covering 97.5% of images at 97.0% accuracy**. The
remaining 2.5% get the uncertain or retake guidance instead of a likely-wrong bin.
Figure: `reports/mobilenet_v2/coverage_curve.png`.

## 6. System design

```
Streamlit app (CPU)
  Classify   upload / webcam / sample -> Predictor -> DisposalEngine -> result card, top-3 chart,
             Grad-CAM toggle, "wrong prediction?" feedback
  Batch      many images or a ZIP -> results table -> CSV
  Analytics  SQLite log -> totals by class and bin, trend, confidence, share uncertain, recyclable share
  Insights   reports/ -> comparison, curves, confusion matrices, threshold, error gallery, EDA
  About      data, model, recommendation logic, limitations, privacy
```

- The model loads once (`st.cache_resource`). Every prediction is logged to SQLite (time,
  class, confidence, state, bin, model). Images are stored only when a user submits a
  correction, in `data/feedback/<correct_class>/`, ready for retraining.
- Screenshots: `docs/screenshots/classify.png`, `docs/screenshots/uncertain_dark.png`,
  `docs/screenshots/insights.png`. **[TODO]** Add a screenshot of the Analytics page after a
  real campus session.

## 7. Experiments and results

### 7.1 Kaggle test split (n = 2,295)

| Run | Accuracy | Macro-F1 | Top-3 | Val macro-F1 | Size | CPU ms |
|---|---|---|---|---|---|---|
| Custom CNN | 67.5% | 62.9% | 91.2% | 62.3% | 1.6 MB | 11 |
| MobileNetV2 head-only | 92.4% | 88.4% | 98.8% | 88.3% | 9.2 MB | 15 |
| **MobileNetV2 fine-tuned** | **94.9%** | **92.7%** | **99.1%** | **94.0%** | **9.2 MB** | **16** |
| MobileNetV2 fine-tuned, no augmentation | 95.9% | 93.4% | 99.2% | 95.4% | 9.2 MB | 15 |
| ResNet18 fine-tuned | 96.0% | 94.0% | 99.3% | 95.1% | 44.8 MB | 22 |

Per-class F1 (MobileNetV2): paper 0.95, cardboard 0.94, plastic 0.83, metal 0.85, glass 0.92,
organic 0.96, battery 0.94, textile 0.98, general 0.97. **Targets met:** accuracy >= 90% and no
class F1 below 0.80.

### 7.2 Ablations

| Question | Result |
|---|---|
| Does transfer learning help? | Yes: +29.8 macro-F1 points over the from-scratch CNN, with similar CPU latency |
| Is fine-tuning worth it? | Yes: +4.3 points over head-only; +16.6 accuracy points on TrashNet |
| Does augmentation help? | Not on these test sets: -0.7 points on Kaggle test, -1.9 accuracy points on TrashNet |
| Is a bigger backbone better? | ResNet18 gains +1.2 points for 4.9x the size and 1.4x the latency |

The augmentation result goes against the plan's expectation, and we report it as measured. Both
the Kaggle and TrashNet images are centred, well-lit single objects. Our augmentations (crops,
rotation, colour jitter, erasing) simulate variation these test sets do not contain, so they
act mainly as extra regularisation on an already well-regularised fine-tune. The campus set,
with clutter and mixed lighting, is the real test of this choice. **[TODO]** Report the campus
numbers for both runs.

### 7.3 Domain shift

| Run | Kaggle test | TrashNet (n = 2,527) | Campus |
|---|---|---|---|
| Custom CNN | 67.5% | 56.9% | **[TODO]** |
| MobileNetV2 fine-tuned | 94.9% | 89.7% | **[TODO]** |
| MobileNetV2, no augmentation | 95.9% | 91.6% | **[TODO]** |
| ResNet18 | 96.0% | 91.0% | **[TODO]** |

On TrashNet, MobileNetV2 scores F1 0.89-0.96 on paper, cardboard, plastic, metal and glass, but
**0.00 on general (137 images)**. The error gallery shows why: TrashNet's "trash" folder holds
plastic pouches, foil wrappers, paper tissues and instant-noodle cups, which the model labels by
material (plastic, metal, paper). TrashNet's "trash" means "not accepted for recycling in that
scheme", while our *general* class was learned from Kaggle's "trash" folder (masks,
toothbrushes, nappies). Part of the measured drop is therefore a label-definition mismatch, not
a vision failure. The uncertain state and the "when in doubt" rule protect against this in the
app.

## 8. Error analysis and limitations

Most common test confusions (MobileNetV2): glass -> plastic (19), glass -> metal (9),
textile -> organic (8), textile -> metal (8), plastic -> glass (7). Grad-CAM
(`reports/mobilenet_v2/eval_test/error_gallery.png`) shows the model attends to caps, labels and
transparency, which clear glass and PET bottles share. Glass -> metal errors often focus on
metal lids. Textile -> organic errors involve leaf-printed fabric and muddy boots.

Limitations:

- One item per photo; there is no detection of several items.
- Training images are clean product-style photos; campus photos will be harder.
- Mixed-material items (laminated cups, chip packets) have no single correct class.
- Unseen categories (e-waste other than batteries, medical waste) are out of scope. The low
  confidence they usually produce triggers the fallback, but this is not guaranteed.

## 9. Conclusion and future work

Transfer learning turns a 67%-accurate from-scratch CNN into a 95%-accurate, 9 MB model that
runs in milliseconds on a laptop CPU. Calibrated confidence states turn its residual errors into
"check with staff" prompts instead of wrong bins. Future work:

- Fine-tune on the campus photos and the corrections collected by the feedback button.
- Detect several items per frame (YOLO-style detection).
- Add a second language for instructions in `disposal_map.json`.
- Mount the camera on bins, using the 2.5 ms ONNX model on edge hardware.

## Appendix: reproducing every number

```bash
make data && make train-baseline && make train && make ablations
make eval && make robustness && make benchmark
```
