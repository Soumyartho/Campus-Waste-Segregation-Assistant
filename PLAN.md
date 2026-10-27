# Campus Waste Segregation Assistant: Implementation Plan

*Neural Network & Deep Learning (PCCCS582) · End-term Project #5*

## 1. How to use this document

`PLAN.md` is the source of truth. `CLAUDE.md` holds the short working rules from Section 13.
Build one phase at a time (Section 12): implement only the named phase, run the tests, and
stop with a summary.

## 2. Project at a glance

The system takes a photo of a waste item (file upload or webcam snapshot), classifies it with
a CNN, and then tells the user which bin it belongs in and how to prepare it before disposal.
The second step is the novelty required by the brief: a recommendation, not just a label.

| PDF field | What the brief says | How this project satisfies it |
|---|---|---|
| Project title | Campus Waste Segregation Assistant | Same |
| Technology / Model | CNN/YOLO | Primary: CNN with transfer learning (MobileNetV2). Baseline: a custom CNN trained from scratch. Optional comparison: YOLOv8n-cls |
| What students build | Classifies waste as paper, plastic, metal, organic, etc. from a camera/image | 9-class classifier (paper, cardboard, plastic, metal, glass, organic, battery, textile, general) with image upload and webcam input in a web app |
| Novelty | Adds disposal recommendation rather than only classification | Config-driven disposal engine with bin colour, preparation steps, do/don't tips and a confidence-aware fallback, plus a waste analytics dashboard |

## 3. Goals and scope

**Goals**

- A trained and evaluated image classifier, with a clear comparison between a from-scratch CNN and transfer learning.
- A disposal recommendation layer that turns every prediction into an action.
- A demo app that a reviewer can run in two commands.
- Honest evaluation, including real photos taken on campus.

**Out of scope**

- Bounding-box detection of several items in one frame.
- Mobile app, IoT bin hardware, user accounts.
- Training on private or scraped data.

## 4. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | Standard for deep learning |
| Deep learning | PyTorch + torchvision | Pretrained MobileNetV2 and ResNet18, simple training loops |
| Optional YOLO | ultralytics (YOLOv8n-cls) | Satisfies the 'CNN/YOLO' line as a comparison model |
| Data and metrics | NumPy, pandas, scikit-learn | Splits, classification report, confusion matrix |
| Images | Pillow, imagehash | Loading, perceptual-hash de-duplication |
| Plots | matplotlib, seaborn | Curves, confusion matrix, sample grids |
| Explainability | pytorch-grad-cam | Shows what the CNN looked at |
| Config | PyYAML + argparse | No hard-coded paths or hyperparameters |
| App | Streamlit (`st.file_uploader`, `st.camera_input`) | Fastest route to upload + webcam UI |
| Logging store | SQLite (stdlib `sqlite3`) | Zero setup, easy analytics queries |
| Export (optional) | ONNX + onnxruntime | Fast CPU inference and latency benchmark |
| Quality | pytest, ruff | Tests and linting |
| Compute | Colab or Kaggle free GPU for training, CPU for the app | Fine-tuning is slow on CPU only |
| Dev tools | Git, Antigravity IDE, Claude Code | As planned |

## 5. Dataset and class design

### 5.1 Datasets

- **Primary:** Kaggle 'Garbage Classification' (12 folders, roughly 15k images): battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass. Verify folder names after download because mirrors differ.
- **External test set (optional):** TrashNet (about 2.5k images, 6 classes). Train on Kaggle, test on TrashNet to measure generalisation to a different photographer and background.
- **Campus test set (self-collected):** 15 to 20 phone photos per class taken around the campus (bins, canteen, corridors) with real backgrounds and mixed lighting. Used for final testing only, never for training or tuning.

### 5.2 Final classes (9)

| Final class | Source folders |
|---|---|
| paper | paper |
| cardboard | cardboard |
| plastic | plastic |
| metal | metal |
| glass | brown-glass + green-glass + white-glass (same disposal, so merged) |
| organic | biological |
| battery | battery |
| textile | clothes + shoes |
| general | trash |

The mapping lives in `configs/class_map.yaml`, so classes can be merged or dropped (for
example removing textile) without touching code.

### 5.3 Data rules

1. Remove exact and near duplicates with a perceptual hash before splitting. Otherwise copies leak across train and test and inflate accuracy.
2. Stratified 70/15/15 split with seed 42, saved to `data/splits/manifest.csv` (path, label, split). Every script loads from the manifest.
3. The textile classes are far larger than the rest. Cap each class at `max_per_class` (default 1200) in the training split and also use class weights in the loss. Validation and test keep the natural distribution, so macro-F1 is the headline metric.
4. The test split and campus set are never used for any decision (early stopping, threshold, model choice). Use validation.
5. Save dataset statistics (counts before and after cleaning, sample grid per class) to `reports/eda/`.

## 6. Feature list

| ID | Feature | Priority |
|---|---|---|
| M1 | Data pipeline: class mapping, de-duplication, stratified splits, manifest | Must |
| M2 | Augmentation pipeline (train only) | Must |
| M3 | Baseline custom CNN trained from scratch | Must |
| M4 | MobileNetV2 transfer learning, two-phase fine-tuning | Must |
| M5 | Evaluation suite: accuracy, macro-F1, per-class report, confusion matrix, training curves | Must |
| M6 | Inference module: top-k, confidence threshold, 'uncertain' state | Must |
| M7 | Disposal recommendation engine driven by `disposal_map.json` | Must |
| M8 | Streamlit app: upload + webcam, shows class, confidence, bin, instructions | Must |
| M9 | README, reproducible commands, unit tests | Must |
| S1 | Grad-CAM heatmap toggle in the app and an error gallery in the report | Should |
| S2 | Model comparison table and ablations (Section 8) | Should |
| S3 | Batch mode: many images or a ZIP, results table, CSV download | Should |
| S4 | Prediction logging and analytics dashboard (counts by class and bin, trend over time, share uncertain) | Should |
| S5 | User feedback: 'wrong prediction' button to pick the correct class, stored for later retraining | Should |
| S6 | Campus test set and cross-dataset (TrashNet) evaluation | Should |
| S7 | Accuracy-vs-coverage curve and threshold tuning on validation | Should |
| C1 | YOLOv8n-cls as a third comparison model | Could |
| C2 | Temperature-scaling calibration and ECE | Could |
| C3 | ONNX export and CPU latency benchmark | Could |
| C4 | Live webcam stream with `streamlit-webrtc` | Could |
| C5 | Second language for disposal instructions via the JSON map | Could |

Build all Must items first. Add Should items in order. Treat Could items as extras if time remains.

## 7. Model design and training

### 7.1 Preprocessing

- Train: RandomResizedCrop(224, scale 0.6 to 1.0), horizontal flip, rotation up to 20 degrees, ColorJitter (brightness 0.3, contrast 0.3, saturation 0.3, hue 0.05), RandomErasing p=0.2.
- Validation, test and inference: Resize(256), CenterCrop(224).
- Both: ImageNet mean and std normalisation.

### 7.2 Models

| ID | Model | Role |
|---|---|---|
| M0 | Custom CNN: 4 blocks (32, 64, 128, 256 filters), each Conv3x3, BatchNorm, ReLU, MaxPool; global average pooling; Dropout 0.4; Linear(256, 9) | Baseline that shows the fundamentals |
| M1 | MobileNetV2 with ImageNet weights, classifier replaced by Dropout(0.3) + Linear(1280, 9) | Primary model |
| M2 | ResNet18 with ImageNet weights (optional) | Second transfer-learning comparison |
| M3 | YOLOv8n-cls (optional) | YOLO-family comparison |

### 7.3 Training recipe

**M1 (MobileNetV2)**

- Phase A, head only: backbone frozen, AdamW, lr 1e-3, 5 epochs.
- Phase B, fine-tune: unfreeze the last few feature blocks, lr 1e-4, cosine decay, up to 15 epochs.
- Early stopping on validation macro-F1 with patience 4.

**M0 (custom CNN)**

- From scratch, AdamW, lr 1e-3, up to 30 epochs, same early stopping.

**Shared settings**

- Batch size 32, weight decay 1e-4, CrossEntropy with label smoothing 0.1 and class weights, mixed precision on GPU, seed 42.
- The best checkpoint stores weights, class names, config and the git commit hash in `models/<run_name>/best.pt`.
- Logs go to `reports/<run_name>/history.csv` and curve plots.

### 7.4 Confidence and rejection

Compute softmax probabilities and return the top 3. Three states, all thresholds set in
`configs/inference.yaml`:

- **Confident:** top probability at or above tau (default 0.60). Show the full recommendation.
- **Uncertain:** between 0.35 and tau. Show the top 2 candidates and the safe-default advice.
- **Unrecognised:** below 0.35. Ask the user to retake the photo with a plain background and the item filling the frame.

Tune tau on the validation set using the accuracy-vs-coverage curve (S7).

## 8. Evaluation plan

**Metrics for every model**

- Test accuracy, macro-F1, top-3 accuracy.
- Per-class precision, recall and F1 (`classification_report`).
- Confusion matrix (counts and row-normalised).
- Training and validation loss and accuracy curves.
- Model size in MB and CPU latency per image in ms.

**Experiments**

| Experiment | Question answered |
|---|---|
| M0 vs M1 (and M2, M3 if built) | How much does transfer learning help? |
| M1 with and without augmentation | How much does augmentation help? |
| M1 head-only vs fine-tuned | Is fine-tuning worth the extra training? |
| Test split vs TrashNet vs campus set | How much accuracy is lost under domain shift? |
| Accuracy vs coverage across tau | What does the rejection threshold cost and buy? |

**Targets** (goals, not guarantees)

- M1 test accuracy of at least 90% and no class F1 below 0.80 on the Kaggle test split.
- Report campus-set accuracy as measured, even if it is lower. A clearly explained gap scores better than a hidden one.
- Error analysis: show the 5 most common confusions (likely glass vs plastic, paper vs cardboard, general vs plastic) with Grad-CAM examples.

## 9. Disposal recommendation engine

A pure-Python module, `src/waste/disposal.py`, that reads `configs/disposal_map.json` and
returns a structured recommendation for any predicted class. Logic lives in code, content
lives in JSON, so campus staff can edit bins and instructions without touching the model.

**Schema (one entry shown)**

```json
{
  "plastic": {
    "display_name": "Plastic",
    "bin": "Blue",
    "bin_label": "Dry recyclables",
    "recyclable": true,
    "steps": ["Empty and rinse the container", "Remove the cap and label if possible", "Flatten bottles"],
    "do_not": ["Do not put food-soiled plastic in the recycling bin"],
    "impact_note": "Clean plastic can be recycled instead of going to landfill"
  }
}
```

**Default mapping** (colours are configurable because campus bin schemes differ)

| Class | Default bin | Key instruction |
|---|---|---|
| paper | Blue, dry recyclables | Keep dry and clean, no greasy paper |
| cardboard | Blue, dry recyclables | Flatten boxes, remove tape if possible |
| plastic | Blue, dry recyclables | Empty, rinse, flatten |
| metal | Blue, dry recyclables | Empty and rinse cans, do not crush sharp edges |
| glass | Yellow, glass | Empty and rinse, handle broken glass carefully, wrap it first |
| organic | Green, wet waste | Food scraps and peels only, no plastic bags or wrappers |
| battery | Red, hazardous / e-waste | Never in general waste, use the e-waste collection point |
| textile | Donation or dry-waste bin | Donate if usable, otherwise dry waste |
| general | Black, general waste | Last resort when the item cannot be recycled |

**Engine behaviour**

- Confident: return the full entry plus the predicted class and probability.
- Uncertain: return the top 2 entries and the rule 'when in doubt, keep it out of recycling and ask a staff member'.
- Unrecognised: return only the retake-photo guidance.
- Validate the JSON at startup: every class in the model must have an entry, and a missing entry must raise a clear error.

## 10. Web app specification (Streamlit)

| Page | Contents |
|---|---|
| Classify | Image upload or webcam snapshot. Result card with predicted class, confidence bar, coloured bin badge, preparation steps and do-not list. Top-3 probability chart. Grad-CAM toggle. 'Wrong prediction?' feedback control |
| Batch | Upload many images or a ZIP. Results table with class, confidence, bin and state. CSV download |
| Analytics | Totals by class and by bin, trend over time, average confidence, share of uncertain predictions, estimated share of items that were recyclable |
| Model insights | Test metrics, confusion matrix, model comparison table, threshold curve, all loaded from `reports/` |
| About | Dataset, model, limitations, how the recommendation works |

**App rules**

- Load the model once with `st.cache_resource`.
- Log every prediction (timestamp, class, confidence, state, model name) to SQLite at `data/logs/app.db`. Do not store the uploaded image unless the user submits feedback.
- The app must run on CPU only and start with `streamlit run app/streamlit_app.py`.
- Show a clear message when the checkpoint is missing instead of crashing.

## 11. Repository structure

```
waste-segregation-assistant/
  PLAN.md  CLAUDE.md  README.md  requirements.txt  Makefile
  configs/
    data.yaml  class_map.yaml  train_custom_cnn.yaml  train_mobilenet.yaml
    inference.yaml  disposal_map.json
  data/                 (git-ignored)
    raw/ processed/ splits/ campus_test/ logs/
  models/               (git-ignored, keep best.pt files here)
  reports/              (metrics, figures, EDA, comparison tables)
  src/waste/
    data/    prepare.py dedupe.py split.py dataset.py transforms.py
    models/  custom_cnn.py mobilenet.py factory.py
    train.py evaluate.py predict.py disposal.py explain.py utils.py
  app/
    streamlit_app.py
    pages/ batch.py analytics.py insights.py about.py
  scripts/ download_data.py export_onnx.py benchmark.py
  tests/   test_data.py test_models.py test_disposal.py test_predict.py
  notebooks/ colab_train.ipynb
```

## 12. Build phases

| Phase | Work | Done when |
|---|---|---|
| 0. Scaffold | Folder structure, `requirements.txt`, `.gitignore`, `Makefile`, config stubs, `CLAUDE.md`, an empty test that passes | `pytest` runs green and the repo installs from a clean venv |
| 1. Data | `download_data.py` (Kaggle API if credentials exist, otherwise print manual steps), class mapping, perceptual-hash de-duplication, stratified split, manifest, `Dataset` and transforms, EDA plots | Manifest exists, class counts printed, no duplicate leakage across splits, tests pass |
| 2. Baseline | Custom CNN, generic `train.py` with early stopping, history CSV, curves | Baseline trains end to end and saves `best.pt` and curves |
| 3. Transfer learning | MobileNetV2 two-phase training from the config, class weights, label smoothing, AMP | Validation macro-F1 clearly above the baseline, checkpoint saved with metadata |
| 4. Evaluation | `evaluate.py`: metrics, per-class report, confusion matrix, comparison table, ablation runs | `reports/` holds all figures and a `comparison.md` table |
| 5. Inference + disposal | `predict.py` (top-k, three states), `disposal.py`, full `disposal_map.json`, threshold tuning and coverage curve | Unit tests cover all three states and a missing-class error |
| 6. App | Streamlit Classify page with upload and webcam, result card, SQLite logging | App runs on CPU and a screenshot shows a full recommendation |
| 7. Extras | Grad-CAM, Batch page, Analytics page, Model insights page, feedback control | Each page loads without errors and uses real logged data |
| 8. Robustness | Campus test set evaluation, TrashNet cross-dataset test, error gallery, optional ResNet18 or YOLOv8n-cls | Results table has Kaggle, TrashNet and campus columns |
| 9. Polish | README with setup, commands, results and screenshots, demo script, final cleanup | A fresh clone can reproduce the app demo from the README |

If there is no local GPU, run Phases 2 to 4 in a Colab or Kaggle notebook that clones the
repo, then download `best.pt` into `models/`. The code must stay runnable on CPU for the app.

## 13. Content for CLAUDE.md

See `CLAUDE.md`.

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Dataset images are clean studio shots, real photos are messier | Campus test set, strong augmentation, honest reporting of the gap |
| Near-duplicate images inflate test accuracy | Perceptual-hash de-duplication before splitting |
| Textile class dominates | Per-class cap, class weights, macro-F1 as the main metric |
| Glass, plastic and general waste get confused | Merge glass colours, show Grad-CAM and top-2 candidates, use the uncertain state |
| Photo contains several items | State the single-item limitation in the app and the report |
| No GPU available | Train on Colab or Kaggle, keep MobileNetV2 as the primary model |
| Kaggle download needs credentials | Script prints manual download steps as a fallback |
| Streamlit reruns reset state | Use `st.cache_resource` and `st.session_state` |
| Bin colours differ on the real campus | All colours and wording live in `disposal_map.json` |

## 15. Report outline and viva preparation

**Report structure**

1. Abstract and problem statement.
2. Motivation and related work (TrashNet, transfer learning, MobileNet).
3. Dataset, class design and cleaning decisions.
4. Methodology: convolution, pooling, batch normalisation, ReLU, softmax, cross-entropy, transfer learning, fine-tuning, regularisation.
5. Disposal recommendation engine and confidence handling.
6. System design and app screenshots.
7. Experiments and results: all tables and figures from Section 8.
8. Error analysis and limitations.
9. Conclusion and future work (multi-item detection, more campus data, bin hardware).

**Likely viva questions to prepare**

- Why transfer learning instead of training from scratch, and what did your baseline show?
- Why freeze the backbone first, then unfreeze?
- Why macro-F1 and not only accuracy?
- Why merge the glass classes, and what leakage did de-duplication prevent?
- How does the confidence threshold work and how did you choose it?
- What happens on an item the model has never seen?
- Where does the model fail and why?

## 16. Definition of done

- [ ] All Must features (M1 to M9) implemented and tested.
- [ ] Baseline and MobileNetV2 results in `reports/comparison.md`.
- [ ] Confusion matrix, per-class report and training curves saved.
- [ ] Disposal map complete for all 9 classes and validated at startup.
- [ ] App runs on CPU with upload and webcam input.
- [ ] At least four Should items (S1 to S7) completed, including the campus test set.
- [ ] README lets a fresh clone reproduce the demo.
- [ ] Report drafted from the Section 15 outline.
