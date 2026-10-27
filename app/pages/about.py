"""About page: dataset, model, how the recommendation works, limitations."""

from __future__ import annotations

import common
import pandas as pd
import streamlit as st

from waste.utils import load_json

cfg = common.app_config()
th = cfg["thresholds"]
dmap = load_json(cfg["disposal_map"])

st.title("About this project")
st.markdown(
    """
The **Campus Waste Segregation Assistant** classifies a photo of a single waste item into one of
nine categories and turns that prediction into an action: which bin to use, how to prepare the
item, and what to avoid. It was built for *Neural Network & Deep Learning (PCCCS582)*.

#### Data
- Trained on the public Kaggle **Garbage Classification** dataset (12 folders, ~15k images),
  mapped to 9 classes: the three glass colours are merged into *glass* (same bin) and
  *clothes* + *shoes* become *textile*.
- Exact and near-duplicate images are removed with a perceptual hash **before** splitting, so no
  copy of a test image is seen in training.
- Stratified 70/15/15 split (seed 42). Textile is capped at 1,200 training images and the loss
  uses class weights; macro-F1 is the headline metric.

#### Models
- **Baseline:** a 4-block CNN trained from scratch (Conv-BatchNorm-ReLU-MaxPool).
- **Primary:** MobileNetV2 pre-trained on ImageNet, trained in two phases: head only, then the
  last feature blocks are unfrozen and fine-tuned with a cosine learning rate.
- Training uses label smoothing, augmentation (crop, flip, rotation, colour jitter, erasing),
  mixed precision and early stopping on validation macro-F1.
"""
)

st.markdown(
    f"""
#### How the recommendation works
1. The model outputs a probability for every class.
2. **Confident** (top probability >= {th['tau']:.2f}): the full recommendation for that class.
3. **Uncertain** ({th['reject']:.2f} to {th['tau']:.2f}): the two most likely categories and the
   safe default - *{dmap['fallbacks']['uncertain']['rule']}*
4. **Not recognised** (below {th['reject']:.2f}): advice on retaking the photo.

The threshold was tuned on the validation set using an accuracy-vs-coverage curve. All bins and
instructions live in `configs/disposal_map.json`, so campus staff can change them without
retraining.
"""
)

rows = [{"Class": v["display_name"], "Bin": v["bin"], "Bin label": v["bin_label"],
         "Recyclable": "yes" if v["recyclable"] else "no", "Key instruction": v["steps"][0]}
        for v in dmap["classes"].values()]
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.markdown(
    """
#### Limitations
- **One item per photo.** The model classifies the whole image; with several items it will
  pick one or become uncertain.
- The training photos are mostly clean, well-lit shots. Real campus photos (clutter, poor
  light, dirty or crushed items) are harder - see the campus results on **Model insights**.
- Mixed-material items (a paper cup with a plastic lining, a chip packet) have no single
  correct class; the uncertain state and the *when in doubt* rule cover them.
- The model has never seen e-waste other than batteries, medical waste or hazardous liquids.
  Those should always go to staff.

#### Privacy
Each prediction is logged (time, class, confidence, state, model) for the analytics page.
Photos are **not** stored unless you submit a correction with *Wrong prediction?*.
"""
)
