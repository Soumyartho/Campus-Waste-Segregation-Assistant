"""Classify page: upload or webcam snapshot -> prediction -> disposal recommendation."""

from __future__ import annotations

import common
import streamlit as st

from waste.predict import State

st.title("What bin does this go in?")
st.caption(
    "Photograph **one item** on a plain background. The model names the material and tells "
    "you which bin to use and how to prepare it."
)

predictor, engine = common.require_model()
cfg = common.app_config()
store = common.get_store()
names = common.display_names(engine)

upload_tab, camera_tab, sample_tab = st.tabs(["Upload a photo", "Use webcam", "Try a sample"])
with upload_tab:
    uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp", "bmp"],
                                label_visibility="collapsed")
with camera_tab:
    snapshot = st.camera_input("Take a photo", label_visibility="collapsed")
with sample_tab:
    samples = common.sample_images()
    options = ["(none)", *samples]
    requested = st.query_params.get("sample")  # ?sample=<name> preselects, handy for demos
    sample = st.selectbox("Sample photo", options,
                          index=options.index(requested) if requested in options else 0)
    if len(options) == 1:
        st.caption("Put photos in `app/samples/` to list them here.")

if snapshot is not None:
    source, data = "webcam", snapshot.getvalue()
elif uploaded is not None:
    source, data = "upload", uploaded.getvalue()
elif sample != "(none)":
    source, data = "sample", samples[sample].read_bytes()
else:
    st.info("Upload an image, take a webcam snapshot or pick a sample to get a recommendation.")
    st.stop()

try:
    image = common.open_image(data)
except Exception:
    st.error("That file could not be read as an image. Please try a JPG or PNG.")
    st.stop()

key = common.image_key(data)
if st.session_state.get("current_key") != key:
    pred = predictor.predict(image)
    rec = engine.recommend(pred)
    primary = rec.primary
    pid = store.log_prediction(
        source, pred.label, pred.confidence, pred.state.value,
        primary.bin if primary else None, primary.recyclable if primary else None,
        predictor.model_name,
    )
    st.session_state.update(current_key=key, pred=pred, rec=rec, pid=pid, cam=None,
                            feedback_sent=False)
pred, rec = st.session_state["pred"], st.session_state["rec"]

left, right = st.columns([1, 1.15], gap="large")
with left:
    show_cam = st.toggle("Show Grad-CAM (what the model looked at)", value=False)
    if show_cam:
        if st.session_state.get("cam") is None:
            from waste.explain import gradcam_overlay

            with st.spinner("Computing Grad-CAM..."):
                target = predictor.class_names.index(pred.label)
                st.session_state["cam"] = gradcam_overlay(
                    predictor.model, predictor.meta["model_name"], image, target
                )
        c1, c2 = st.columns(2)
        c1.image(image, caption="Your photo", width="stretch")
        c2.image(st.session_state["cam"], caption=f"Evidence for '{names[pred.label]}'",
                 width="stretch")
    else:
        st.image(image, width="stretch")

    st.markdown("**Top predictions**")
    st.altair_chart(common.probability_chart(pred, names), width="stretch")

with right:
    common.render_recommendation(pred, rec)

    with st.expander("Wrong prediction? Tell us the right category"):
        if st.session_state.get("feedback_sent"):
            st.success("Thanks! Your correction was saved for future retraining.")
        else:
            options = predictor.class_names
            default = options.index(pred.top_k[1][0]) if pred.state != State.CONFIDENT else 0
            correct = st.selectbox("Correct category", options, index=default,
                                   format_func=lambda c: names[c])
            st.caption("Submitting saves this photo so it can be reviewed and used to "
                       "improve the model. Nothing is stored otherwise.")
            if st.button("Submit correction", type="primary"):
                store.log_feedback(st.session_state["pid"], pred.label, correct, image,
                                   cfg["app"]["feedback_dir"])
                st.session_state["feedback_sent"] = True
                st.rerun()
