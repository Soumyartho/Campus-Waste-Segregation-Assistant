"""Analytics page (S4): what has been classified, which bins, how confident, over time."""

from __future__ import annotations

import altair as alt
import common
import streamlit as st

st.title("Waste analytics")
st.caption("Built from every prediction logged by this app (no images are stored).")

store = common.get_store()
df = store.predictions()
if df.empty:
    st.info("No predictions logged yet. Classify a few items on the **Classify** or **Batch** "
            "page and they will appear here.")
    st.stop()

display = {c: c.title() for c in df["predicted_class"].unique()}
try:
    engine = common.get_engine(tuple(common.class_names()))
    display.update(common.display_names(engine))
except Exception:
    pass

f1, f2 = st.columns([2, 1])
first, last = df["ts"].min().date(), df["ts"].max().date()
with f1:
    picked = st.date_input("Date range", value=(first, last), min_value=first, max_value=last)
with f2:
    sources = st.multiselect("Source", sorted(df["source"].unique()),
                             default=sorted(df["source"].unique()))
start, end = (picked if isinstance(picked, tuple) and len(picked) == 2 else (first, last))
mask = (df["ts"].dt.date >= start) & (df["ts"].dt.date <= end) & df["source"].isin(sources)
view = df[mask]
if view.empty:
    st.warning("No predictions match these filters.")
    st.stop()

confident = view[view["state"] == "confident"]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Items classified", f"{len(view):,}")
k2.metric("Average confidence", f"{view['confidence'].mean():.0%}")
k3.metric("Uncertain or unrecognised", f"{(view['state'] != 'confident').mean():.0%}")
k4.metric("Recyclable (confident items)",
          f"{confident['recyclable'].mean():.0%}" if len(confident) else "-",
          help="Share of confidently classified items whose category is recyclable.")

c1, c2 = st.columns(2)
by_class = (view["predicted_class"].map(display).value_counts()
            .rename_axis("class").reset_index(name="items"))
c1.altair_chart(common.bar_chart(by_class, "class", "items", "Items by class"), width="stretch")
by_bin = (confident["bin"].fillna("unknown").value_counts()
          .rename_axis("bin").reset_index(name="items"))
c2.altair_chart(common.bar_chart(by_bin, "bin", "items", "Items by bin (confident only)"),
                width="stretch")

span_days = (view["ts"].max() - view["ts"].min()).days
freq, label = ("D", "Day") if span_days >= 2 else ("h", "Hour")
trend = (view.set_index("ts").resample(freq).size().rename("items").reset_index())
hover = alt.selection_point(fields=["ts"], nearest=True, on="pointerover", empty=False)
base = alt.Chart(trend).encode(x=alt.X("ts:T", title=label),
                               y=alt.Y("items:Q", title="Items"))
line = base.mark_line(color=common.ACCENT, strokeWidth=2)
points = base.mark_circle(size=60, color=common.ACCENT).encode(
    opacity=alt.condition(hover, alt.value(1), alt.value(0)),
    tooltip=[alt.Tooltip("ts:T", title=label), alt.Tooltip("items:Q", title="Items")],
).add_params(hover)
st.altair_chart((line + points).properties(title="Items classified over time", height=240),
                width="stretch")

c3, c4 = st.columns(2)
states = view["state"].value_counts().rename_axis("state").reset_index(name="items")
c3.altair_chart(common.bar_chart(states, "state", "items", "Confidence states"), width="stretch")
avg_conf = (view.assign(cls=view["predicted_class"].map(display))
            .groupby("cls")["confidence"].mean().rename("confidence").reset_index()
            .rename(columns={"cls": "class"}))
c4.altair_chart(common.bar_chart(avg_conf, "class", "confidence", "Average confidence by class",
                                 fmt=".0%"), width="stretch")

fb = store.feedback()
st.subheader("User corrections")
if fb.empty:
    st.caption("No corrections submitted yet.")
else:
    pairs = (fb.assign(predicted=fb["predicted_class"].map(lambda c: display.get(c, c)),
                       correct=fb["correct_class"].map(lambda c: display.get(c, c)))
             .groupby(["predicted", "correct"]).size().rename("count").reset_index()
             .sort_values("count", ascending=False))
    st.metric("Corrections stored for retraining", len(fb))
    st.dataframe(pairs, hide_index=True, width="stretch")

with st.expander("Raw prediction log"):
    st.dataframe(view.sort_values("ts", ascending=False), hide_index=True, width="stretch")
    st.download_button("Download log as CSV", view.to_csv(index=False).encode("utf-8"),
                       file_name="prediction_log.csv", mime="text/csv")
