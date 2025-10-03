import time, uuid, json, pathlib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Moral Medicine — PoC", page_icon="⚖️", layout="centered")

# ---------- Config ----------
RESP_PATH = pathlib.Path("responses.csv")

VIGNETTES = [
    {
        "id":"v1",
        "prompt":"One ICU bed is available. Who should be prioritized?",
        "A":{"age":"18–35","n":1,"cost":"£50k","p":0.7,"qol":"good"},
        "B":{"age":"65–80","n":2,"cost":"£200k","p":0.5,"qol":"fair"},
        "tags":["cost_vs_count","age_vs_qol"], "check": False
    },
    {
        "id":"v2",
        "prompt":"Funding decision for today’s list. Choose where to allocate.",
        "A":{"age":"0–12","n":1,"cost":"£10k","p":0.3,"qol":"excellent"},
        "B":{"age":"65–80","n":1,"cost":"£1k","p":0.7,"qol":"fair"},
        "tags":["young_vs_old","qol_vs_success"], "check": False
    },
    {
        "id":"v3",  # dominance check: A strictly better on all attributes
        "prompt":"One theatre slot remains. Who goes ahead?",
        "A":{"age":"18–35","n":2,"cost":"£10k","p":0.9,"qol":"excellent"},
        "B":{"age":"18–35","n":1,"cost":"£200k","p":0.1,"qol":"poor"},
        "tags":["attention_check"], "check": True
    },
    {
        "id":"v4",
        "prompt":"Only one treatment course is available.",
        "A":{"age":"36–64","n":1,"cost":"£50k","p":0.7,"qol":"good"},
        "B":{"age":"80+","n":5,"cost":"£200k","p":0.5,"qol":"fair"},
        "tags":["many_vs_few","age"], "check": False
    },
]

QOL_TO_UTIL = {"poor":0.2, "fair":0.5, "good":0.75, "excellent":0.95}

def cost_to_num(cost_str: str) -> float:
    """£50k -> 50000; £1k -> 1000; £200k -> 200000"""
    s = cost_str.replace("£","").lower().replace(",","").strip()
    if s.endswith("k"):
        return float(s[:-1]) * 1000
    return float(s)

# ---------- Init session ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "order" not in st.session_state:
    st.session_state.order = np.random.permutation(len(VIGNETTES)).tolist()
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "t0" not in st.session_state:
    st.session_state.t0 = time.time()
if "consented" not in st.session_state:
    st.session_state.consented = False

st.title("⚖️ Moral Medicine — tiny demo")

with st.expander("What is this?"):
    st.write(
        "- A quick demo to crowd-source intuitions about medical trade-offs (like cheap vs expensive, young vs old, many vs few, survival vs quality of life). "
        "No medical advice; anonymous; you can skip any question."
    )

# ---------- Sidebar nav ----------
view = st.sidebar.radio("View", ["Take part", "Results"], index=0)

# ---------- Consent ----------
if not st.session_state.consented:
    st.info("Please read and confirm to continue.")
    consent = st.checkbox("I understand this is an anonymous demo for research/education, and I can stop anytime.", value=False)
    if st.button("Continue"):
        st.session_state.consented = bool(consent)
        st.session_state.t0 = time.time()
    if not st.session_state.consented:
        st.stop()

# ---------- Helpers ----------
def append_response(row: dict):
    df = pd.DataFrame([row])
    if RESP_PATH.exists():
        df.to_csv(RESP_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(RESP_PATH, index=False)

def to_delta(A, B):
    return {
        "d_cost": cost_to_num(B["cost"]) - cost_to_num(A["cost"]),
        "d_p": B["p"] - A["p"],
        "d_qol": QOL_TO_UTIL[B["qol"]] - QOL_TO_UTIL[A["qol"]],
        "d_count": B["n"] - A["n"],
    }

# ---------- Take part ----------
if view == "Take part":
    if st.session_state.idx >= len(VIGNETTES):
        st.success("Thanks! You’ve completed the demo.")
        st.button("Restart", on_click=lambda: (st.session_state.update(idx=0, order=np.random.permutation(len(VIGNETTES)).tolist())))
        st.stop()

    v = VIGNETTES[st.session_state.order[st.session_state.idx]]

    st.subheader(v["prompt"])
    colA, colB = st.columns(2, gap="large")

    def show_option(label, opt):
        with label:
            st.markdown(f"### Option {label.container}_LABEL")
    # Pretty cards
    with colA:
        st.markdown("#### Option A")
        st.metric("Age", v["A"]["age"])
        st.metric("People affected", v["A"]["n"])
        st.metric("Cost", v["A"]["cost"])
        st.metric("Chance of success", f"{int(v['A']['p']*100)}%")
        st.metric("Quality of life after", v["A"]["qol"].title())
        choose_A = st.button("Choose A", use_container_width=True)

    with colB:
        st.markdown("#### Option B")
        st.metric("Age", v["B"]["age"])
        st.metric("People affected", v["B"]["n"])
        st.metric("Cost", v["B"]["cost"])
        st.metric("Chance of success", f"{int(v['B']['p']*100)}%")
        st.metric("Quality of life after", v["B"]["qol"].title())
        choose_B = st.button("Choose B", use_container_width=True)

    skip = st.link_button("Skip", "#", help="Move on without answering")

    # Handle input
    picked = None
    if choose_A: picked = "A"
    if choose_B: picked = "B"
    if skip: picked = "skip"

    if picked:
        rt_ms = int((time.time() - st.session_state.t0) * 1000)
        deltas = to_delta(v["A"], v["B"])
        row = {
            "session_id": st.session_state.session_id,
            "ts": pd.Timestamp.utcnow().isoformat(),
            "vignette_id": v["id"],
            "choice": picked,
            "rt_ms": rt_ms,
            "is_check_item": v["check"],
            "tags": json.dumps(v["tags"]),
            # store what was shown + deltas (for later modeling)
            "A": json.dumps(v["A"]),
            "B": json.dumps(v["B"]),
            **deltas
        }
        append_response(row)

        # gentle UX nudge for very fast clicks
        if picked != "skip" and rt_ms < 800:
            st.warning("That was quick — thanks! Take your time if you like; these are tricky.")

        st.session_state.idx += 1
        st.session_state.t0 = time.time()
        st.rerun()

# ---------- Results ----------
else:
    st.subheader("Live results (local)")
    if not RESP_PATH.exists():
        st.info("No responses yet. Take the survey first!")
        st.stop()

    df = pd.read_csv(RESP_PATH)
    st.caption(f"{len(df)} responses logged")

    # Basic aggregates
    agg = (
        df.assign(is_A=lambda x: (x["choice"]=="A").astype(int),
                  is_B=lambda x: (x["choice"]=="B").astype(int))
          .groupby("vignette_id")[["is_A","is_B"]]
          .sum()
          .assign(total=lambda x: x["is_A"]+x["is_B"])
          .assign(pct_B=lambda x: (x["is_B"]/x["total"]).fillna(0))
          .reset_index()
    )
    st.dataframe(agg[["vignette_id","is_A","is_B","total","pct_B"]], use_container_width=True)

    # Tiny logistic model: predict choosing B from deltas
    model_area = st.container()
    valid = df[df["choice"].isin(["A","B"])].copy()
    if len(valid) >= 10:
        X = valid[["d_cost","d_p","d_qol","d_count"]].values
        y = (valid["choice"]=="B").astype(int).values
        try:
            lr = LogisticRegression(max_iter=200).fit(X, y)
            coef = pd.Series(lr.coef_[0], index=["Δcost","Δsuccess","ΔQoL","Δcount"])
            model_area.markdown("#### Preference signals (logistic regression coefficients)")
            model_area.write(coef.sort_values(ascending=False).to_frame("weight"))
            model_area.caption("Positive weight ⇒ pushes choice toward Option B when B exceeds A on that attribute.")
        except Exception as e:
            st.warning(f"Model didn’t converge yet: {e}")
    else:
        st.info("Need at least ~10 non-skipped responses for the tiny model.")

    with st.expander("Privacy & notes"):
        st.write(
            "- Data is stored **locally** in `responses.csv` for this demo. "
            "For a real deployment, connect a database (see below).\n"
            "- The ‘attention_check’ vignette helps flag random clicking."
        )