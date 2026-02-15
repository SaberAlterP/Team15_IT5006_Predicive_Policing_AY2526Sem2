import inspect
import math
from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------
# 1) Page Configuration & UI Theme
# -----------------------------
st.set_page_config(page_title="Chicago Crime Dashboard", layout="wide")
px.defaults.template = "plotly_white"

_FORM_SUBMIT_SIG = inspect.signature(st.form_submit_button).parameters
_COL_SIG = inspect.signature(st.columns).parameters


def form_submit(label, **kwargs):
    """Compatibility wrapper for st.form_submit_button (older versions may not support type=)."""
    if "type" not in _FORM_SUBMIT_SIG:
        kwargs.pop("type", None)
    return st.form_submit_button(label, **kwargs)


def cols(spec, **kwargs):
    """Compatibility wrapper for st.columns (older versions may not support vertical_alignment=)."""
    if "vertical_alignment" not in _COL_SIG and "vertical_alignment" in kwargs:
        kwargs.pop("vertical_alignment", None)
    return st.columns(spec, **kwargs)


# -----------------------------
# 2) CSS (Sidebar min width + sidebar card + multiselect tags + Apply button visible + Loading overlay)
# -----------------------------
st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; }

/* Sidebar min width */
:root{ --sidebar-min-width: 360px; }
section[data-testid="stSidebar"]{
  min-width: var(--sidebar-min-width) !important;
  width: max(var(--sidebar-min-width), 21rem) !important;
}
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div:first-child,
section[data-testid="stSidebar"] div[data-testid="stSidebarContent"]{
  min-width: var(--sidebar-min-width) !important;
}
div[data-testid="stAppViewContainer"] > div{ min-width: 0; }

/* Sidebar theme */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #fbfcfe 0%, #f6f7fb 100%);
  border-right: 1px solid #e7e9f2;
}
section[data-testid="stSidebar"] .block-container{
  padding-top: 0.6rem;
  padding-bottom: 0.8rem;
}

section[data-testid="stSidebar"] div[data-testid="stForm"],
section[data-testid="stSidebar"] form{
  background: #ffffff;
  border: 1px solid #e7e9f2;
  border-radius: 16px;
  padding: 16px 14px;
  box-shadow: 0 8px 22px rgba(17, 24, 39, 0.06);
}

section[data-testid="stSidebar"] h3{
  margin: 0.2rem 0 0.7rem 0;
  letter-spacing: -0.02em;
  font-size: 1.35rem;
  font-weight: 800;
  color: #111827;
}

section[data-testid="stSidebar"] label{
  font-weight: 650 !important;
  color: #111827 !important;
}

section[data-testid="stSidebar"] [data-baseweb="select"] > div{
  border-radius: 12px !important;
  border-color: #e5e7eb !important;
  background: #ffffff !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within{
  border-color: #f08b8b !important;
  box-shadow: 0 0 0 3px rgba(240, 139, 139, 0.25) !important;
}

section[data-testid="stSidebar"] [data-baseweb="tag"]{
  background: #e45b5b !important;
  color: #ffffff !important;
  border-radius: 10px !important;
  border: 1px solid rgba(0,0,0,0.05) !important;
  font-weight: 700 !important;
}
section[data-testid="stSidebar"] [data-baseweb="tag"] span{ color:#fff !important; }
section[data-testid="stSidebar"] [data-baseweb="tag"] svg{ fill:#fff !important; }

section[data-testid="stSidebar"] button{
  border-radius: 12px !important;
  padding: 0.55rem 0.85rem !important;
  font-weight: 750 !important;
  border: 1px solid #e5e7eb !important;
  background: #ffffff !important;
}

section[data-testid="stSidebar"] hr{
  margin: 0.8rem 0 0.9rem 0;
  border: none;
  border-top: 1px solid #e7e9f2;
}

section[data-testid="stSidebar"] .btn-right{
  display: flex;
  justify-content: flex-end;
  align-items: flex-end;
  width: 100%;
}

section[data-testid="stSidebar"] .btn-right div[data-testid="stFormSubmitButton"]{
  width: auto !important;
  flex: 0 0 auto !important;
}

section[data-testid="stSidebar"] .btn-right div[data-testid="stFormSubmitButton"] > button{
  width: 112px !important;
  height: 44px !important;
  padding: 0 !important;
  line-height: 44px !important;
  white-space: nowrap !important;
  box-sizing: border-box !important;
  flex: 0 0 auto !important;
}

section[data-testid="stSidebar"] form div[data-testid="stFormSubmitButton"]:last-of-type button{
  background: linear-gradient(180deg, #f2a0a0 0%, #e86a6a 100%) !important;
  color: #ffffff !important;
  border: 1px solid rgba(0,0,0,0.06) !important;
  box-shadow: 0 10px 24px rgba(232, 106, 106, 0.25) !important;
  height: 52px !important;
  padding: 0.8rem 0.9rem !important;
  border-radius: 14px !important;
  font-size: 1.05rem !important;
}
section[data-testid="stSidebar"] form div[data-testid="stFormSubmitButton"]:last-of-type button *{
  color: #ffffff !important;
  fill: #ffffff !important;
}

/* KPI cards */
.stMetric{
  background-color:#ffffff;
  padding:10px 12px;
  border-radius:10px;
  border: 1px solid #eef0f6;
  box-shadow:0 8px 20px rgba(17, 24, 39, 0.05);
  margin:0 !important;
}
div[data-testid="stMetricLabel"]{
  font-size:0.85rem !important;
  line-height:1.1 !important;
  margin-bottom:0.25rem !important;
  white-space:nowrap !important;
  color: #374151 !important;
  font-weight: 650 !important;
}
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] > div{
  font-size:clamp(1.2rem, 2.0vw, 2.2rem) !important;
  line-height:1.05 !important;
  font-variant-numeric: tabular-nums;
  white-space:nowrap !important;
  overflow:visible !important;
  text-overflow:clip !important;
  color: #111827 !important;
}

/* Loading overlay */
.loading-overlay{
  position: fixed;
  inset: 0;
  z-index: 100000;
  background: rgba(248, 250, 252, 0.92);
  backdrop-filter: blur(6px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
}
.loading-card{
  width: min(560px, 92vw);
  background: #ffffff;
  border: 1px solid #e7e9f2;
  border-radius: 18px;
  box-shadow: 0 20px 60px rgba(17, 24, 39, 0.14);
  padding: 18px 18px 16px 18px;
}
.loading-title{
  font-size: 1.1rem;
  font-weight: 850;
  color: #111827;
  margin: 0 0 6px 0;
}
.loading-sub{
  font-size: 0.95rem;
  color: #4b5563;
  margin: 0 0 14px 0;
  line-height: 1.4;
}
.loading-bar{
  width: 100%;
  height: 10px;
  border-radius: 999px;
  background: #eef2ff;
  overflow: hidden;
  border: 1px solid #e7e9f2;
}
.loading-bar > div{
  height: 100%;
  width: 40%;
  background: linear-gradient(90deg, #f2a0a0, #e86a6a, #f2a0a0);
  background-size: 200% 100%;
  animation: loading-move 1.1s ease-in-out infinite;
  border-radius: 999px;
}
@keyframes loading-move{
  0% { transform: translateX(-60%); background-position: 0% 50%; }
  50% { transform: translateX(60%); background-position: 100% 50%; }
  100% { transform: translateX(-60%); background-position: 0% 50%; }
}
.loading-foot{
  margin-top: 10px;
  font-size: 0.85rem;
  color: #6b7280;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# 3) Data Loading (Cloud-safe)
# -----------------------------
HF_DATASET_ID = "Ayanamikus/chicago-crime"

CACHE_DIR = Path("../.streamlit_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Sampling sizes (requested: 300k main sample)
LOCAL_MAIN_SAMPLE = CACHE_DIR / "main_sample_300k.parquet"
LOCAL_MAP_SAMPLE = CACHE_DIR / "map_sample_60k.parquet"


def _show_loading_overlay(title: str, subtitle: str = ""):
    ph = st.empty()
    ph.markdown(
        f"""
<div class="loading-overlay">
  <div class="loading-card">
    <div class="loading-title">{title}</div>
    <div class="loading-sub">{subtitle}</div>
    <div class="loading-bar"><div></div></div>
    <div class="loading-foot">Please keep this tab open.</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    return ph


def _postprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if "Year" not in df.columns and "year" in df.columns:
        df["Year"] = df["year"]

    if "hour" not in df.columns:
        df["hour"] = df["Date"].dt.hour if "Date" in df.columns else pd.NA
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["Date"].dt.day_name() if "Date" in df.columns else pd.NA
    if "month_year" not in df.columns:
        if "Date" in df.columns:
            df["month_year"] = df["Date"].dt.to_period("M").astype(str)
        else:
            df["month_year"] = pd.NA
    if "month" not in df.columns:
        if "Date" in df.columns:
            df["month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
        else:
            df["month"] = pd.NaT

    return df


def _balanced_sample_by_year(
    dfx: pd.DataFrame,
    year_col: str,
    n_total: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Stratified sampling by year: allocate roughly equal quota per year.
    Keeps per-year representation for animations.
    """
    if dfx.empty:
        return dfx

    years = sorted([y for y in dfx[year_col].dropna().unique()])
    if not years:
        return dfx.sample(n=min(n_total, len(dfx)), random_state=seed)

    quota = max(1, n_total // len(years))

    parts: List[pd.DataFrame] = []
    rng_seed = seed
    for y in years:
        sub = dfx[dfx[year_col] == y]
        if sub.empty:
            continue
        take = min(quota, len(sub))
        parts.append(sub.sample(n=take, random_state=rng_seed))
        rng_seed += 1

    out = (
        pd.concat(parts, ignore_index=True)
        if parts
        else dfx.sample(n=min(n_total, len(dfx)), random_state=seed)
    )

    if len(out) < min(n_total, len(dfx)):
        remaining = dfx.drop(index=out.index, errors="ignore")
        need = min(n_total - len(out), len(remaining))
        if need > 0:
            out = pd.concat(
                [out, remaining.sample(n=need, random_state=seed + 999)],
                ignore_index=True,
            )

    return out


@st.cache_data(show_spinner=False)
def load_main_sample(sample_rows: int = 300_000, seed: int = 42) -> pd.DataFrame:
    if LOCAL_MAIN_SAMPLE.exists():
        df = pd.read_parquet(LOCAL_MAIN_SAMPLE, engine="pyarrow")
        return _postprocess_df(df)

    from datasets import load_dataset

    ds = load_dataset(HF_DATASET_ID, split="train")

    # Keep only columns needed by the app (reduces memory)
    needed_cols = [
        "Date",
        "Year",
        "year",
        "Primary Type",
        "Arrest",
        "Domestic",
        "Latitude",
        "Longitude",
        "Location Description",
        "hour",
        "day_of_week",
        "month_year",
    ]
    keep = [c for c in needed_cols if c in ds.column_names]
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)

    n = len(ds)
    if sample_rows and 0 < sample_rows < n:
        ds = ds.shuffle(seed=seed).select(range(sample_rows))

    df = ds.to_pandas()
    df.to_parquet(LOCAL_MAIN_SAMPLE, index=False, engine="pyarrow", compression="zstd")
    return _postprocess_df(df)


@st.cache_data(show_spinner=False)
def load_map_sample(max_points: int = 60_000, seed: int = 42) -> pd.DataFrame:
    if LOCAL_MAP_SAMPLE.exists():
        m = pd.read_parquet(LOCAL_MAP_SAMPLE, engine="pyarrow")
        # no extra postprocess needed beyond Year/Date
        return _postprocess_df(m)

    from datasets import load_dataset

    ds = load_dataset(HF_DATASET_ID, split="train")

    # Only map-related columns
    needed_cols = [
        "Date",
        "Year",
        "year",
        "Primary Type",
        "Arrest",
        "Domestic",
        "Latitude",
        "Longitude",
        "Location Description",
    ]
    keep = [c for c in needed_cols if c in ds.column_names]
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)

    intermediate = max(200_000, max_points * 4)
    intermediate = min(intermediate, len(ds))
    ds = ds.shuffle(seed=seed).select(range(intermediate))

    m = ds.to_pandas()
    m = _postprocess_df(m)

    # Require coordinates
    m = m.dropna(subset=["Latitude", "Longitude"]).copy()
    if "Year" not in m.columns and "year" in m.columns:
        m["Year"] = m["year"]

    # Balanced by year for animation
    m2 = _balanced_sample_by_year(m, "Year", n_total=max_points, seed=seed)
    m2.to_parquet(LOCAL_MAP_SAMPLE, index=False, engine="pyarrow", compression="zstd")
    return m2


loading = _show_loading_overlay(
    "Loading dataset (Cloud-safe)…",
    f"Dataset: {HF_DATASET_ID} | Building cached samples for dashboard and maps.",
)
try:
    # Requested: main sample = 300k
    df_all = load_main_sample(sample_rows=300_000, seed=42)
    map_df_all = load_map_sample(max_points=60_000, seed=42)
finally:
    loading.empty()

if df_all is None or df_all.empty:
    st.error("No data loaded. Please check the Hugging Face dataset availability and your network.")
    st.stop()

# -----------------------------
# 4) Sidebar Filters (select all + apply)
# -----------------------------
required = {"Year", "Primary Type", "Arrest", "Domestic"}
missing_req = [c for c in required if c not in df_all.columns]
if missing_req:
    st.error(f"Missing required columns in dataset: {missing_req}")
    st.stop()

all_years = sorted(df_all["Year"].dropna().unique())
all_types = sorted(df_all["Primary Type"].dropna().unique())


def _sanitize_selection(values, options, fallback):
    s = [v for v in (values or []) if v in options]
    return s if s else fallback


if "applied_years" not in st.session_state:
    st.session_state["applied_years"] = all_years
if "applied_types" not in st.session_state:
    default_types = [t for t in ["THEFT", "BATTERY", "NARCOTICS"] if t in all_types] or all_types
    st.session_state["applied_types"] = default_types

if "draft_years" not in st.session_state:
    st.session_state["draft_years"] = st.session_state["applied_years"]
if "draft_types" not in st.session_state:
    st.session_state["draft_types"] = st.session_state["applied_types"]

if "_pending_select_all_years" not in st.session_state:
    st.session_state["_pending_select_all_years"] = False
if "_pending_select_all_types" not in st.session_state:
    st.session_state["_pending_select_all_types"] = False

st.session_state["applied_years"] = _sanitize_selection(st.session_state["applied_years"], all_years, all_years)
st.session_state["draft_years"] = _sanitize_selection(
    st.session_state["draft_years"], all_years, st.session_state["applied_years"]
)
st.session_state["applied_types"] = _sanitize_selection(st.session_state["applied_types"], all_types, all_types)
st.session_state["draft_types"] = _sanitize_selection(
    st.session_state["draft_types"], all_types, st.session_state["applied_types"]
)

if st.session_state.get("_pending_select_all_years", False):
    st.session_state["draft_years"] = all_years
    st.session_state["_pending_select_all_years"] = False

if st.session_state.get("_pending_select_all_types", False):
    st.session_state["draft_types"] = all_types
    st.session_state["_pending_select_all_types"] = False

with st.sidebar.form("filters_form", clear_on_submit=False):
    st.markdown("### Filters")

    y_sel, y_btn = cols([7, 3], vertical_alignment="bottom")
    with y_sel:
        st.multiselect("Years", all_years, key="draft_years")
    with y_btn:
        st.markdown('<div class="btn-right">', unsafe_allow_html=True)
        btn_all_years = form_submit("select all", key="select_all_years", use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

    t_sel, t_btn = cols([7, 3], vertical_alignment="bottom")
    with t_sel:
        st.multiselect("Categories", all_types, key="draft_types")
    with t_btn:
        st.markdown('<div class="btn-right">', unsafe_allow_html=True)
        btn_all_types = form_submit("select all", key="select_all_types", use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    apply_btn = st.form_submit_button("Apply", use_container_width=True)

st.sidebar.markdown("---")
with st.sidebar.expander("Performance / Cache", expanded=False):
    st.caption("Streamlit Cloud storage is ephemeral; cache persists only while the instance lives.")
    st.write("Main sample cache:", str(LOCAL_MAIN_SAMPLE))
    st.write("Map sample cache:", str(LOCAL_MAP_SAMPLE))
    if st.button("Clear cached samples (this session)"):
        if LOCAL_MAIN_SAMPLE.exists():
            LOCAL_MAIN_SAMPLE.unlink()
        if LOCAL_MAP_SAMPLE.exists():
            LOCAL_MAP_SAMPLE.unlink()
        st.cache_data.clear()
        st.success("Cleared. Rerun now.")
        st.rerun()

if btn_all_years:
    st.session_state["_pending_select_all_years"] = True
    st.rerun()

if btn_all_types:
    st.session_state["_pending_select_all_types"] = True
    st.rerun()

if apply_btn:
    st.session_state["applied_years"] = _sanitize_selection(st.session_state.get("draft_years"), all_years, all_years)
    st.session_state["applied_types"] = _sanitize_selection(st.session_state.get("draft_types"), all_types, all_types)

s_years = st.session_state["applied_years"]
s_types = st.session_state["applied_types"]

df = df_all[(df_all["Year"].isin(s_years)) & (df_all["Primary Type"].isin(s_types))].copy()

# Map DF uses separate sample but must respect same filters
map_df = map_df_all.copy()
if "Year" in map_df.columns:
    map_df = map_df[map_df["Year"].isin(s_years)]
if "Primary Type" in map_df.columns:
    map_df = map_df[map_df["Primary Type"].isin(s_types)]

# -----------------------------
# 5) Title + KPI
# -----------------------------
st.title("Chicago Crime Analysis Dashboard (2014–2024)")
st.markdown("---")

r1c1, r1c2, r1c3 = st.columns(3)
r1c1.metric("Total Sample (charts)", f"{len(df):,}")
r1c2.metric("Avg. Arrest Rate", f"{(df['Arrest'].mean() * 100):.1f}%")
r1c3.metric("Domestic Rate", f"{(df['Domestic'].mean() * 100):.1f}%")

r2c1, r2c2 = st.columns(2)
r2c1.metric("Categories", f"{df['Primary Type'].nunique()}")
r2c2.metric("Selected Years", f"{len(s_years)}")

# Added new tab for sampling method description
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Temporal Trends", "Spatial Analysis (Map)", "Categories & Arrests", "Technical Insights", "Sampling Method"]
)

# ----------------------------
# Tab 1: Temporal Trends
# ----------------------------
with tab1:
    if "hour" in df.columns and df["hour"].notna().any():
        h_data_sum = df.dropna(subset=["hour"]).groupby("hour").size()
        peak_hour = int(h_data_sum.idxmax()) if not h_data_sum.empty else 0
    else:
        peak_hour = 0

    st.markdown("### Interactive Temporal Patterns")

    t_col_nav, t_col_plot = st.columns([1, 4])
    with t_col_nav:
        metric = st.radio("Metric", ["Crime Volume", "Arrest Rate (%)"], key="t_metric")
        smooth = st.checkbox("Smooth Trend (6M)", key="t_smooth")
        show_range_slider = st.checkbox("Show range slider", value=True, key="t_slider")

        st.markdown("---")
        granularity = st.selectbox("Granularity", ["Monthly", "Weekly", "Daily"], index=0, key="t_granularity")
        top_n_types = st.slider("Top N Types (for stacked chart)", 3, 12, 6, key="t_topn")

    with t_col_plot:
        if "Date" not in df.columns or df["Date"].isna().all():
            st.warning("No valid Date column available; temporal plots may be limited.")
        else:
            dfx = df.dropna(subset=["Date"]).copy()

            if granularity == "Monthly":
                dfx["t"] = dfx["Date"].dt.to_period("M").dt.to_timestamp()
            elif granularity == "Weekly":
                dfx["t"] = dfx["Date"].dt.to_period("W-MON").dt.start_time
            else:
                dfx["t"] = dfx["Date"].dt.floor("D")

            if metric == "Crime Volume":
                ts = dfx.groupby("t").size().reset_index(name="Val")
                y_title = "Incidents"
            else:
                ts = dfx.groupby("t")["Arrest"].mean().reset_index(name="Val")
                ts["Val"] = ts["Val"] * 100
                y_title = "Arrest rate (%)"

            ts = ts.sort_values("t")
            if smooth and len(ts) > 0:
                ts["Val"] = ts["Val"].rolling(window=6, min_periods=1).mean()

            fig_ts = px.line(ts, x="t", y="Val", title=f"{granularity} Trend: {metric}")
            fig_ts.update_layout(xaxis_title="Time", yaxis_title=y_title, hovermode="x unified")
            if show_range_slider:
                fig_ts.update_layout(xaxis=dict(rangeslider=dict(visible=True)))

            fig_ts.add_vrect(
                x0="2020-03-01",
                x1="2021-06-30",
                fillcolor="lightgrey",
                opacity=0.25,
                line_width=0,
                annotation_text="COVID-19 Lockdown",
            )
            st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")
    st.subheader("Interactive Category Composition Over Time (Stacked)")

    if "Date" in df.columns and df["Date"].notna().any():
        dfx = df.dropna(subset=["Date"]).copy()
        dfx["t"] = dfx["Date"].dt.to_period("M").dt.to_timestamp()

        top_types = dfx["Primary Type"].value_counts().head(top_n_types).index.tolist()
        dfx["Type_grouped"] = dfx["Primary Type"].where(dfx["Primary Type"].isin(top_types), other="OTHER")

        area = dfx.groupby(["t", "Type_grouped"]).size().reset_index(name="Count").sort_values("t")
        fig_area = px.area(
            area,
            x="t",
            y="Count",
            color="Type_grouped",
            title=f"Monthly Volume by Type (Top {top_n_types} + OTHER)",
        )
        fig_area.update_layout(hovermode="x unified")
        st.plotly_chart(fig_area, use_container_width=True)
    else:
        st.info("Cannot build stacked composition plot: missing/invalid Date.")

    st.markdown("---")
    st.subheader("Hourly Activity Density (Heatmap)")

    if "day_of_week" not in df.columns or "hour" not in df.columns:
        st.warning("Cannot draw heatmap: missing 'day_of_week' or 'hour' column.")
    else:
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        h_map = (
            df.dropna(subset=["day_of_week", "hour"])
            .groupby(["day_of_week", "hour"])
            .size()
            .unstack(fill_value=0)
            .reindex(day_order)
        )
        fig_h = px.imshow(
            h_map,
            labels=dict(x="Hour (0-23)", y="Day of Week", color="Count"),
            color_continuous_scale="GnBu",
        )
        st.plotly_chart(fig_h, use_container_width=True)

# ----------------------------
# Tab 2: Spatial Analysis (Map)
# ----------------------------
with tab2:
    st.markdown("### Interactive Spatial Patterns")

    needed = {"Latitude", "Longitude", "Year", "Primary Type"}
    missing = [c for c in needed if c not in map_df.columns]
    if missing:
        st.warning(f"Cannot draw map: missing columns {missing}")
    else:
        s_col_nav, s_col_map = st.columns([1, 4])
        with s_col_nav:
            st.markdown("#### Map controls")
            map_mode = st.radio(
                "Mode",
                ["Animated points by Year", "Density (heatmap)", "Hexbin-like (grid)"],
                index=0,
                key="s_mode",
            )
            max_points = st.slider("Max points (performance)", 2000, 60000, 20000, step=2000, key="s_maxpts")
            color_by = st.selectbox("Color by", ["Primary Type", "Arrest", "Domestic"], index=0, key="s_colorby")
            show_only_arrests = st.checkbox("Show arrests only", value=False, key="s_only_arrests")

            st.caption(
                "Note: Map uses a cached balanced sample (not full 2.8M points) to stay responsive on Streamlit Cloud."
            )

        m_df = map_df.dropna(subset=["Latitude", "Longitude"]).copy()
        if show_only_arrests and "Arrest" in m_df.columns:
            m_df = m_df[m_df["Arrest"] == True]  # noqa: E712

        m_df = m_df.rename(columns={"Latitude": "lat", "Longitude": "lon"}).sort_values("Year")

        # Enforce max points with per-year cap to keep animation consistent
        if len(m_df) > max_points and m_df["Year"].nunique() > 0:
            per_year = max(1, max_points // max(1, m_df["Year"].nunique()))
            m_df = (
                m_df.sample(frac=1, random_state=42)
                .groupby("Year", group_keys=False)
                .head(per_year)
                .sort_values("Year")
            )

        with s_col_map:
            if m_df.empty:
                st.info("No map points under the current filters.")
            elif map_mode == "Animated points by Year":
                fig_m = px.scatter_map(
                    m_df,
                    lat="lat",
                    lon="lon",
                    color=color_by,
                    animation_frame="Year",
                    zoom=10,
                    height=650,
                    map_style="carto-positron",
                    hover_data={
                        "Primary Type": True,
                        "Arrest": True if "Arrest" in m_df.columns else False,
                        "Domestic": True if "Domestic" in m_df.columns else False,
                        "Location Description": True if "Location Description" in m_df.columns else False,
                    },
                )
                st.plotly_chart(fig_m, use_container_width=True)

            elif map_mode == "Density (heatmap)":
                fig_d = px.density_map(
                    m_df,
                    lat="lat",
                    lon="lon",
                    radius=18,
                    zoom=10,
                    height=650,
                    map_style="carto-positron",
                    hover_data={"Primary Type": True, "Arrest": True if "Arrest" in m_df.columns else False},
                )
                fig_d.update_layout(title="Spatial Density (Heatmap)")
                st.plotly_chart(fig_d, use_container_width=True)

            else:
                fig_g = px.scatter_map(
                    m_df,
                    lat="lat",
                    lon="lon",
                    color=color_by,
                    zoom=10,
                    height=650,
                    map_style="carto-positron",
                    opacity=0.35,
                    hover_data={"Primary Type": True, "Arrest": True if "Arrest" in m_df.columns else False},
                )
                st.plotly_chart(fig_g, use_container_width=True)

# ----------------------------
# Tab 3: Categories & Arrests
# ----------------------------
with tab3:
    if df.empty:
        st.warning("No data under current filters.")
    else:
        top_item = df["Primary Type"].value_counts().idxmax()
        top_pct = (df["Primary Type"].value_counts().max() / len(df)) * 100
        st.info(
            f"Under the current filter, **{top_item}** is the most frequent type, "
            f"representing **{top_pct:.1f}%** of the sample."
        )

        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            arr_data = df.groupby("Primary Type")["Arrest"].mean().reset_index()
            arr_data["Rate"] = arr_data["Arrest"] * 100
            fig_c = px.bar(
                arr_data.sort_values("Rate"),
                x="Rate",
                y="Primary Type",
                orientation="h",
                color="Rate",
                color_continuous_scale="RdYlGn",
                title="Arrest Rate by Category",
            )
            st.plotly_chart(fig_c, use_container_width=True)

        with col_c2:
            if "hour" in df.columns and df["hour"].notna().any():
                h_data_sum = df.dropna(subset=["hour"]).groupby("hour").size()
                peak_hour = int(h_data_sum.idxmax()) if not h_data_sum.empty else 0
            else:
                peak_hour = 0

            st.markdown("### Enforcement Insights")
            st.markdown(
                f"""
- **Peak Timing**: The highest risk window in the current selection appears at **{peak_hour:02d}:00**.
- **Typical pattern**:
  - **High efficiency** crimes tend to be proactive (e.g., narcotics).
  - **Low efficiency** crimes tend to be reactive (e.g., theft).
"""
            )

# ----------------------------
# Tab 4: Technical Insights
# ----------------------------
with tab4:
    st.markdown("### Data Diagnostics")
    st.markdown(
        """
- **Maps**: Use a cached balanced sample with coordinates to keep animations/heatmaps responsive.
"""
    )
    st.write("Main sample rows:", f"{len(df_all):,}")
    st.write("Map sample rows:", f"{len(map_df_all):,}")

    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        corr = df[num_cols].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Numeric Correlations")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("No numeric columns available for correlation plot")

# ----------------------------
# Tab 5: Sampling Method (NEW)
# ----------------------------
with tab5:
    st.markdown("### Sampling Method")

    st.markdown(
        """
This dashboard does **not** load the full Chicago crime dataset into memory. Instead, it uses **two cached samples**:

1. **Main chart sample (simple random sample)**
   - Source: Hugging Face dataset split `"train"`.
   - Method: **shuffle once with a fixed seed** and then **take the first N rows** (`select(range(sample_rows))`).
   - Purpose: keep charts (time series, category bars, heatmaps) responsive while still being broadly representative.

2. **Map sample (year-balanced stratified sample)**
   - Source: same dataset, but restricted to map-related columns.
   - Steps:
     1. Shuffle and take an **intermediate candidate pool** of size `intermediate = max(200_000, max_points * 4)`
        (capped by dataset size) to limit memory/time.
     2. Convert to pandas and **drop rows without coordinates** (`Latitude`, `Longitude`).
     3. Apply **stratified sampling by Year**: allocate a roughly equal quota per year so each year appears on the animated map.
        - Quota: `quota = n_total // number_of_years` (at least 1).
        - For each year: sample up to `quota` rows.
        - If the result is still smaller than `n_total`, fill the remaining slots by sampling from the leftover rows.
   - Purpose: ensure **consistent year-to-year representation** for animation frames and prevent recent years (often higher volume)
     from dominating the map.
"""
    )

    st.markdown("---")
    st.markdown("#### Current configuration")
    st.write("Main sample target rows:", "300,000")
    st.write("Map sample target rows:", "60,000")
    st.write("Random seed:", "42")

