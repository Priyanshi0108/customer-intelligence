from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# CONFIGURATION
# ============================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

PATHS = {
    "features": DATA_DIR / "customer_features.csv",
    "transactions": DATA_DIR / "cleaned_transactions.csv",
    "segments": REPORTS / "cluster_profiling" / "customer_segments.csv",
    "anomalies": REPORTS / "anomaly" / "all_methods_comparison.csv",
    "KMeans": REPORTS / "kmeans" / "kmeans_experiment_results.csv",
    "DBSCAN": REPORTS / "dbscan" / "dbscan_experiment_results.csv",
    "Hierarchical": REPORTS / "hierarchical" / "hierarchical_experiment_results.csv",
    "Gaussian Mixture Model": REPORTS / "gmm" / "gmm_experiment_results.csv",
        "retention_predictions": REPORTS / "retention" / "test_predictions.csv",
    "retention_actions": REPORTS / "retention" / "customer_relief_recommendations.csv",
    "retention_shap": REPORTS / "retention" / "customer_shap_explanation.csv",
    "retention_global_shap": REPORTS / "retention" / "shap_global_importance.csv",
}

st.set_page_config(
    page_title="Insight360 · Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

C = {
    "blue": "#2563EB", "cyan": "#0891B2", "purple": "#7C3AED",
    "green": "#059669", "orange": "#EA580C", "red": "#DC2626",
    "navy": "#0F172A", "text": "#1E293B", "muted": "#64748B",
}
PALETTE = ["#2563EB", "#7C3AED", "#0891B2", "#059669", "#EA580C", "#DB2777", "#64748B", "#CA8A04"]
ANOMALY_METHODS = ["iforest_anomaly", "lof_anomaly", "iqr_anomaly", "robust_zscore_anomaly"]
METHOD_LABELS = {
    "iforest_anomaly": "Isolation Forest",
    "lof_anomaly": "Local Outlier Factor",
    "iqr_anomaly": "IQR",
    "robust_zscore_anomaly": "Robust Z-Score",
}
RFM = ["recency_days", "frequency", "monetary", "total_quantity", "product_diversity"]


# ============================================================
# HTML RENDERING (root-cause fix for "raw code shown on page")
# Markdown treats any line indented by 4+ spaces, or any HTML block
# containing blank lines, as a code block. Stripping indentation and
# blank lines before rendering guarantees the HTML is parsed as HTML.
# ============================================================
def html(markup: str):
    clean = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)


html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
.stApp { background: #F4F7FB; color: #1E293B; }
[data-testid="stHeader"], [data-testid="stDecoration"], [data-testid="stToolbar"], #MainMenu, footer { display: none !important; }
.block-container { padding-top: 1.4rem !important; padding-bottom: 3rem !important; max-width: 1600px; }
[data-testid="stSidebar"] { background: #FFFFFF !important; border-right: 1px solid #E2E8F0; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span { color: #1E293B !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding: 9px 12px; border-radius: 10px; font-weight: 600; width: 100%; transition: all .15s ease; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: #EFF6FF; }
h1, h2, h3 { color: #0F172A !important; letter-spacing: -0.4px; }
.kpi-card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; padding: 18px 20px; min-height: 128px; box-shadow: 0 1px 3px rgba(15,23,42,.04); transition: all .2s ease; }
.kpi-card:hover { border-color: #93C5FD; box-shadow: 0 10px 26px rgba(37,99,235,.10); transform: translateY(-2px); }
.kpi-label { font-size: 12px; color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing: .6px; margin-bottom: 10px; }
.kpi-value { font-size: 28px; color: #0F172A; font-weight: 800; letter-spacing: -0.8px; line-height: 1.1; }
.kpi-caption { font-size: 12px; color: #94A3B8; margin-top: 8px; }
.kpi-delta { display: inline-block; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 999px; margin-left: 8px; vertical-align: middle; }
.kpi-delta.up { background: #DCFCE7; color: #15803D; }
.kpi-delta.down { background: #FEE2E2; color: #B91C1C; }
.kpi-delta.flat { background: #F1F5F9; color: #475569; }
.page-banner { background: linear-gradient(115deg, #0F172A 0%, #1E3A8A 60%, #2563EB 100%); border-radius: 18px; padding: 26px 30px; margin-bottom: 22px; box-shadow: 0 8px 24px rgba(30,58,138,.18); }
.page-banner h1 { font-size: 28px; margin: 0 0 6px 0; color: #FFFFFF !important; }
.page-banner p { color: #BFDBFE; font-size: 14px; margin: 0; }
.section-heading { font-size: 17px; font-weight: 700; color: #0F172A; margin: 14px 0 10px 0; }
.insight { background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #2563EB; border-radius: 12px; padding: 12px 16px; font-size: 13.5px; color: #334155; margin: 8px 0 14px 0; }
.chip { display: inline-block; background: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 999px; margin-right: 6px; }
[data-testid="stDataFrame"] { border: 1px solid #E2E8F0; border-radius: 12px; overflow: hidden; }
[data-testid="stPlotlyChart"] { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; padding: 6px; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #E2E8F0; }
.stTabs [data-baseweb="tab"] { height: 42px; padding: 0 16px; border-radius: 10px 10px 0 0; font-weight: 600; color: #64748B; }
.stTabs [aria-selected="true"] { color: #1D4ED8 !important; }
.stDownloadButton button { background: #2563EB; color: #fff; border-radius: 10px; border: none; font-weight: 600; }
.stDownloadButton button:hover { background: #1D4ED8; color: #fff; }
hr { border-color: #E2E8F0 !important; }
</style>
""")


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(show_spinner="Loading customer intelligence data...")
def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def norm_id(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "CustomerID" not in df.columns:
        return df
    df = df.copy()
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    return df


features = norm_id(load_csv(str(PATHS["features"])))
transactions = norm_id(load_csv(str(PATHS["transactions"])))
segments = norm_id(load_csv(str(PATHS["segments"])))
anomalies = norm_id(load_csv(str(PATHS["anomalies"])))

model_results = {k: load_csv(str(PATHS[k])) for k in ["KMeans", "DBSCAN", "Hierarchical", "Gaussian Mixture Model"]}

retention_predictions = norm_id(
    load_csv(str(PATHS["retention_predictions"]))
)

retention_actions = norm_id(
    load_csv(str(PATHS["retention_actions"]))
)

retention_shap = norm_id(
    load_csv(str(PATHS["retention_shap"]))
)

retention_global_shap = load_csv(
    str(PATHS["retention_global_shap"])
)


if features.empty or "CustomerID" not in features.columns:
    st.error("customer_features.csv is missing or has no CustomerID column (data/processed/).")
    st.stop()


def seg_label(c):
    if pd.isna(c):
        return "Unassigned"
    try:
        return f"Cluster {c:g}"
    except (TypeError, ValueError):
        return f"Cluster {c}"


customer_data = features.copy()
if not segments.empty and {"CustomerID", "cluster"}.issubset(segments.columns):
    lookup = segments[["CustomerID", "cluster"]].drop_duplicates("CustomerID")
    customer_data = features.merge(lookup, on="CustomerID", how="left")
    customer_data["segment"] = customer_data["cluster"].map(seg_label)
HAS_SEG = "segment" in customer_data.columns

if not anomalies.empty:
    for m in ANOMALY_METHODS:
        if m in anomalies.columns:
            anomalies[m] = pd.to_numeric(anomalies[m], errors="coerce").fillna(0).astype(int)
    avail = [m for m in ANOMALY_METHODS if m in anomalies.columns]
    anomalies["total_agreement"] = anomalies[avail].sum(axis=1) if avail else 0
AVAIL_METHODS = [m for m in ANOMALY_METHODS if m in anomalies.columns]


# ============================================================
# UI HELPERS
# ============================================================
def fmt_money(v):
    if pd.isna(v):
        return "£0"
    if abs(v) >= 1e6:
        return f"£{v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"£{v / 1e3:.1f}K"
    return f"£{v:,.0f}"


def kpi(label, value, caption="", accent=C["blue"], delta=None, direction="flat"):
    badge = f'<span class="kpi-delta {direction}">{delta}</span>' if delta else ""
    html(f"""
    <div class="kpi-card" style="border-top: 3px solid {accent};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}{badge}</div>
        <div class="kpi-caption">{caption}</div>
    </div>
    """)


def banner(title, subtitle):
    html(f'<div class="page-banner"><h1>{title}</h1><p>{subtitle}</p></div>')


def heading(title):
    html(f'<div class="section-heading">{title}</div>')


def insight(text):
    html(f'<div class="insight">💡 {text}</div>')


def style_fig(fig, height=400):
    fig.update_layout(
        height=height, template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, Arial, sans-serif", color=C["text"], size=12),
        margin=dict(l=15, r=15, t=55, b=20),
        title=dict(font=dict(size=15, color=C["navy"]), x=0.02, xanchor="left"),
        hoverlabel=dict(bgcolor=C["navy"], font_size=12, font_color="white"),
        legend=dict(bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig


def chart(fig, key, height=400):
    style_fig(fig, height)
    cfg = {"displaylogo": False, "scrollZoom": True, "responsive": True}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg, key=key)
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, config=cfg, key=key)


def table(df, height=350):
    try:
        st.dataframe(df, width="stretch", height=height, hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, height=height, hide_index=True)


def download(df, name, label="Download CSV"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv")


def have(df, *cols):
    return [c for c in cols if c in df.columns]


def radar(series: dict, cats: list, key: str):
    fig = go.Figure()
    for i, (name, vals) in enumerate(series.items()):
        fig.add_trace(go.Scatterpolar(
            r=list(vals) + [vals[0]], theta=cats + [cats[0]], fill="toself", name=name,
            line=dict(color=PALETTE[i % len(PALETTE)]), opacity=0.65,
        ))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], ticksuffix="%")), title="Percentile profile")
    chart(fig, key, 420)


# ============================================================
# SIDEBAR: NAVIGATION + GLOBAL FILTERS
# ============================================================
with st.sidebar:
    html("""
    <div style="padding: 12px 4px 14px 4px;">
        <div style="font-size:11px; color:#64748B; font-weight:700; letter-spacing:1px;">CUSTOMER INTELLIGENCE</div>
        <div style="font-size:24px; font-weight:800; color:#0F172A;">Insight<span style="color:#2563EB;">360</span></div>
        <div style="font-size:11px; color:#94A3B8;">ML-powered customer analytics</div>
    </div>
    """)
    page = st.radio(
        "Navigation",
        ["📈 Overview", "🧩 Customer Segmentation", "🚨 Anomaly Detection", "🔍 Customer Investigation", "🧪 Model Evaluation","💚 Retention Intelligence"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Global filters**")

    data = customer_data.copy()
    if HAS_SEG:
        all_segs = sorted(data["segment"].unique())
        chosen = st.multiselect("Segments", all_segs, default=all_segs)
        data = data[data["segment"].isin(chosen)]
    if "monetary" in features.columns and features["monetary"].nunique() > 1:
        lo, hi = float(features["monetary"].min()), float(features["monetary"].max())
        rng = st.slider("Revenue range (£)", lo, hi, (lo, hi))
        data = data[data["monetary"].between(*rng)]

    filtered_ids = set(data["CustomerID"])
    st.caption(f"{len(data):,} of {len(features):,} customers in view")
    st.divider()
    html("""
    <div style="font-size:11px; color:#64748B; line-height:1.9;">
        <b style="color:#334155;">PROJECT STATUS</b><br>
        <span style="color:#059669;">●</span> Dataset loaded<br>
        <span style="color:#059669;">●</span> Customer features ready<br>
        <span style="color:#059669;">●</span> ML experiments available
    </div>
    """)
    st.caption("Online Retail · UCI Dataset")

page = page.split(" ", 1)[1]

if data.empty:
    st.warning("No customers match the current filters. Adjust the sidebar filters.")
    st.stop()


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    banner("Customer Intelligence Overview",
           "Understand customer behavior, revenue concentration, segmentation and unusual activity.")

    total_rev = pd.to_numeric(data.get("monetary", pd.Series(dtype=float)), errors="coerce").sum()
    all_rev = pd.to_numeric(features.get("monetary", pd.Series(dtype=float)), errors="coerce").sum()
    avg_freq = pd.to_numeric(data.get("frequency", pd.Series(dtype=float)), errors="coerce").mean()
    flagged = 0
    if not anomalies.empty:
        flagged = anomalies.loc[(anomalies["total_agreement"] >= 1) & anomalies["CustomerID"].isin(filtered_ids), "CustomerID"].nunique()
    top10 = data.nlargest(max(1, len(data) // 10), "monetary")["monetary"].sum() if "monetary" in data else 0
    share = top10 / total_rev * 100 if total_rev else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Customers", f"{len(data):,}", "In current view", C["blue"])
    with c2: kpi("Revenue", fmt_money(total_rev), "Filtered customers", C["green"],
                 f"{total_rev / all_rev * 100:.0f}% of total" if all_rev else None)
    with c3: kpi("Avg. Frequency", f"{avg_freq:.1f}", "Invoices per customer", C["purple"])
    with c4: kpi("Top 10% Share", f"{share:.0f}%", "Revenue from best customers", C["cyan"])
    with c5: kpi("Flagged", f"{flagged:,}", "≥1 anomaly method", C["orange"])

    st.write("")
    insight(f"The top 10% of customers generate <b>{share:.0f}%</b> of revenue in this view. "
            "Use the sidebar filters to see how concentration changes by segment.")

    tab1, tab2, tab3 = st.tabs(["Revenue", "Behavior", "Correlations"])

    with tab1:
        left, right = st.columns([1.4, 1])
        with left:
            top_n = st.slider("Number of top customers", 5, 50, 15, key="topn")
            top = data.nlargest(top_n, "monetary").copy()
            top["CustomerID"] = top["CustomerID"].astype(int).astype(str)
            fig = px.bar(top.sort_values("monetary"), x="monetary", y="CustomerID", orientation="h",
                         color="monetary", color_continuous_scale="Blues",
                         labels={"monetary": "Revenue (£)", "CustomerID": "Customer"},
                         title=f"Top {top_n} Customers by Revenue",
                         hover_data=have(top, "frequency", "recency_days"))
            fig.update_layout(coloraxis_showscale=False)
            chart(fig, "ov_top", 460)
        with right:
            ranked = data.sort_values("monetary", ascending=False).reset_index(drop=True)
            ranked["cum_rev"] = ranked["monetary"].cumsum() / ranked["monetary"].sum() * 100
            ranked["cust_pct"] = (ranked.index + 1) / len(ranked) * 100
            fig = px.area(ranked, x="cust_pct", y="cum_rev", title="Revenue Concentration (Pareto)",
                          labels={"cust_pct": "% of customers", "cum_rev": "% of revenue"},
                          color_discrete_sequence=[C["blue"]])
            fig.add_hline(y=80, line_dash="dot", line_color=C["orange"], annotation_text="80% of revenue")
            chart(fig, "ov_pareto", 460)

    with tab2:
        left, right = st.columns(2)
        with left:
            fig = px.histogram(data, x="frequency", nbins=35, marginal="box",
                               color_discrete_sequence=[C["purple"]], title="Purchase Frequency",
                               labels={"frequency": "Number of invoices"})
            chart(fig, "ov_freq")
        with right:
            if "recency_days" in data:
                fig = px.histogram(data, x="recency_days", nbins=35, marginal="box",
                                   color_discrete_sequence=[C["cyan"]], title="Recency (days since last purchase)")
                chart(fig, "ov_rec")
        log_axes = st.toggle("Log scale (recommended for skewed revenue)", value=True)
        fig = px.scatter(data, x="frequency", y="monetary", color="segment" if HAS_SEG else None,
                         size="total_quantity" if "total_quantity" in data else None, size_max=28,
                         hover_data=have(data, "CustomerID", "recency_days"), color_discrete_sequence=PALETTE,
                         log_x=log_axes, log_y=log_axes, title="Customer Value vs Purchase Frequency",
                         labels={"frequency": "Purchase frequency", "monetary": "Monetary value (£)", "segment": "Segment"})
        fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0)))
        chart(fig, "ov_scatter", 480)

    with tab3:
        num_cols = [c for c in data.select_dtypes("number").columns if c not in ("CustomerID", "cluster")]
        if len(num_cols) > 1:
            corr = data[num_cols].corr().round(2)
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            title="Feature Correlation Matrix")
            chart(fig, "ov_corr", 520)

    with st.expander("View customer feature dataset"):
        table(data.head(200))
        download(data, "customers_filtered.csv", "Download filtered customers")


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================
elif page == "Customer Segmentation":
    banner("Customer Segmentation", "Discover customer groups based on purchasing behavior and find your most valuable segments.")

    if not HAS_SEG:
        st.warning("Cluster labels unavailable. Check reports/cluster_profiling/customer_segments.csv")
        st.stop()

    merged = data.dropna(subset=["cluster"])
    counts = merged["segment"].value_counts()
    total = len(merged)
    rev_by = merged.groupby("segment")["monetary"].sum() if "monetary" in merged else pd.Series(dtype=float)

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Segments", str(merged["segment"].nunique()), "Distinct clusters", C["blue"])
    with c2: kpi("Largest Segment", f"{counts.max():,}", f"{counts.idxmax()} · {counts.max() / total * 100:.1f}%", C["purple"])
    with c3: kpi("Top Revenue Segment", rev_by.idxmax() if len(rev_by) else "N/A",
                 f"{rev_by.max() / rev_by.sum() * 100:.0f}% of revenue" if len(rev_by) else "", C["green"])
    with c4: kpi("Customers Clustered", f"{total:,}", "With assigned cluster", C["cyan"])
    st.write("")

    tab1, tab2, tab3 = st.tabs(["Distribution", "Segment Profiles", "Explore Segment"])

    with tab1:
        left, right = st.columns(2)
        with left:
            dist = counts.rename_axis("segment").reset_index(name="customers")
            fig = px.pie(dist, names="segment", values="customers", hole=0.58, color_discrete_sequence=PALETTE,
                         title="Customers by Segment")
            fig.update_traces(textinfo="percent+label")
            chart(fig, "seg_pie")
        with right:
            if len(rev_by):
                rev_df = rev_by.rename_axis("segment").reset_index(name="revenue")
                fig = px.bar(rev_df, x="segment", y="revenue", color="segment", color_discrete_sequence=PALETTE,
                             text_auto=".2s", title="Revenue by Segment", labels={"revenue": "Revenue (£)"})
                fig.update_layout(showlegend=False)
                chart(fig, "seg_rev")

    with tab2:
        prof_cols = have(merged, "recency_days", "frequency", "monetary", "total_quantity", "product_diversity",
                         "average_order_value", "customer_tenure_days", "avg_purchase_interval")
        profile = merged.groupby("segment")[prof_cols].mean().round(2)
        left, right = st.columns([1.1, 1])
        with left:
            z = (profile - profile.min()) / (profile.max() - profile.min()).replace(0, 1)
            fig = px.imshow(z.T, text_auto=False, aspect="auto", color_continuous_scale="Blues",
                            title="Relative Strength of Each Feature (0 = lowest segment, 1 = highest)")
            fig.update_traces(customdata=profile.T.values, hovertemplate="%{x} · %{y}<br>Mean: %{customdata}<extra></extra>")
            chart(fig, "seg_heat", 440)
        with right:
            rad_cols = have(merged, *RFM)
            pct = merged[rad_cols].rank(pct=True) * 100
            pct["segment"] = merged["segment"]
            seg_pct = pct.groupby("segment").mean()
            radar({s: seg_pct.loc[s].tolist() for s in seg_pct.index}, rad_cols, "seg_radar")
        table(profile.reset_index())

    with tab3:
        sel = st.selectbox("Select a segment", sorted(merged["segment"].unique()))
        sub = merged[merged["segment"] == sel].sort_values("monetary" if "monetary" in merged else "CustomerID", ascending=False)
        st.write(f"Customers in **{sel}**: **{len(sub):,}**")
        table(sub.head(200))
        download(sub, f"{sel.lower().replace(' ', '_')}_customers.csv", "Download selected segment")


# ============================================================
# ANOMALY DETECTION
# ============================================================
elif page == "Anomaly Detection":
    banner("Anomaly Detection & Risk Analysis",
           "Compare multiple detection algorithms and prioritize unusual customer behavior for review.")

    if anomalies.empty or not AVAIL_METHODS:
        st.warning("Anomaly results not found. Check reports/anomaly/all_methods_comparison.csv")
        st.stop()

    adf = anomalies[anomalies["CustomerID"].isin(filtered_ids)].copy()
    n_flag = adf.loc[adf["total_agreement"] >= 1, "CustomerID"].nunique()
    n_high = adf.loc[adf["total_agreement"] >= 3, "CustomerID"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Methods Compared", str(len(AVAIL_METHODS)), "Independent detectors", C["blue"])
    with c2: kpi("Flagged", f"{n_flag:,}", "≥1 method", C["orange"], f"{n_flag / max(len(adf), 1) * 100:.1f}%", "flat")
    with c3: kpi("High Agreement", f"{n_high:,}", "≥3 methods", C["red"])
    with c4: kpi("Unanimous", f"{(adf['total_agreement'] == len(AVAIL_METHODS)).sum():,}", "All methods agree", C["purple"])
    st.write("")

    tab1, tab2, tab3 = st.tabs(["Method Comparison", "Agreement Map", "Investigate"])

    with tab1:
        left, right = st.columns(2)
        with left:
            mc = pd.DataFrame({"Method": [METHOD_LABELS[m] for m in AVAIL_METHODS],
                               "Flagged": [int(adf[m].sum()) for m in AVAIL_METHODS]})
            fig = px.bar(mc, x="Method", y="Flagged", color="Method", text="Flagged",
                         color_discrete_sequence=PALETTE, title="Customers Flagged by Each Algorithm")
            fig.update_layout(showlegend=False)
            chart(fig, "an_methods")
        with right:
            ag = adf["total_agreement"].value_counts().sort_index().reset_index()
            ag.columns = ["Methods Agreeing", "Customers"]
            fig = px.bar(ag, x="Methods Agreeing", y="Customers", text="Customers", color="Methods Agreeing",
                         color_continuous_scale="Oranges", title="How Many Models Flagged Each Customer?")
            fig.update_layout(coloraxis_showscale=False)
            chart(fig, "an_agree")

    with tab2:
        left, right = st.columns(2)
        with left:
            mat = adf[AVAIL_METHODS].T.dot(adf[AVAIL_METHODS])
            mat.index = mat.columns = [METHOD_LABELS[m] for m in AVAIL_METHODS]
            fig = px.imshow(mat, text_auto=True, color_continuous_scale="Blues",
                            title="Overlap: Customers Flagged by Both Methods")
            chart(fig, "an_overlap", 440)
        with right:
            extra = have(features, "frequency", "monetary")
            plot = adf.merge(features[["CustomerID"] + [c for c in extra if c not in adf.columns]], on="CustomerID", how="left")
            if {"frequency", "monetary"}.issubset(plot.columns):
                plot["Agreement"] = plot["total_agreement"].astype(str)
                fig = px.scatter(plot.sort_values("total_agreement"), x="frequency", y="monetary", color="Agreement",
                                 log_x=True, log_y=True, hover_data=["CustomerID"],
                                 color_discrete_sequence=["#CBD5E1", "#FDBA74", "#F97316", "#DC2626", "#7F1D1D"],
                                 title="Anomalies in Value / Frequency Space")
                chart(fig, "an_scatter", 440)

    with tab3:
        min_agree = st.slider("Minimum number of methods that must agree", 1, len(AVAIL_METHODS), min(3, len(AVAIL_METHODS)))
        sub = adf[adf["total_agreement"] >= min_agree].copy()
        miss = [c for c in have(features, "recency_days", "frequency", "monetary", "total_quantity") if c not in sub.columns]
        if miss:
            sub = sub.merge(features[["CustomerID"] + miss], on="CustomerID", how="left")
        sub = sub.sort_values("total_agreement", ascending=False)
        st.write(f"**{len(sub):,} customers** meet this threshold.")
        show = ["CustomerID", "total_agreement", *AVAIL_METHODS, *have(sub, "recency_days", "frequency", "monetary", "total_quantity")]
        table(sub[show].head(300), 420)
        download(sub, "anomaly_investigation_results.csv", "Download anomaly results")
        insight("Agreement between algorithms is a <b>review-priority signal</b>, not a probability of fraud. "
                "Inspect the underlying transactions before drawing conclusions.")


# ============================================================
# CUSTOMER INVESTIGATION
# ============================================================
elif page == "Customer Investigation":
    banner("Customer Investigation", "Inspect individual behavior, transaction history, segment membership and anomaly signals.")

    ids = sorted(data["CustomerID"].dropna().unique())
    selected = st.selectbox("Search and select Customer ID", ids, format_func=lambda x: str(int(x)))
    cust = features[features["CustomerID"] == selected].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Revenue", fmt_money(cust.get("monetary", 0)), "Total monetary value", C["green"])
    with c2: kpi("Frequency", f"{cust.get('frequency', 0):,.0f}", "Number of invoices", C["blue"])
    with c3: kpi("Recency", f"{cust.get('recency_days', 0):,.0f} days", "Since last purchase", C["orange"])
    with c4: kpi("Product Diversity", f"{cust.get('product_diversity', 0):,.0f}", "Unique products", C["purple"])
    st.write("")

    tab1, tab2, tab3 = st.tabs(["Profile & Risk", "Transactions", "Raw Data"])

    with tab1:
        left, right = st.columns([1.1, 1])
        with left:
            rad_cols = have(features, *RFM)
            pct = features[rad_cols].rank(pct=True) * 100
            me = pct[features["CustomerID"] == selected].iloc[0].tolist()
            series = {f"Customer {int(selected)}": me}
            if HAS_SEG:
                row = customer_data[customer_data["CustomerID"] == selected]
                if not row.empty and pd.notna(row.iloc[0]["cluster"]):
                    mask = (customer_data["cluster"] == row.iloc[0]["cluster"]).values
                    series[f"{row.iloc[0]['segment']} average"] = pct[mask].mean().tolist()
                    html(f'<span class="chip">{row.iloc[0]["segment"]}</span>')
            radar(series, rad_cols, "inv_radar")
            st.caption("Percentile vs all customers (for recency, a higher percentile means a longer time since last purchase).")
        with right:
            heading("Anomaly Detection Status")
            ca = anomalies[anomalies["CustomerID"] == selected] if not anomalies.empty else pd.DataFrame()
            if ca.empty or not AVAIL_METHODS:
                st.info("No anomaly result available for this customer.")
            else:
                row = ca.iloc[0]
                status = pd.DataFrame({
                    "Detection Method": [METHOD_LABELS[m] for m in AVAIL_METHODS],
                    "Status": ["🚩 Flagged" if int(row[m]) == 1 else "✅ Not flagged" for m in AVAIL_METHODS],
                })
                table(status, 190)
                agreement = int(row.get("total_agreement", 0))
                st.metric("Model Agreement", f"{agreement}/{len(AVAIL_METHODS)}")
                if agreement >= 3:
                    st.warning("Multiple algorithms flagged this customer. Prioritize manual investigation.")
                elif agreement >= 1:
                    st.info("At least one algorithm flagged this customer.")
                else:
                    st.success("No anomaly method flagged this customer.")

    tx = pd.DataFrame()
    if not transactions.empty and "CustomerID" in transactions.columns:
        tx = transactions[transactions["CustomerID"] == selected].copy()

    with tab2:
        if tx.empty:
            st.info("No transaction records found for this customer.")
        else:
            if "InvoiceDate" in tx.columns:
                tx["InvoiceDate"] = pd.to_datetime(tx["InvoiceDate"], errors="coerce")
            if {"Revenue", "InvoiceNo"}.issubset(tx.columns):
                agg = {"InvoiceDate": ("InvoiceDate", "min"), "Revenue": ("Revenue", "sum")}
                if "Quantity" in tx: agg["Quantity"] = ("Quantity", "sum")
                if "StockCode" in tx: agg["Products"] = ("StockCode", "nunique")
                inv = tx.groupby("InvoiceNo", as_index=False).agg(**agg).sort_values("InvoiceDate")
                inv["Cumulative"] = inv["Revenue"].cumsum()
                left, right = st.columns(2)
                with left:
                    fig = px.bar(inv, x="InvoiceDate", y="Revenue", color="Revenue", color_continuous_scale="Blues",
                                 title="Revenue per Invoice", hover_data=have(inv, "InvoiceNo", "Quantity", "Products"))
                    fig.update_layout(coloraxis_showscale=False)
                    chart(fig, "inv_bar")
                with right:
                    fig = px.area(inv, x="InvoiceDate", y="Cumulative", title="Cumulative Revenue",
                                  color_discrete_sequence=[C["green"]])
                    chart(fig, "inv_cum")
                if "Description" in tx.columns:
                    topp = tx.groupby("Description")["Revenue"].sum().nlargest(10).sort_values().reset_index()
                    fig = px.bar(topp, x="Revenue", y="Description", orientation="h", title="Top Products by Revenue",
                                 color_discrete_sequence=[C["purple"]])
                    chart(fig, "inv_prod")
                heading("Invoice Summary")
                table(inv.sort_values("InvoiceDate", ascending=False).head(100))

    with tab3:
        heading("Feature Values")
        table(pd.DataFrame({"Feature": cust.index, "Value": cust.astype(str).values}), 320)
        if not tx.empty:
            heading("Raw Transactions")
            table(tx.head(300))
            download(tx, f"customer_{int(selected)}_transactions.csv", "Download customer transactions")




# ============================================================
# RETENTION INTELLIGENCE
# ============================================================
elif page == "Retention Intelligence":
    banner(
        "Retention Intelligence",
        "Identify customers at risk of becoming inactive and explore personalized retention actions."
    )

    # --------------------------------------------------------
    # 1. Validate retention data
    # --------------------------------------------------------
    required_predictions = {"CustomerID", "snapshot_date", "churn_probability"}
    required_actions = {
        "CustomerID", "snapshot_date", "churn_probability",
        "priority", "type", "action", "reason", "safeguard"
    }

    missing_predictions = required_predictions - set(retention_predictions.columns)
    missing_actions = required_actions - set(retention_actions.columns)

    if retention_predictions.empty:
        st.warning(
            "Retention predictions are unavailable. "
            "Check reports/retention/test_predictions.csv."
        )
        st.stop()

    if missing_predictions:
        st.error(f"Missing prediction columns: {sorted(missing_predictions)}")
        st.stop()

    if retention_actions.empty:
        st.warning(
            "Personalized recommendations are unavailable. "
            "Check reports/retention/customer_relief_recommendations.csv."
        )
        st.stop()

    if missing_actions:
        st.error(f"Missing recommendation columns: {sorted(missing_actions)}")
        st.stop()

    # Work on copies so the original loaded data remains unchanged.
    pred = retention_predictions.copy()
    actions = retention_actions.copy()
    shap_data = retention_shap.copy()
    global_shap = retention_global_shap.copy()

    pred["snapshot_date"] = pd.to_datetime(
        pred["snapshot_date"], errors="coerce"
    )
    actions["snapshot_date"] = pd.to_datetime(
        actions["snapshot_date"], errors="coerce"
    )

    if not shap_data.empty and "snapshot_date" in shap_data.columns:
        shap_data["snapshot_date"] = pd.to_datetime(
            shap_data["snapshot_date"], errors="coerce"
        )

    pred["churn_probability"] = pd.to_numeric(
        pred["churn_probability"], errors="coerce"
    )

    actions["priority"] = pd.to_numeric(
        actions["priority"], errors="coerce"
    )

    pred = pred.dropna(
        subset=["CustomerID", "snapshot_date", "churn_probability"]
    )

    actions = actions.dropna(
        subset=["CustomerID", "snapshot_date"]
    )

    # Apply the dashboard's global customer filters.
    pred = pred[pred["CustomerID"].isin(filtered_ids)].copy()
    actions = actions[actions["CustomerID"].isin(filtered_ids)].copy()

    if pred.empty:
        st.info(
            "No retention predictions are available for customers "
            "matching the current sidebar filters."
        )
        st.stop()

    # --------------------------------------------------------
    # 2. Keep only the latest prediction for each customer
    # --------------------------------------------------------
    latest = (
        pred.sort_values("snapshot_date")
        .groupby("CustomerID", as_index=False)
        .tail(1)
        .copy()
    )

    latest["risk_level"] = np.select(
        [
            latest["churn_probability"] >= 0.80,
            latest["churn_probability"] >= 0.60,
        ],
        [
            "High",
            "Medium",
        ],
        default="Low",
    )

    latest["risk_percentage"] = latest["churn_probability"] * 100

    high = latest[latest["risk_level"] == "High"]
    medium = latest[latest["risk_level"] == "Medium"]
    low = latest[latest["risk_level"] == "Low"]

    avg_risk = latest["churn_probability"].mean() * 100
    high_share = len(high) / max(len(latest), 1) * 100

    # --------------------------------------------------------
    # 3. Risk overview
    # --------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "Risk Overview",
        "Customer Action Center",
        "Why Customers May Leave",
        "Retention Strategy",
    ])

    with tab1:
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            kpi(
                "Customers Scored",
                f"{len(latest):,}",
                "Latest available snapshot",
                C["blue"],
            )

        with c2:
            kpi(
                "High Risk",
                f"{len(high):,}",
                "Churn probability ≥80%",
                C["red"],
            )

        with c3:
            kpi(
                "Medium Risk",
                f"{len(medium):,}",
                "Churn probability 60–80%",
                C["orange"],
            )

        with c4:
            kpi(
                "Low Risk",
                f"{len(low):,}",
                "Churn probability <60%",
                C["green"],
            )

        with c5:
            kpi(
                "Average Risk",
                f"{avg_risk:.1f}%",
                "Average predicted probability",
                C["purple"],
            )

        st.write("")

        insight(
            f"<b>{high_share:.1f}%</b> of scored customers are currently "
            "in the high-risk category. Prioritize reviewing these customers "
            "before designing retention campaigns."
        )

        left, right = st.columns(2)

        with left:
            risk_counts = (
                latest["risk_level"]
                .value_counts()
                .reindex(["High", "Medium", "Low"], fill_value=0)
                .rename_axis("Risk Level")
                .reset_index(name="Customers")
            )

            fig = px.pie(
                risk_counts,
                names="Risk Level",
                values="Customers",
                hole=0.60,
                color="Risk Level",
                color_discrete_map={
                    "High": C["red"],
                    "Medium": C["orange"],
                    "Low": C["green"],
                },
                title="Customer Risk Distribution",
            )
            fig.update_traces(textinfo="percent+label")
            chart(fig, "ret_risk_pie", 420)

        with right:
            fig = px.histogram(
                latest,
                x="churn_probability",
                nbins=30,
                color="risk_level",
                color_discrete_map={
                    "High": C["red"],
                    "Medium": C["orange"],
                    "Low": C["green"],
                },
                title="Distribution of Churn Probabilities",
                labels={
                    "churn_probability": "Predicted churn probability",
                    "risk_level": "Risk category",
                },
            )
            fig.update_layout(barmode="overlay")
            fig.update_traces(opacity=0.75)
            chart(fig, "ret_risk_hist", 420)

        heading("Customers Requiring Attention")

        queue = latest.sort_values(
            "churn_probability", ascending=False
        ).copy()

        queue_display = queue[
            [
                "CustomerID",
                "snapshot_date",
                "churn_probability",
                "risk_level",
            ]
        ].copy()

        queue_display["churn_probability"] = (
            queue_display["churn_probability"] * 100
        ).round(1)

        queue_display = queue_display.rename(
            columns={
                "snapshot_date": "Snapshot Date",
                "CustomerID": "Customer ID",
                "churn_probability": "Churn Risk (%)",
                "risk_level": "Risk Level",
            }
        )

        table(queue_display.head(100), 400)
        download(
            queue_display,
            "retention_risk_queue.csv",
            "Download Risk Queue",
        )

    # --------------------------------------------------------
    # 4. Customer action center
    # --------------------------------------------------------
    with tab2:
        heading("Personalized Customer Retention")

        if actions.empty:
            st.info(
                "No personalized recommendations are available "
                "for the currently filtered customers."
            )
        else:
            # Use only recommendations from each customer's latest
            # available prediction snapshot.
            latest_keys = latest[
                ["CustomerID", "snapshot_date"]
            ].drop_duplicates()

            action_queue = actions.merge(
                latest_keys,
                on=["CustomerID", "snapshot_date"],
                how="inner",
            )

            action_queue = action_queue.merge(
                latest[["CustomerID", "risk_level"]],
                on="CustomerID",
                how="left",
            )

            action_queue = action_queue.sort_values(
                ["priority", "churn_probability"],
                ascending=[False, False],
            )

            if action_queue.empty:
                st.info(
                    "No recommendations match the latest prediction "
                    "snapshots. Check that the prediction and recommendation "
                    "files were generated by the same model run."
                )
            else:
                customer_options = sorted(
                    action_queue["CustomerID"].dropna().unique()
                )

                selected_customer = st.selectbox(
                    "Select a customer to investigate",
                    customer_options,
                    format_func=lambda x: str(int(x)),
                    key="retention_customer",
                )

                customer_risk = latest[
                    latest["CustomerID"] == selected_customer
                ]

                customer_actions = action_queue[
                    action_queue["CustomerID"] == selected_customer
                ].copy()

                if not customer_risk.empty:
                    customer_row = customer_risk.iloc[0]

                    a, b, c, d = st.columns(4)

                    with a:
                        kpi(
                            "Customer ID",
                            str(int(selected_customer)),
                            "Selected customer",
                            C["blue"],
                        )

                    with b:
                        kpi(
                            "Churn Probability",
                            f"{customer_row['churn_probability'] * 100:.1f}%",
                            "Predicted inactivity risk",
                            C["red"],
                        )

                    with c:
                        kpi(
                            "Risk Category",
                            customer_row["risk_level"],
                            "Latest prediction",
                            C["orange"],
                        )

                    with d:
                        kpi(
                            "Recommended Actions",
                            str(len(customer_actions)),
                            "Personalized interventions",
                            C["green"],
                        )

                    st.caption(
                        "Prediction snapshot: "
                        f"{customer_row['snapshot_date'].strftime('%d %b %Y')}"
                    )

                heading("Recommended Retention Actions")

                for _, action_row in customer_actions.iterrows():
                    action_type = str(action_row.get("type", "Action"))
                    action_name = str(action_row.get("action", ""))
                    reason = str(action_row.get("reason", ""))
                    safeguard = str(action_row.get("safeguard", ""))
                    priority = action_row.get("priority", np.nan)

                    with st.container(border=True):
                        st.markdown(f"**{action_name}**")

                        st.caption(
                            f"{action_type} · Priority: "
                            f"{priority if pd.notna(priority) else 'N/A'}"
                        )

                        st.write(f"**Why this action?** {reason}")

                        st.info(f"**Business safeguard:** {safeguard}")

                heading("Customer Recommendation Data")

                display_cols = [
                    "type",
                    "action",
                    "priority",
                    "reason",
                    "safeguard",
                ]

                table(customer_actions[display_cols], 300)

                download(
                    customer_actions,
                    f"customer_{int(selected_customer)}_retention_actions.csv",
                    "Download Customer Actions",
                )

                # Historical churn-risk trend.
                history = pred[
                    pred["CustomerID"] == selected_customer
                ].sort_values("snapshot_date")

                if len(history) > 1:
                    heading("Historical Churn Risk")

                    fig = px.line(
                        history,
                        x="snapshot_date",
                        y="churn_probability",
                        markers=True,
                        title="How This Customer's Risk Changed Over Time",
                        labels={
                            "snapshot_date": "Snapshot Date",
                            "churn_probability": "Churn Probability",
                        },
                        color_discrete_sequence=[C["red"]],
                    )

                    fig.update_yaxes(tickformat=".0%")
                    chart(fig, "ret_customer_history", 380)

    # --------------------------------------------------------
    # 5. Explain churn using SHAP
    # --------------------------------------------------------
    with tab3:
        heading("What Drives Customer Inactivity?")

        if global_shap.empty:
            st.info(
                "Global SHAP importance is unavailable. "
                "Check reports/retention/shap_global_importance.csv."
            )
        else:
            feature_col = next(
                (
                    col for col in [
                        "feature",
                        "Feature",
                    ]
                    if col in global_shap.columns
                ),
                None,
            )

            importance_col = next(
                (
                    col for col in [
                        "mean_abs_shap",
                        "mean_absolute_shap",
                        "mean_abs_shap_value",
                        "importance",
                        "mean_absolute_value",
                    ]
                    if col in global_shap.columns
                ),
                None,
            )

            if feature_col is None or importance_col is None:
                st.warning(
                    "Could not identify the feature and importance columns "
                    "in the global SHAP file. Inspect its column names."
                )
                table(global_shap.head(100))
            else:
                global_plot = global_shap.copy()
                global_plot[importance_col] = pd.to_numeric(
                    global_plot[importance_col],
                    errors="coerce",
                )
                global_plot = global_plot.dropna(
                    subset=[importance_col]
                ).sort_values(importance_col, ascending=True)

                fig = px.bar(
                    global_plot,
                    x=importance_col,
                    y=feature_col,
                    orientation="h",
                    color=importance_col,
                    color_continuous_scale="Blues",
                    title="Global Feature Importance — Mean Absolute SHAP",
                    labels={
                        importance_col: "Mean absolute SHAP value",
                        feature_col: "Feature",
                    },
                )
                fig.update_layout(coloraxis_showscale=False)
                chart(fig, "ret_global_shap", 460)

        st.divider()

        heading("Individual Customer Explanation")

        if shap_data.empty:
            st.info(
                "Customer-level SHAP explanations are unavailable. "
                "Check reports/retention/customer_shap_explanation.csv."
            )
        else:
            required_shap = {
                "CustomerID",
                "snapshot_date",
                "feature",
                "shap_value",
                "direction",
            }

            missing_shap = required_shap - set(shap_data.columns)

            if missing_shap:
                st.warning(
                    f"Missing SHAP columns: {sorted(missing_shap)}"
                )
            else:
                shap_customers = sorted(
                    shap_data["CustomerID"]
                    .dropna()
                    .unique()
                )

                if shap_customers:
                    shap_customer = st.selectbox(
                        "Choose a customer for SHAP explanation",
                        shap_customers,
                        format_func=lambda x: str(int(x)),
                        key="shap_customer",
                    )

                    customer_snapshots = shap_data[
                        shap_data["CustomerID"] == shap_customer
                    ]["snapshot_date"].dropna()

                    if not customer_snapshots.empty:
                        selected_snapshot = customer_snapshots.max()

                        local_shap = shap_data[
                            (shap_data["CustomerID"] == shap_customer)
                            & (
                                shap_data["snapshot_date"]
                                == selected_snapshot
                            )
                        ].copy()

                        local_shap["shap_value"] = pd.to_numeric(
                            local_shap["shap_value"],
                            errors="coerce",
                        )

                        local_shap = local_shap.dropna(
                            subset=["shap_value"]
                        ).sort_values("shap_value")

                        if not local_shap.empty:
                            local_shap["Impact"] = np.where(
                                local_shap["shap_value"] > 0,
                                "Increases predicted risk",
                                "Reduces predicted risk",
                            )

                            fig = px.bar(
                                local_shap,
                                x="shap_value",
                                y="feature",
                                orientation="h",
                                color="Impact",
                                color_discrete_map={
                                    "Increases predicted risk": C["red"],
                                    "Reduces predicted risk": C["green"],
                                },
                                title=(
                                    f"SHAP Explanation — Customer "
                                    f"{int(shap_customer)}"
                                ),
                                hover_data=have(
                                    local_shap,
                                    "customer_value",
                                    "direction",
                                ),
                            )

                            fig.add_vline(
                                x=0,
                                line_dash="dot",
                                line_color="#64748B",
                            )

                            chart(fig, "ret_local_shap", 460)

                            explanation_cols = have(
                                local_shap,
                                "feature",
                                "customer_value",
                                "shap_value",
                                "direction",
                            )

                            table(
                                local_shap[explanation_cols].sort_values(
                                    "shap_value",
                                    ascending=False,
                                ),
                                300,
                            )

                            st.caption(
                                "Positive SHAP values push the model's "
                                "predicted churn probability upward; "
                                "negative values push it downward. "
                                "SHAP explains model behavior, not causality."
                            )
                        else:
                            st.info(
                                "No valid SHAP values were found for this customer."
                            )

    # --------------------------------------------------------
    # 6. Retention strategy
    # --------------------------------------------------------
    with tab4:
        heading("Retention Intervention Framework")

        strategy = pd.DataFrame([
            {
                "Customer Signal": "High recency",
                "Possible Intervention": "Personalized comeback reminder",
                "Business Goal": "Re-engage inactive customers",
                "Safeguard": "Limit reminder frequency",
            },
            {
                "Customer Signal": "Low historical monetary value",
                "Possible Intervention": "Small conditional comeback voucher",
                "Business Goal": "Encourage another purchase",
                "Safeguard": "Minimum basket and incentive budget",
            },
            {
                "Customer Signal": "Low purchase frequency",
                "Possible Intervention": "Second-purchase reward",
                "Business Goal": "Build repeat-purchase behavior",
                "Safeguard": "Avoid unnecessary discounts",
            },
            {
                "Customer Signal": "Low product diversity",
                "Possible Intervention": "Curated product discovery",
                "Business Goal": "Increase relevant product discovery",
                "Safeguard": "Use purchase history for relevance",
            },
            {
                "Customer Signal": "Long purchase interval",
                "Possible Intervention": "Reorder reminder",
                "Business Goal": "Support repeat ordering",
                "Safeguard": "Respect expected buying cycles",
            },
            {
                "Customer Signal": "High-value inactive customer",
                "Possible Intervention": "VIP win-back outreach",
                "Business Goal": "Protect valuable relationships",
                "Safeguard": "Use personalized service before discounts",
            },
        ])

        table(strategy, 360)

        st.divider()

        heading("Recommended Business Workflow")

        workflow = [
            (
                "1. Identify",
                "Score customers using their historical purchasing behavior."
            ),
            (
                "2. Prioritize",
                "Review high-risk customers and their predicted churn probability."
            ),
            (
                "3. Explain",
                "Use SHAP to understand which features influenced each prediction."
            ),
            (
                "4. Personalize",
                "Select a relevant monetary or non-monetary retention action."
            ),
            (
                "5. Experiment",
                "Test interventions through controlled A/B experiments."
            ),
            (
                "6. Measure",
                "Track repeat purchases, retention lift, revenue and incentive cost."
            ),
        ]

        for title, description in workflow:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(description)

        st.warning(
            "Important: These interventions are model-informed recommendations, "
            "not proven causal treatments. SHAP explains predictions; it does not "
            "prove that a discount or reminder will prevent churn. Use A/B testing "
            "to measure actual retention impact."
        )

        st.caption(
            "The current prediction queue is based on the available held-out "
            "test snapshots. It is not yet a live scoring system for every customer."
        )


# MODEL EVALUATION

elif page == "Model Evaluation":
    banner("Model Evaluation & Experiment Tracking",
           "Compare clustering algorithms across metrics and understand configuration trade-offs.")

    METRICS = have(pd.DataFrame(columns=["silhouette_score", "davies_bouldin_score", "calinski_harabasz_score",
                                         "inertia", "bic", "aic"]), "silhouette_score", "davies_bouldin_score",
                   "calinski_harabasz_score", "inertia", "bic", "aic")
    all_metric_names = ["silhouette_score", "davies_bouldin_score", "calinski_harabasz_score", "inertia", "bic", "aic"]
    PARAMS = ["n_clusters", "n_components", "eps", "min_samples", "iterations"]

    tab1, tab2 = st.tabs(["Explore an Algorithm", "Compare Algorithms"])

    with tab1:
        name = st.selectbox("Select clustering algorithm", list(model_results.keys()))
        res = model_results[name].copy()
        if res.empty:
            st.warning(f"Results for {name} are unavailable. Check the reports folder.")
        else:
            ms = [m for m in all_metric_names if m in res.columns]
            res[ms] = res[ms].apply(pd.to_numeric, errors="coerce")
            sil = res["silhouette_score"].max() if "silhouette_score" in res else np.nan
            dbi = res["davies_bouldin_score"].min() if "davies_bouldin_score" in res else np.nan
            chi = res["calinski_harabasz_score"].max() if "calinski_harabasz_score" in res else np.nan
            c1, c2, c3, c4 = st.columns(4)
            with c1: kpi("Best Silhouette", f"{sil:.4f}" if pd.notna(sil) else "N/A", "Higher is better", C["green"])
            with c2: kpi("Best Davies-Bouldin", f"{dbi:.4f}" if pd.notna(dbi) else "N/A", "Lower is better", C["blue"])
            with c3: kpi("Best Calinski-Harabasz", f"{chi:,.1f}" if pd.notna(chi) else "N/A", "Higher is better", C["purple"])
            with c4: kpi("Experiments", f"{len(res):,}", "Saved configurations", C["cyan"])
            st.write("")

            xs = [c for c in PARAMS if c in res.columns]
            if xs and ms:
                default_x = next((c for c in ["eps", "n_components", "n_clusters"] if c in xs), xs[0])
                a, b, c = st.columns(3)
                with a: x = st.selectbox("Configuration (x-axis)", xs, index=xs.index(default_x))
                with b: y = st.selectbox("Metric (y-axis)", ms, index=ms.index("silhouette_score") if "silhouette_score" in ms else 0)
                color_opts = have(res, "linkage", "covariance_type", "min_samples", "dataset")
                with c: col = st.selectbox("Compare by", ["None"] + color_opts)
                col = None if col == "None" else col
                plot = res.dropna(subset=[x, y])
                if not plot.empty:
                    fig = px.line(plot.sort_values(x), x=x, y=y, color=col, markers=True, color_discrete_sequence=PALETTE,
                                  title=f"{name}: {y} across experiments",
                                  hover_data=have(plot, *PARAMS, "linkage", "covariance_type", "silhouette_score"))
                    chart(fig, "me_line", 420)

            if "silhouette_score" in res and res["silhouette_score"].notna().any():
                heading("Best Experiment Configuration")
                best = res.loc[res["silhouette_score"].idxmax()].astype(str).rename("Value").reset_index()
                best.columns = ["Parameter", "Value"]
                table(best, 300)

            heading("All Experiment Results")
            table(res)
            download(res, f"{name.lower().replace(' ', '_')}_experiments.csv", "Download experiment results")

    with tab2:
        rows = []
        for n, r in model_results.items():
            if not r.empty and "silhouette_score" in r.columns:
                s = pd.to_numeric(r["silhouette_score"], errors="coerce")
                if s.notna().any():
                    row = {"Algorithm": n, "Best Silhouette": s.max(), "Experiments": len(r)}
                    if "davies_bouldin_score" in r:
                        row["Best Davies-Bouldin"] = pd.to_numeric(r["davies_bouldin_score"], errors="coerce").min()
                    rows.append(row)
        if rows:
            comp = pd.DataFrame(rows).sort_values("Best Silhouette", ascending=False)
            fig = px.bar(comp, x="Algorithm", y="Best Silhouette", color="Algorithm", text_auto=".3f",
                         color_discrete_sequence=PALETTE, title="Best Silhouette Score by Algorithm")
            fig.update_layout(showlegend=False)
            chart(fig, "me_compare")
            table(comp.round(4), 220)
        else:
            st.info("No experiment results available to compare.")

    heading("How to Interpret the Metrics")
    table(pd.DataFrame([
        {"Metric": "Silhouette Score", "Meaning": "How well-separated clusters are", "Better": "Higher"},
        {"Metric": "Davies-Bouldin Index", "Meaning": "Similarity between clusters", "Better": "Lower"},
        {"Metric": "Calinski-Harabasz Index", "Meaning": "Between- vs within-cluster dispersion", "Better": "Higher"},
        {"Metric": "Inertia", "Meaning": "Within-cluster sum of squared distances", "Better": "Lower (consider cluster count)"},
        {"Metric": "BIC / AIC", "Meaning": "Penalized likelihood for model selection", "Better": "Lower"},
    ]), 230)
    insight("Don't pick a model from a single metric. Weigh cluster size, noise percentage, interpretability, "
            "stability and business usefulness. Compare silhouette scores directly only when the data and distance "
            "representation are comparable.")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
html("""
<div style="text-align:center; color:#94A3B8; font-size:11px; padding:10px;">
    INSIGHT360 &nbsp;·&nbsp; ML-powered customer analytics &nbsp;·&nbsp; Built with Python, Streamlit and Plotly
</div>
""")