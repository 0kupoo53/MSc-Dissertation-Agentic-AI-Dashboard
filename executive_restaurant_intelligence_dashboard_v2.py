# =========================================================
# EXECUTIVE RESTAURANT INTELLIGENCE DASHBOARD (ENHANCED)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ---------------------------------------------------------
# PAGE CONFIG & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Executive Restaurant Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main { padding: 1.2rem; }
    .block-container { padding-top: 0.8rem; }
    h1, h2, h3 { font-family: "Segoe UI", sans-serif; }
    .metric-label { font-size: 0.9rem !important; }
    .metric-value { font-size: 1.3rem !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------
@st.cache_data
def load_forecasts():
    with open("channel_forecasts.pkl", "rb") as f:
        channel_fc = pickle.load(f)
    with open("deliveroo_category_forecasts.pkl", "rb") as f:
        category_fc = pickle.load(f)

    channel_fc["ds"] = pd.to_datetime(channel_fc["ds"])
    category_fc["ds"] = pd.to_datetime(category_fc["ds"])

    # Remove Aggregator from forecasts if present
    if "Channel" in channel_fc.columns:
        channel_fc = channel_fc[channel_fc["Channel"] != "Aggregator"]

    return channel_fc, category_fc


@st.cache_data
def load_digital_channels():
    df = pd.read_csv("Public_DigitalChannels_Monthly_Revenue_FinancialYear.csv")
    fy_start = df["Financial_Year"].str.split("-").str[0].astype(int)
    month_num = df["Month_Num"].astype(int)
    year = np.where(month_num >= 4, fy_start, fy_start + 1)
    df["ds"] = pd.to_datetime(dict(year=year, month=month_num, day=1))
    return df


@st.cache_data
def load_aggregator():
    df = pd.read_csv("Public_Aggregator_Monthly_Revenue_FinancialYear.csv")
    fy_start = df["Financial_Year"].str.split("-").str[0].astype(int)
    month_num = df["Month_Num"].astype(int)
    year = np.where(month_num >= 4, fy_start, fy_start + 1)
    df["ds"] = pd.to_datetime(dict(year=year, month=month_num, day=1))
    return df


@st.cache_data
def load_deliveroo_categories():
    df = pd.read_csv("Public_Deliveroo_Category_Summary_2023_2025.csv")
    df["Month"] = df["Month"].astype(str).str.strip()
    df["ds"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"],
        format="%Y-%B"
    )
    return df


@st.cache_data
def load_epos_annual():
    return pd.read_csv("Public_EPOS_Annual_Category_Mix_2023_2025.csv")


@st.cache_data
def load_pnl():
    file = "Public_Derived_Consolidated_PnL_2023_2025.csv"
    for enc in ["cp1252", "latin1", "utf-8"]:
        try:
            df = pd.read_csv(file, sep=None, engine="python", encoding=enc)
            df = df.dropna(how="all")
            return df
        except Exception:
            continue
    st.error("Unable to parse P&L CSV. Please ensure it is saved as a clean CSV.")
    return pd.DataFrame()


@st.cache_data
def load_category_mapping():
    mapping = pd.read_csv("category_mapping.csv")
    mapping["Deliveroo_Category"] = mapping["Deliveroo_Category"].str.strip()
    mapping["EPOS_Category"] = mapping["EPOS_Category"].str.strip()
    return mapping

# ---------------------------------------------------------
# TRENDS + NARRATIVE + RECOMMENDATION ENGINE
# ---------------------------------------------------------
def compute_trends(selected_fc, value_col=None):
    """Compute short- and medium-term trends and volatility safely."""
    selected_fc = selected_fc.copy()

    # Auto-detect correct forecast column
    if value_col is None:
        if "yhat" in selected_fc.columns:
            value_col = "yhat"
        elif "Forecast" in selected_fc.columns:
            value_col = "Forecast"
        else:
            raise KeyError("No forecast column found. Expected 'yhat' or 'Forecast'.")

    selected_fc = selected_fc.sort_values("ds")
    series = selected_fc[value_col].astype(float).dropna()

    if len(series) < 6:
        return {
            "trend_3m": None,
            "trend_6m": None,
            "volatility": None,
            "value_col": value_col
        }

    trend_3m = series.tail(3).pct_change().mean() * 100
    trend_6m = series.tail(6).pct_change().mean() * 100
    volatility = series.pct_change().std()

    return {
        "trend_3m": trend_3m,
        "trend_6m": trend_6m,
        "volatility": volatility,
        "value_col": value_col
    }


def generate_strategic_recommendations(
    mode,
    selected,
    selected_fc,
    anomalies=None,
    value_col=None
):
    """
    Agentic recommendation engine:
    - Uses trend, volatility, anomalies
    - Channel/category-specific logic
    - SDG mapping, confidence, priority
    """

    recs = []
    metrics = compute_trends(selected_fc, value_col=value_col)

    t3 = metrics["trend_3m"]
    t6 = metrics["trend_6m"]
    vol = metrics["volatility"]

    def add_rec(area, evidence, rec, sdg, conf, priority):
        recs.append({
            "Insight Area": area,
            "Evidence": evidence,
            "Recommendation": rec,
            "SDG Link": sdg,
            "Confidence": conf,
            "Action Priority": priority
        })

    if t3 is not None:
        if t3 < -5:
            add_rec(
                f"{selected} Short-Term Decline",
                f"3-month trend: {t3:.2f}%",
                "Investigate recent operational or marketing changes; review pricing and promotions.",
                "SDG 8, SDG 12",
                "Medium",
                "High"
            )
        elif t3 > 10:
            add_rec(
                f"{selected} Short-Term Growth Opportunity",
                f"3-month trend: {t3:.2f}%",
                "Leverage growth with targeted campaigns and increased stock allocation.",
                "SDG 8",
                "High",
                "Medium"
            )

    if t6 is not None:
        if t6 < -3:
            add_rec(
                f"{selected} Medium-Term Underperformance",
                f"6-month trend: {t6:.2f}%",
                "Review category/channel strategy, menu mix, and customer experience.",
                "SDG 8",
                "Medium",
                "High"
            )
        elif t6 > 5:
            add_rec(
                f"{selected} Sustained Growth",
                f"6-month trend: {t6:.2f}%",
                "Consider long-term investment in this channel/category.",
                "SDG 8, SDG 9",
                "High",
                "Medium"
            )

    if vol is not None:
        if vol > 0.3:
            add_rec(
                f"{selected} High Volatility",
                f"Volatility index: {vol:.2f}",
                "Stabilise marketing cadence, review operational consistency, and monitor customer feedback.",
                "SDG 9",
                "High",
                "Medium"
            )
        elif vol < 0.1:
            add_rec(
                f"{selected} Stable Performance",
                f"Volatility index: {vol:.2f}",
                "Maintain current strategy and monitor monthly trends.",
                "SDG 9",
                "High",
                "Low"
            )

    if anomalies is not None and selected in anomalies:
        if anomalies[selected] == "Anomaly":
            add_rec(
                f"{selected} Performance Anomaly",
                "Recent performance flagged as anomalous.",
                "Investigate data quality, operational disruptions, or external events.",
                "SDG 9",
                "High",
                "High"
            )

    if mode == "Channel":
        selected_lower = selected.lower()

        if "deliveroo" in selected_lower:
            add_rec(
                "Deliveroo Operational Strategy",
                "Channel selected: Deliveroo.",
                "Review delivery radius, prep times, and menu availability windows.",
                "SDG 8",
                "Medium",
                "Medium"
            )
        elif "justeat" in selected_lower or "just eat" in selected_lower:
            add_rec(
                "JustEat Commercial Strategy",
                "Channel selected: JustEat.",
                "Optimise commission structure, menu pricing, and promotional participation.",
                "SDG 8, SDG 12",
                "Medium",
                "Medium"
            )
        elif "website" in selected_lower:
            add_rec(
                "Website Digital Strategy",
                "Channel selected: Website.",
                "Review SEO, paid ads, and conversion funnel to stabilise revenue.",
                "SDG 9",
                "High",
                "Medium"
            )

    if mode == "Category":
        name = selected.lower()

        if "appetiser" in name or "starter" in name:
            add_rec(
                "Appetiser Strategy",
                "Category selected: Appetisers/Starters.",
                "Bundle appetisers with mains to increase average order value.",
                "SDG 12",
                "High",
                "Medium"
            )
        if "main" in name:
            add_rec(
                "Mains Optimisation",
                "Category selected: Mains.",
                "Review portion sizes, pricing, and menu engineering for profitability.",
                "SDG 8, SDG 12",
                "High",
                "High"
            )
        if "dessert" in name:
            add_rec(
                "Dessert Upsell Strategy",
                "Category selected: Desserts.",
                "Introduce end-of-meal prompts and bundle offers.",
                "SDG 8",
                "Medium",
                "Medium"
            )
        if "drink" in name or "beverage" in name:
            add_rec(
                "Beverage Margin Strategy",
                "Category selected: Drinks.",
                "Highlight high-margin drinks and review pricing tiers.",
                "SDG 8",
                "High",
                "Medium"
            )

    if len(recs) == 0:
        add_rec(
            f"{selected} General Monitoring",
            "No major anomalies, volatility, or trend shifts detected.",
            "Maintain current strategy and review monthly.",
            "SDG 9",
            "High",
            "Low"
        )

    return pd.DataFrame(recs)

# ---------------------------------------------------------
# HELPER: UNIFIED CHANNEL ACTUALS
# ---------------------------------------------------------
@st.cache_data
def build_channel_actuals(digital_df, aggregator_df):
    digital = digital_df[["ds", "Channel", "Revenue"]].copy()
    agg = aggregator_df[["ds", "Channel", "Revenue"]].copy()
    all_channels = pd.concat([digital, agg], ignore_index=True)

    # Keep only real channels (no Aggregator)
    all_channels = all_channels[~all_channels["Channel"].eq("Aggregator")]

    all_channels = all_channels[
        (all_channels["ds"] >= "2023-01-01") &
        (all_channels["ds"] <= "2025-12-31")
    ]
    return all_channels

# ---------------------------------------------------------
# FORECAST ACCURACY HELPERS
# ---------------------------------------------------------
def compute_forecast_metrics(actual, pred, lower=None, upper=None):
    mask = (~actual.isna()) & (~pred.isna())
    a = actual[mask]
    p = pred[mask]
    if len(a) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    mae = np.mean(np.abs(a - p))
    rmse = np.sqrt(np.mean((a - p) ** 2))
    mape = np.mean(np.abs((a - p) / a)) * 100 if (a != 0).any() else np.nan
    bias = np.mean(p - a)

    coverage = np.nan
    if lower is not None and upper is not None:
        l = lower[mask]
        u = upper[mask]
        coverage = np.mean((a >= l) & (a <= u)) * 100

    return mae, rmse, mape, bias, coverage


def build_channel_accuracy_table(channel_fc, channel_actuals):
    rows = []
    for ch in sorted(channel_fc["Channel"].unique()):
        fc = channel_fc[channel_fc["Channel"] == ch][["ds", "yhat", "yhat_lower", "yhat_upper"]]
        act = channel_actuals[channel_actuals["Channel"] == ch].groupby("ds")["Revenue"].sum().reset_index()
        merged = pd.merge(fc, act, on="ds", how="inner")
        mae, rmse, mape, bias, coverage = compute_forecast_metrics(
            merged["Revenue"], merged["yhat"], merged["yhat_lower"], merged["yhat_upper"]
        )
        rows.append({
            "Channel": ch,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE (%)": mape,
            "Bias": bias,
            "PI Coverage (%)": coverage,
            "Obs Count": len(merged)
        })
    return pd.DataFrame(rows)


def build_category_accuracy_table(category_fc, deliv_raw):
    rows = []
    for cat in sorted(category_fc["Category"].unique()):
        fc = category_fc[category_fc["Category"] == cat][["ds", "yhat", "yhat_lower", "yhat_upper"]]
        act = deliv_raw[deliv_raw["Category"] == cat].groupby("ds")["Gross_Revenue"].sum().reset_index()
        merged = pd.merge(fc, act, on="ds", how="inner")
        mae, rmse, mape, bias, coverage = compute_forecast_metrics(
            merged["Gross_Revenue"], merged["yhat"], merged["yhat_lower"], merged["yhat_upper"]
        )
        rows.append({
            "Category": cat,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE (%)": mape,
            "Bias": bias,
            "PI Coverage (%)": coverage,
            "Obs Count": len(merged)
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------
# SIMPLE NARRATIVE ENGINE
# ---------------------------------------------------------
def generate_narrative_for_series(df, label_col, label_value):
    # Clean selection
    df[label_col] = df[label_col].astype(str).str.strip()
    label_value = str(label_value).strip()

    # Filter
    sub = df[df[label_col] == label_value].copy()
    if sub.empty:
        return f"No forecast data available for {label_value}."

    sub = sub.sort_values("ds")
    recent = sub.tail(6)

    # Determine correct forecast column
    if "Forecast" in recent.columns:
        value_col = "Forecast"
    elif "yhat" in recent.columns:
        value_col = "yhat"
    else:
        return "No usable forecast values found for narrative generation."

    # Extract values
    start = recent[value_col].iloc[0]
    end = recent[value_col].iloc[-1]
    change = end - start
    pct_change = (change / start * 100) if start != 0 else 0

    volatility = recent[value_col].std()
    vol_desc = "highly volatile" if volatility > (recent[value_col].mean() * 0.3) else "relatively stable"

    direction = "increasing" if change > 0 else "decreasing" if change < 0 else "stable"

    narrative = f"""
Over the most recent forecast period, **{label_value}** appears to be **{direction}** overall.

- Starting forecast: **{start:,.2f}**
- Ending forecast: **{end:,.2f}**
- Absolute change: **{change:,.2f}**
- Percentage change: **{pct_change:,.1f}%**
- Volatility: **{vol_desc}**

Interpretation:
- A {direction} pattern suggests that {label_value} is experiencing a structural shift in demand.
- The {vol_desc} behaviour indicates how predictable this area may be for planning.
"""

    return narrative

# ---------------------------------------------------------
# DATA QUALITY CHECK HELPERS
# ---------------------------------------------------------
def data_quality_summary(df, name, date_col=None):
    info = {}
    info["Dataset"] = name
    info["Rows"] = len(df)
    info["Columns"] = len(df.columns)
    info["Missing Cells"] = int(df.isna().sum().sum())
    info["Missing Rows"] = int(df.isna().any(axis=1).sum())
    info["Duplicate Rows"] = int(df.duplicated().sum())
    if date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col])
        info["Min Date"] = dates.min()
        info["Max Date"] = dates.max()
        info["Unique Dates"] = dates.nunique()
    else:
        info["Min Date"] = None
        info["Max Date"] = None
        info["Unique Dates"] = None
    return info

# ---------------------------------------------------------
# LOAD ALL DATA
# ---------------------------------------------------------
channel_fc, category_fc = load_forecasts()
digital_channels = load_digital_channels()
aggregator_monthly = load_aggregator()
deliv_raw = load_deliveroo_categories()
epos_annual = load_epos_annual()
pnl = load_pnl()
mapping = load_category_mapping()

channel_actuals = build_channel_actuals(digital_channels, aggregator_monthly)

channel_accuracy_df = build_channel_accuracy_table(channel_fc, channel_actuals)
category_accuracy_df = build_category_accuracy_table(category_fc, deliv_raw)

# ---------------------------------------------------------
# SIDEBAR NAVIGATION (ORDER CONFIRMED)
# ---------------------------------------------------------
st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Go to:",
    [
        "🏠 Executive Overview",
        "📈 Multi-Channel Performance",
        "🔍 Multi-Channel Comparison",
        "🍽️ Category & Menu Engineering",
        "🔮 Forecasting Hub",
        "📊 Forecast Accuracy Report",
        "🧪 Scenario Modelling",
        "🧹 Data Quality Check",
        "📉 Model Diagnostics",
        "💰 P&L & Financial Structure",
        "🧠 Narrative Insights"
    ]
)

# =========================================================
# 1) EXECUTIVE OVERVIEW 
# =========================================================
if page == "🏠 Executive Overview":
    st.title("Executive Restaurant Intelligence Dashboard")

    # --- FIX: Ensure datetime and sorting ---
    channel_actuals["ds"] = pd.to_datetime(channel_actuals["ds"])
    df_sorted = (
        channel_actuals
        .groupby(["ds", "Channel"])["Revenue"]
        .sum()
        .reset_index()
        .sort_values("ds")
    )

    total_revenue = df_sorted["Revenue"].sum()
    latest_date = df_sorted["ds"].max()
    latest_month_revenue = df_sorted[df_sorted["ds"] == latest_date]["Revenue"].sum()

    prev_year_date = latest_date.replace(year=latest_date.year - 1)
    prev_year_revenue = df_sorted[df_sorted["ds"] == prev_year_date]["Revenue"].sum()
    yoy = ((latest_month_revenue - prev_year_revenue) / prev_year_revenue * 100
           if prev_year_revenue > 0 else np.nan)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue (2023–2025)", f"£{total_revenue:,.0f}")
    col2.metric("Latest Month Revenue", f"£{latest_month_revenue:,.0f}")
    col3.metric("YoY Growth (Latest Month)", f"{yoy:,.1f}%" if not np.isnan(yoy) else "N/A")

    # Channel mix
    channel_mix = df_sorted.groupby("Channel")["Revenue"].sum().reset_index()
    fig_mix = px.pie(
        channel_mix,
        names="Channel",
        values="Revenue",
        title="Channel Revenue Mix (2023–2025)",
        hole=0.4
    )

    # EPOS mix
    cat_mix = epos_annual.groupby("Category")["Annual_Sales"].sum().reset_index()
    fig_cat = px.bar(
        cat_mix,
        x="Category",
        y="Annual_Sales",
        title="EPOS Category Mix (Annual Sales, 2023–2025)",
    )

    col_a, col_b = st.columns(2)
    col_a.plotly_chart(fig_mix, use_container_width=True)
    col_b.plotly_chart(fig_cat, use_container_width=True)

    # --- MULTI-CHANNEL TREND ---
    fig_trend = px.line(
        df_sorted,
        x="ds",
        y="Revenue",
        color="Channel",
        title="Multi-Channel Revenue Trend (Actuals)"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.caption("Data window: 2023-01-01 to 2025-12-31 (actuals). Forecasts are explored in the Forecasting Hub.")

# =========================================================
# 2) MULTI-CHANNEL PERFORMANCE 
# =========================================================
elif page == "📈 Multi-Channel Performance":
    st.title("Multi-Channel Performance & Correlation")

    # --- FIX: Ensure datetime, aggregation, and sorting ---
    channel_actuals["ds"] = pd.to_datetime(channel_actuals["ds"])
    df_sorted = (
        channel_actuals
        .groupby(["ds", "Channel"])["Revenue"]
        .sum()
        .reset_index()
        .sort_values("ds")
    )

    # --- FIXED TREND GRAPH ---
    fig_trend = px.line(
        df_sorted,
        x="ds",
        y="Revenue",
        color="Channel",
        title="Channel Revenue Over Time"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Correlation
    pivot = df_sorted.pivot_table(
        index="ds",
        columns="Channel",
        values="Revenue",
        aggfunc="sum"
    ).fillna(0)

    if pivot.shape[1] >= 2:
        corr = pivot.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Correlation Between Channels (Revenue)"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough channels to compute correlation.")

    # Summary table
    summary = df_sorted.groupby("Channel").agg(
        Total_Revenue=("Revenue", "sum"),
        Avg_Monthly_Revenue=("Revenue", "mean"),
        Max_Monthly_Revenue=("Revenue", "max"),
        Min_Monthly_Revenue=("Revenue", "min")
    ).reset_index()

    st.subheader("Channel Performance Summary")
    st.dataframe(summary)

    csv = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Channel Performance Summary (CSV)",
        data=csv,
        file_name="channel_performance_summary.csv",
        mime="text/csv"
    )

# =========================================================
# 3) MULTI-CHANNEL COMPARISON
# =========================================================
elif page == "🔍 Multi-Channel Comparison":
    st.title("Multi-Channel Comparison")

    st.markdown("Compare key metrics across Deliveroo, JustEat, and Website.")

    # Aggregate KPIs
    kpi = channel_actuals.groupby("Channel").agg(
        Total_Revenue=("Revenue", "sum"),
        Avg_Monthly_Revenue=("Revenue", "mean"),
        Std_Dev=("Revenue", "std")
    ).reset_index()

    col1, col2, col3 = st.columns(3)
    for i, row in kpi.iterrows():
        if i == 0:
            c = col1
        elif i == 1:
            c = col2
        else:
            c = col3
        c.metric(
            f"{row['Channel']} – Total Revenue",
            f"£{row['Total_Revenue']:,.0f}",
            help="Total revenue across the full period."
        )

    fig_bar = px.bar(
        kpi,
        x="Channel",
        y="Total_Revenue",
        title="Total Revenue by Channel",
        color="Channel"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Merge with accuracy metrics
    comp = pd.merge(
        kpi,
        channel_accuracy_df,
        on="Channel",
        how="left"
    )

    st.subheader("Channel Comparison Table (Performance + Accuracy)")
    st.dataframe(comp)

    csv = comp.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Channel Comparison (CSV)",
        data=csv,
        file_name="channel_comparison.csv",
        mime="text/csv"
    )

# =========================================================
# 4) CATEGORY & MENU ENGINEERING
# =========================================================
elif page == "🍽️ Category & Menu Engineering":
    st.title("Category Performance & Menu Engineering")

    cat_rev = deliv_raw.groupby("Category").agg(
        Total_Revenue=("Gross_Revenue", "sum"),
        Total_Quantity=("Quantity", "sum")
    ).reset_index()

    cat_rev = cat_rev.merge(
        mapping,
        left_on="Category",
        right_on="Deliveroo_Category",
        how="left"
    )

    epos_cat = epos_annual.groupby("Category").agg(
        EPOS_Annual_Sales=("Annual_Sales", "sum"),
        EPOS_Avg_Share=("Category_Share", "mean")
    ).reset_index()

    cat_merged = cat_rev.merge(
        epos_cat,
        left_on="EPOS_Category",
        right_on="Category",
        how="left",
        suffixes=("", "_EPOS")
    )

    cat_merged = cat_merged.drop(columns=["Category_EPOS"], errors="ignore")

    vol_median = cat_merged["Total_Revenue"].median()
    margin_median = cat_merged["EPOS_Annual_Sales"].median()

    def classify_menu(row):
        vol = row["Total_Revenue"]
        margin = row["EPOS_Annual_Sales"]
        if pd.isna(margin):
            return "Unclassified"
        if vol >= vol_median and margin >= margin_median:
            return "Star"
        elif vol >= vol_median and margin < margin_median:
            return "Plowhorse"
        elif vol < vol_median and margin >= margin_median:
            return "Puzzle"
        else:
            return "Dog"

    cat_merged["Menu_Class"] = cat_merged.apply(classify_menu, axis=1)

    fig_cat_rev = px.bar(
        cat_merged.sort_values("Total_Revenue", ascending=False),
        x="Category",
        y="Total_Revenue",
        color="Menu_Class",
        title="Deliveroo Category Revenue & Menu Engineering Class",
        color_discrete_map={
            "Star": "green",
            "Plowhorse": "orange",
            "Puzzle": "blue",
            "Dog": "red",
            "Unclassified": "grey"
        }
    )
    st.plotly_chart(fig_cat_rev, use_container_width=True)

    st.subheader("Category Detail (Deliveroo + EPOS)")
    st.dataframe(cat_merged[[
        "Category",
        "Total_Revenue",
        "Total_Quantity",
        "EPOS_Annual_Sales",
        "EPOS_Avg_Share",
        "Menu_Class"
    ]])

    csv = cat_merged.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Category Detail (CSV)",
        data=csv,
        file_name="category_detail_deliveroo_epos.csv",
        mime="text/csv"
    )

    st.markdown(
        """
**Menu Engineering Classes**  
- **Star**: High volume, high margin → protect and promote  
- **Plowhorse**: High volume, low margin → optimise pricing/cost  
- **Puzzle**: Low volume, high margin → improve visibility/positioning  
- **Dog**: Low volume, low margin → candidates to drop or rework  
"""
    )

# =========================================================
# 5) FORECASTING HUB
# =========================================================
elif page == "🔮 Forecasting Hub":
    st.title("Forecasting Hub")

    tab1, tab2 = st.tabs(["Channel Forecasts", "Category Forecasts"])

    # ---------- CHANNEL FORECASTS ----------
    with tab1:
        st.subheader("Channel Forecasts")

        channels = sorted(channel_fc["Channel"].unique().tolist())
        selected = st.selectbox("Select a channel:", channels, key="fc_channel")

        df_fc = channel_fc[channel_fc["Channel"] == selected].copy()
        df_fc = df_fc.sort_values("ds")
        df_fc = df_fc[df_fc["ds"] >= "2023-01-01"]

        actual = channel_actuals[channel_actuals["Channel"] == selected].copy()
        actual = actual.groupby("ds")["Revenue"].sum().reset_index()

        merged = pd.merge(
            df_fc[["ds", "yhat", "yhat_lower", "yhat_upper"]],
            actual,
            on="ds",
            how="inner"
        )
        mae, rmse, mape, bias, coverage = compute_forecast_metrics(
            merged["Revenue"],
            merged["yhat"],
            merged["yhat_lower"],
            merged["yhat_upper"]
        )

        fig = go.Figure()
        if not actual.empty:
            fig.add_trace(go.Scatter(
                x=actual["ds"], y=actual["Revenue"],
                mode="lines+markers",
                name="Actual Revenue",
                line=dict(color="grey")
            ))
        fig.add_trace(go.Scatter(
            x=df_fc["ds"], y=df_fc["yhat"],
            mode="lines",
            name="Forecast",
            line=dict(color="royalblue")
        ))
        fig.update_layout(title=f"Actual vs Forecast – {selected}", xaxis_title="Date", yaxis_title="Revenue")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Latest Forecast", f"{df_fc['yhat'].iloc[-1]:,.2f}")
        col2.metric("MAE", f"{mae:,.2f}" if not np.isnan(mae) else "N/A")
        col3.metric("RMSE", f"{rmse:,.2f}" if not np.isnan(rmse) else "N/A")
        col4.metric("MAPE", f"{mape:,.1f}%" if not np.isnan(mape) else "N/A")
        col5.metric("PI Coverage", f"{coverage:,.1f}%" if not np.isnan(coverage) else "N/A")

        st.write("Forecast Data")
        st.dataframe(df_fc)

        csv = df_fc.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Channel Forecast Data (CSV)",
            data=csv,
            file_name=f"forecast_channel_{selected}.csv",
            mime="text/csv"
        )

    # ---------- CATEGORY FORECASTS ----------
    with tab2:
        st.subheader("Deliveroo Category Forecasts")

        categories = sorted(category_fc["Category"].unique().tolist())
        selected_cat = st.selectbox("Select a category:", categories, key="fc_category")

        df_fc_cat = category_fc[category_fc["Category"] == selected_cat].copy()
        df_fc_cat = df_fc_cat.sort_values("ds")
        df_fc_cat = df_fc_cat[df_fc_cat["ds"] >= "2023-01-01"]

        actual_cat = deliv_raw[deliv_raw["Category"] == selected_cat].copy()
        actual_cat = actual_cat.groupby("ds")["Gross_Revenue"].sum().reset_index()

        merged_cat = pd.merge(
            df_fc_cat[["ds", "yhat", "yhat_lower", "yhat_upper"]],
            actual_cat,
            on="ds",
            how="inner"
        )
        mae_c, rmse_c, mape_c, bias_c, coverage_c = compute_forecast_metrics(
            merged_cat["Gross_Revenue"],
            merged_cat["yhat"],
            merged_cat["yhat_lower"],
            merged_cat["yhat_upper"]
        )

        fig2 = go.Figure()
        if not actual_cat.empty:
            fig2.add_trace(go.Scatter(
                x=actual_cat["ds"], y=actual_cat["Gross_Revenue"],
                mode="lines+markers",
                name="Actual Gross Revenue",
                line=dict(color="grey")
            ))
        fig2.add_trace(go.Scatter(
            x=df_fc_cat["ds"], y=df_fc_cat["yhat"],
            mode="lines",
            name="Forecast",
            line=dict(color="darkgreen")
        ))
        fig2.update_layout(title=f"Actual vs Forecast – {selected_cat}", xaxis_title="Date", yaxis_title="Revenue")
        st.plotly_chart(fig2, use_container_width=True)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Latest Forecast", f"{df_fc_cat['yhat'].iloc[-1]:,.2f}")
        col2.metric("MAE", f"{mae_c:,.2f}" if not np.isnan(mae_c) else "N/A")
        col3.metric("RMSE", f"{rmse_c:,.2f}" if not np.isnan(rmse_c) else "N/A")
        col4.metric("MAPE", f"{mape_c:,.1f}%" if not np.isnan(mape_c) else "N/A")
        col5.metric("PI Coverage", f"{coverage_c:,.1f}%" if not np.isnan(coverage_c) else "N/A")

        st.write("Forecast Data")
        st.dataframe(df_fc_cat)

        csv = df_fc_cat.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Category Forecast Data (CSV)",
            data=csv,
            file_name=f"forecast_category_{selected_cat}.csv",
            mime="text/csv"
        )

# =========================================================
# 6) FORECAST ACCURACY REPORT
# =========================================================
elif page == "📊 Forecast Accuracy Report":
    st.title("Forecast Accuracy Report")

    tab1, tab2 = st.tabs(["Channel Accuracy", "Category Accuracy"])

    with tab1:
        st.subheader("Channel Forecast Accuracy")
        st.dataframe(channel_accuracy_df)

        csv = channel_accuracy_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Channel Accuracy Report (CSV)",
            data=csv,
            file_name="forecast_accuracy_channels.csv",
            mime="text/csv"
        )

    with tab2:
        st.subheader("Category Forecast Accuracy")
        st.dataframe(category_accuracy_df)

        csv2 = category_accuracy_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Category Accuracy Report (CSV)",
            data=csv2,
            file_name="forecast_accuracy_categories.csv",
            mime="text/csv"
        )

# =========================================================
# 7) SCENARIO MODELLING
# =========================================================
elif page == "🧪 Scenario Modelling":
    st.title("Scenario Modelling")

    st.markdown(
        """
Use this page to apply **what-if** adjustments to the forecasts and see the impact on revenue.

Scenarios can represent:
- Price changes  
- Marketing uplift  
- Channel cannibalisation  
- Operational constraints  
"""
    )

    df_fc = channel_fc.copy()
    df_fc = df_fc[df_fc["ds"] >= "2023-01-01"].copy()

    global_growth = st.slider("Global Growth / Shock (%)", -50, 100, 0)
    channels = sorted(df_fc["Channel"].unique().tolist())
    selected_channel = st.selectbox("Channel for additional adjustment:", channels)
    channel_growth = st.slider(f"{selected_channel} Specific Adjustment (%)", -50, 100, 0)

    df_fc["adjusted_yhat"] = df_fc["yhat"] * (1 + global_growth / 100)
    mask = df_fc["Channel"] == selected_channel
    df_fc.loc[mask, "adjusted_yhat"] = df_fc.loc[mask, "adjusted_yhat"] * (1 + channel_growth / 100)

    agg = df_fc.groupby("ds")["adjusted_yhat"].sum().reset_index()

    fig = px.line(
        agg,
        x="ds",
        y="adjusted_yhat",
        title="Adjusted Total Forecast (All Channels)"
    )
    fig.update_yaxes(title="Adjusted Revenue")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Scenario Summary by Channel")
    summary = df_fc.groupby("Channel")["adjusted_yhat"].sum().reset_index()
    summary.rename(columns={"adjusted_yhat": "Scenario_Total_Revenue"}, inplace=True)
    st.dataframe(summary)

    csv = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Scenario Summary (CSV)",
        data=csv,
        file_name="scenario_summary_by_channel.csv",
        mime="text/csv"
    )

# =========================================================
# 8) DATA QUALITY CHECK (NEW)
# =========================================================
elif page == "🧹 Data Quality Check":
    st.title("Data Quality Check")

    datasets = []

    datasets.append(data_quality_summary(digital_channels, "Digital Channels", "ds"))
    datasets.append(data_quality_summary(aggregator_monthly, "Aggregator Monthly", "ds"))
    datasets.append(data_quality_summary(deliv_raw, "Deliveroo Categories", "ds"))
    datasets.append(data_quality_summary(epos_annual, "EPOS Annual", None))
    datasets.append(data_quality_summary(pnl, "P&L", None))
    datasets.append(data_quality_summary(channel_fc, "Channel Forecasts", "ds"))
    datasets.append(data_quality_summary(category_fc, "Category Forecasts", "ds"))

    dq_df = pd.DataFrame(datasets)
    st.subheader("Dataset Quality Summary")
    st.dataframe(dq_df)

    csv = dq_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Data Quality Summary (CSV)",
        data=csv,
        file_name="data_quality_summary.csv",
        mime="text/csv"
    )

    st.markdown(
        """
This page highlights:
- Missing values  
- Duplicate rows  
- Date coverage  
- Basic structural issues  

"""
    )

# =========================================================
# 9) MODEL DIAGNOSTICS (NEW)
# =========================================================
elif page == "📉 Model Diagnostics":
    st.title("Model Diagnostics")

    tab1, tab2 = st.tabs(["Channel Residuals", "Category Residuals"])

    # CHANNEL RESIDUALS
    with tab1:
        st.subheader("Channel Forecast Residuals")

        channels = sorted(channel_fc["Channel"].unique().tolist())
        selected = st.selectbox("Select channel:", channels, key="diag_channel")

        fc = channel_fc[channel_fc["Channel"] == selected][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        act = channel_actuals[channel_actuals["Channel"] == selected].groupby("ds")["Revenue"].sum().reset_index()

        merged = pd.merge(fc, act, on="ds", how="inner")
        if merged.empty:
            st.info("No overlapping actuals and forecasts for this channel.")
        else:
            merged["residual"] = merged["Revenue"] - merged["yhat"]

            fig_res = px.line(
                merged,
                x="ds",
                y="residual",
                title=f"Residuals Over Time – {selected}"
            )
            st.plotly_chart(fig_res, use_container_width=True)

            fig_hist = px.histogram(
                merged,
                x="residual",
                nbins=20,
                title=f"Residual Distribution – {selected}"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            coverage = np.mean(
                (merged["Revenue"] >= merged["yhat_lower"]) &
                (merged["Revenue"] <= merged["yhat_upper"])
            ) * 100
            st.metric("Prediction Interval Coverage", f"{coverage:,.1f}%")

    # CATEGORY RESIDUALS
    with tab2:
        st.subheader("Category Forecast Residuals")

        categories = sorted(category_fc["Category"].unique().tolist())
        selected_cat = st.selectbox("Select category:", categories, key="diag_category")

        fc = category_fc[category_fc["Category"] == selected_cat][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        act = deliv_raw[deliv_raw["Category"] == selected_cat].groupby("ds")["Gross_Revenue"].sum().reset_index()

        merged = pd.merge(fc, act, on="ds", how="inner")
        if merged.empty:
            st.info("No overlapping actuals and forecasts for this category.")
        else:
            merged["residual"] = merged["Gross_Revenue"] - merged["yhat"]

            fig_res = px.line(
                merged,
                x="ds",
                y="residual",
                title=f"Residuals Over Time – {selected_cat}"
            )
            st.plotly_chart(fig_res, use_container_width=True)

            fig_hist = px.histogram(
                merged,
                x="residual",
                nbins=20,
                title=f"Residual Distribution – {selected_cat}"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            coverage = np.mean(
                (merged["Gross_Revenue"] >= merged["yhat_lower"]) &
                (merged["Gross_Revenue"] <= merged["yhat_upper"])
            ) * 100
            st.metric("Prediction Interval Coverage", f"{coverage:,.1f}%")

# =========================================================
# 10) P&L & FINANCIAL STRUCTURE
# =========================================================
elif page == "💰 P&L & Financial Structure":
    st.title("P&L Structure and Profitability")

    st.write("Derived Consolidated P&L (2023–2025)")
    st.dataframe(pnl)

    pnl_cat = pnl.groupby("Account Category")["Value (£)"].sum().reset_index()

    fig = px.bar(
        pnl_cat,
        x="Account Category",
        y="Value (£)",
        title="P&L by Account Category",
        color="Account Category",
        color_discrete_sequence=px.colors.sequential.Blues
    )
    st.plotly_chart(fig, use_container_width=True)

    net_row = pnl[pnl["Account Category"].str.contains("NET PROFIT", case=False, na=False)]
    if not net_row.empty:
        net_value = net_row["Value (£)"].iloc[0]
        st.metric("Net Profit (EBIT)", f"£{net_value:,.0f}")
    else:
        st.info("Net profit row not found in P&L file.")

    csv = pnl.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download P&L Data (CSV)",
        data=csv,
        file_name="pnl_derived_consolidated.csv",
        mime="text/csv"
    )

    st.markdown(
        """
This view helps answer:
- How much profit the restaurant generates  
- How revenue is converted into gross profit and net profit  
- Which cost blocks dominate the P&L  
"""
    )

# =========================================================
# 11) NARRATIVE INSIGHTS
# =========================================================
elif page == "🧠 Narrative Insights":
    st.title("Narrative Insights")

    st.markdown(
        """
Select a **channel** or **category** to generate an executive-style narrative based on forecast trends.
"""
    )

    mode = st.radio("Narrative for:", ["Channel", "Category"])

    if mode == "Channel":
        options = sorted(channel_fc["Channel"].unique().tolist())
        selected = st.selectbox("Select channel:", options)
        narrative = generate_narrative_for_series(channel_fc, "Channel", selected)

        # Example: extract simple metrics for recommendations
        try:
            selected_fc = channel_fc[channel_fc["Channel"] == selected]
            recent_growth = selected_fc["Forecast"].pct_change().iloc[-1] * 100
        except:
            recent_growth = None

    else:
        options = sorted(category_fc["Category"].unique().tolist())
        selected = st.selectbox("Select category:", options)
        narrative = generate_narrative_for_series(category_fc, "Category", selected)

        try:
            selected_fc = category_fc[category_fc["Category"] == selected]
            recent_growth = selected_fc["Forecast"].pct_change().iloc[-1] * 100
        except:
            recent_growth = None

    # -----------------------------
    # 1. Display Narrative
    # -----------------------------
    st.subheader(f"Narrative for {selected}")
    st.markdown(narrative)

      # -----------------------------
    # 2. Strategic Recommendation Log (Full Agentic Engine)
    # -----------------------------
    st.subheader("📌 Strategic Recommendation Log")

    rec_df = generate_strategic_recommendations(
    mode=mode,
    selected=selected,
    selected_fc=selected_fc,
    anomalies=None
)

    st.dataframe(rec_df, use_container_width=True)


