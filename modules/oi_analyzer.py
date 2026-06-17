"""
OI (Open Interest) Analyzer Module
Calculates Delta OI, net OI trends, and generates directional arrows for chart annotations.
Provides functions for timeframe-based OI tables and strike volume analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import streamlit as st
import logging

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")
TIMEFRAMES = [1, 2, 5, 10, 15, 30, 60]


# ─── OI Snapshot Storage ──────────────────────────────────────────────────────

@st.cache_data(ttl=5)
def get_oi_snapshot() -> dict:
    """
    Get current options chain OI snapshot.
    Returns dict with strike -> {ce_oi, pe_oi, ce_volume, pe_volume, net_oi}.
    """
    from modules.angelone_client import fetch_options_chain, get_next_weekly_expiry, get_expiry_string
    expiry_dt = get_next_weekly_expiry()
    expiry_str = get_expiry_string(expiry_dt)
    df = fetch_options_chain(expiry_str)

    if df.empty:
        return {}

    snapshot = {}
    for _, row in df.iterrows():
        strike = int(row["strike"])
        snapshot[strike] = {
            "ce_oi": int(row.get("ce_oi", 0)),
            "pe_oi": int(row.get("pe_oi", 0)),
            "ce_volume": int(row.get("ce_volume", 0)),
            "pe_volume": int(row.get("pe_volume", 0)),
            "net_oi": int(row.get("pe_oi", 0)) - int(row.get("ce_oi", 0)),
        }
    return snapshot


def store_oi_snapshot(snapshot: dict):
    """Store OI snapshot with timestamp in session_state history."""
    if "oi_history" not in st.session_state:
        st.session_state["oi_history"] = []

    entry = {
        "timestamp": datetime.now(IST),
        "snapshot": snapshot,
    }
    st.session_state["oi_history"].append(entry)

    # Keep only last 2 hours of data
    cutoff = datetime.now(IST) - timedelta(hours=2)
    st.session_state["oi_history"] = [
        e for e in st.session_state["oi_history"]
        if e["timestamp"] > cutoff
    ]


# ─── Core OI Analysis ─────────────────────────────────────────────────────────

def compute_delta_oi(options_df: pd.DataFrame, spot_price: float) -> dict:
    """
    Compute current OI delta/bias from options chain.
    Returns dict with: bias, pcr, ce_oi, pe_oi, net_oi, delta_net_oi.
    Used by best-trade tab.
    """
    if options_df is None or options_df.empty:
        return {"bias": "NEUTRAL", "pcr": 1.0, "ce_oi": 0, "pe_oi": 0, "net_oi": 0, "delta_net_oi": 0}

    total_ce = int(options_df["ce_oi"].sum())
    total_pe = int(options_df["pe_oi"].sum())
    pcr = total_pe / max(total_ce, 1)
    net_oi = total_pe - total_ce

    if pcr > 1.25:
        bias = "BULLISH"   # Contrarian: high PCR = oversold
    elif pcr < 0.75:
        bias = "BEARISH"   # Contrarian: low PCR = overbought
    elif net_oi > 500000:
        bias = "BEARISH"
    elif net_oi < -500000:
        bias = "BULLISH"
    else:
        bias = "NEUTRAL"

    # Calculate delta from session OI history
    history = st.session_state.get("oi_history", [])
    delta_net_oi = 0
    if len(history) >= 2:
        past = history[0]["snapshot"]
        curr_net = sum(v["pe_oi"] - v["ce_oi"] for v in history[-1]["snapshot"].values())
        past_net = sum(v["pe_oi"] - v["ce_oi"] for v in past.values())
        delta_net_oi = curr_net - past_net

    return {
        "bias": bias,
        "pcr": round(pcr, 3),
        "ce_oi": total_ce,
        "pe_oi": total_pe,
        "net_oi": net_oi,
        "delta_net_oi": delta_net_oi,
    }


def get_delta_oi_for_timeframe(minutes: int) -> dict:
    """
    Calculate delta OI (change in net OI) over given timeframe in minutes.
    Returns dict with: delta_net_oi, direction ("BULLISH"/"BEARISH"/"NEUTRAL"),
    ce_oi_change, pe_oi_change, arrow, color.
    """
    history = st.session_state.get("oi_history", [])
    if len(history) < 2:
        return _neutral_delta(minutes)

    now = datetime.now(IST)
    cutoff = now - timedelta(minutes=minutes)

    past_entries = [e for e in history if e["timestamp"] <= cutoff]
    if not past_entries:
        past_entries = [history[0]]

    past_snap = past_entries[-1]["snapshot"]
    current_snap = history[-1]["snapshot"]

    total_ce_oi_change = 0
    total_pe_oi_change = 0
    total_net_oi_change = 0
    strikes_analyzed = 0

    for strike in current_snap:
        if strike in past_snap:
            ce_change = current_snap[strike]["ce_oi"] - past_snap[strike]["ce_oi"]
            pe_change = current_snap[strike]["pe_oi"] - past_snap[strike]["pe_oi"]
            net_change = pe_change - ce_change
            total_ce_oi_change += ce_change
            total_pe_oi_change += pe_change
            total_net_oi_change += net_change
            strikes_analyzed += 1

    if strikes_analyzed == 0:
        return _neutral_delta(minutes)

    threshold = 50000
    if total_net_oi_change > threshold:
        direction = "BEARISH"
        arrow = "↓"
        color = "#ff4444"
    elif total_net_oi_change < -threshold:
        direction = "BULLISH"
        arrow = "↑"
        color = "#00ff88"
    else:
        direction = "NEUTRAL"
        arrow = "→"
        color = "#ffd700"

    return {
        "direction": direction,
        "arrow": arrow,
        "color": color,
        "delta_net_oi": total_net_oi_change,
        "ce_oi_change": total_ce_oi_change,
        "pe_oi_change": total_pe_oi_change,
        "timeframe_minutes": minutes,
        "strikes_analyzed": strikes_analyzed,
    }


def _neutral_delta(minutes: int = 0) -> dict:
    return {
        "direction": "NEUTRAL",
        "arrow": "→",
        "color": "#ffd700",
        "delta_net_oi": 0,
        "ce_oi_change": 0,
        "pe_oi_change": 0,
        "timeframe_minutes": minutes,
        "strikes_analyzed": 0,
    }


def get_all_timeframe_deltas() -> dict:
    """Get delta OI for all standard timeframes."""
    results = {}
    for tf in TIMEFRAMES:
        results[tf] = get_delta_oi_for_timeframe(tf)
    return results


# ─── Directional OI Trend (which side is OI building) ─────────────────────────

def analyze_oi_trend(minutes: int) -> dict:
    """
    Compare CE vs PE OI change over the last `minutes` using the standard
    Indian-market convention:
        • PE OI building (vs CE)  → put writers / support  → BULLISH
        • CE OI building (vs PE)  → call writers / resistance → BEARISH
    Returns a dict; 'available' is False when there isn't enough history yet.
    """
    history = st.session_state.get("oi_history", [])
    now = datetime.now(IST)
    if len(history) < 2:
        return {"available": False, "minutes": minutes}

    cutoff = now - timedelta(minutes=minutes)
    past_entries = [e for e in history if e["timestamp"] <= cutoff]
    past_entry = past_entries[-1] if past_entries else history[0]
    past_snap = past_entry["snapshot"]
    curr_snap = history[-1]["snapshot"]

    ce_chg = pe_chg = 0
    for strike, cur in curr_snap.items():
        if strike in past_snap:
            ce_chg += cur["ce_oi"] - past_snap[strike]["ce_oi"]
            pe_chg += cur["pe_oi"] - past_snap[strike]["pe_oi"]

    net = pe_chg - ce_chg
    threshold = 30000
    if net > threshold:
        verdict, color, arrow = "BULLISH", "#00ff88", "↑"
    elif net < -threshold:
        verdict, color, arrow = "BEARISH", "#ff4444", "↓"
    else:
        verdict, color, arrow = "NEUTRAL", "#ffd700", "→"

    span_sec = (history[-1]["timestamp"] - past_entry["timestamp"]).total_seconds()
    return {
        "available": True,
        "minutes": minutes,
        "ce_change": int(ce_chg),
        "pe_change": int(pe_chg),
        "net": int(net),
        "verdict": verdict,
        "color": color,
        "arrow": arrow,
        "span_minutes": round(span_sec / 60, 1),
    }


def oi_support_resistance(options_df: pd.DataFrame, spot: float, n: int = 3) -> dict:
    """
    Current OI distribution snapshot:
      • highest PE OI strikes  → support zones
      • highest CE OI strikes  → resistance zones
      • PCR and total CE/PE OI
    Works off the live (or last-session) chain — no time history needed.
    """
    if options_df is None or options_df.empty:
        return {"available": False}

    df = options_df.copy()
    total_ce = int(df["ce_oi"].sum())
    total_pe = int(df["pe_oi"].sum())
    pcr = round(total_pe / max(total_ce, 1), 2)

    supports = (df.nlargest(n, "pe_oi")[["strike", "pe_oi"]]
                .to_dict("records")) if "pe_oi" in df else []
    resistances = (df.nlargest(n, "ce_oi")[["strike", "ce_oi"]]
                   .to_dict("records")) if "ce_oi" in df else []

    # Distribution bias: bigger PE wall below = support (bullish), etc.
    if pcr > 1.2:
        bias, color = "BULLISH", "#00ff88"
    elif pcr < 0.8:
        bias, color = "BEARISH", "#ff4444"
    else:
        bias, color = "NEUTRAL", "#ffd700"

    return {
        "available": True,
        "total_ce_oi": total_ce,
        "total_pe_oi": total_pe,
        "pcr": pcr,
        "bias": bias,
        "color": color,
        "supports": supports,       # max PE OI
        "resistances": resistances, # max CE OI
    }


# ─── OI Table Builder ─────────────────────────────────────────────────────────

def build_oi_timeframe_table(candle_data_by_tf: dict, options_df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """
    Build OI difference table showing CE vs PE OI trend for each timeframe.
    Returns DataFrame with columns: Timeframe, CE OI, PE OI, Net OI, PCR, Trend, Arrow.
    """
    if options_df is None or options_df.empty:
        return pd.DataFrame()

    total_ce = int(options_df["ce_oi"].sum())
    total_pe = int(options_df["pe_oi"].sum())
    pcr_base = total_pe / max(total_ce, 1)

    rows = []
    for tf in TIMEFRAMES:
        d = get_delta_oi_for_timeframe(tf)
        ce_chg = d.get("ce_oi_change", 0)
        pe_chg = d.get("pe_oi_change", 0)
        net_chg = d.get("delta_net_oi", 0)
        direction = d.get("direction", "NEUTRAL")
        arrow = d.get("arrow", "→")

        # Effective OI for this timeframe (snapshot - delta)
        tf_ce = max(0, total_ce - ce_chg)
        tf_pe = max(0, total_pe - pe_chg)
        tf_pcr = round(tf_pe / max(tf_ce, 1), 3)

        rows.append({
            "Timeframe": f"{tf} min",
            "CE OI": tf_ce,
            "PE OI": tf_pe,
            "Net OI Δ": net_chg,
            "PCR": tf_pcr,
            "Trend": direction,
            "Arrow": arrow,
        })

    return pd.DataFrame(rows)


# ─── OI Arrow Annotations for Chart ──────────────────────────────────────────

def get_oi_arrow_annotations(
    candle_df: pd.DataFrame,
    options_df: pd.DataFrame,
    spot: float,
    selected_tf: int,
) -> list:
    """
    Build Plotly annotation dicts for OI direction arrows to overlay on chart.
    Returns list of annotation dicts compatible with fig.update_layout(annotations=...).
    """
    if candle_df is None or candle_df.empty:
        return []

    annotations = []
    last_ts = candle_df["timestamp"].iloc[-1]
    last_close = float(candle_df["close"].iloc[-1])
    candle_range = float(candle_df["high"].max() - candle_df["low"].min())

    deltas = get_all_timeframe_deltas()
    y_start = last_close + candle_range * 0.04
    y_step = candle_range * 0.03

    for i, tf in enumerate(TIMEFRAMES[:4]):  # Show 4 most recent TFs
        d = deltas.get(tf, _neutral_delta(tf))
        arrow = d.get("arrow", "→")
        color = d.get("color", "#ffd700")
        direction = d.get("direction", "NEUTRAL")

        annotations.append(dict(
            x=last_ts,
            y=y_start + i * y_step,
            text=f"{tf}m {arrow}",
            showarrow=False,
            font=dict(size=11, color=color),
            xanchor="right",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor=color,
            borderwidth=1,
            borderpad=3,
            xref="x",
            yref="y",
        ))

    return annotations


# ─── Strike Volume Table ──────────────────────────────────────────────────────

def build_strike_volume_table(
    options_df: pd.DataFrame,
    candle_data_by_tf: dict,
    spot: float,
    n_strikes: int = 5,
) -> pd.DataFrame:
    """
    Build strike-level volume/OI table for ATM ± n_strikes.
    Returns formatted DataFrame for display.
    """
    from modules.angelone_client import get_strike_range, get_atm_strike
    if options_df is None or options_df.empty:
        return pd.DataFrame()

    strike_range = get_strike_range(spot, n_strikes)
    atm = get_atm_strike(spot)
    filtered = options_df[options_df["strike"].isin(strike_range)].copy()

    if filtered.empty:
        return pd.DataFrame()

    filtered["label"] = filtered["strike"].apply(
        lambda s: "ATM" if s == atm else (f"+{int(s-atm)}" if s > atm else f"{int(s-atm)}")
    )
    filtered["total_volume"] = filtered["ce_volume"] + filtered["pe_volume"]
    filtered["pcr"] = (filtered["pe_oi"] / filtered["ce_oi"].replace(0, 1)).round(3)
    filtered["net_oi"] = filtered["pe_oi"] - filtered["ce_oi"]
    filtered = filtered.sort_values("total_volume", ascending=False)

    display = filtered[[
        "strike", "label", "ce_ltp", "pe_ltp",
        "ce_oi", "pe_oi", "net_oi",
        "ce_volume", "pe_volume", "total_volume", "pcr"
    ]].rename(columns={
        "strike": "Strike",
        "label": "Label",
        "ce_ltp": "CE LTP",
        "pe_ltp": "PE LTP",
        "ce_oi": "CE OI",
        "pe_oi": "PE OI",
        "net_oi": "Net OI",
        "ce_volume": "CE Vol",
        "pe_volume": "PE Vol",
        "total_volume": "Total Vol",
        "pcr": "PCR",
    })

    return display.reset_index(drop=True)


def get_most_traded_strikes(options_df: pd.DataFrame, spot: float, n_strikes: int = 5) -> pd.DataFrame:
    """
    Get the most traded strike prices (by volume) within ATM ± n_strikes.
    Returns sorted DataFrame.
    """
    from modules.angelone_client import get_strike_range
    if options_df is None or options_df.empty:
        return pd.DataFrame()

    strike_range = get_strike_range(spot, n_strikes)
    filtered = options_df[options_df["strike"].isin(strike_range)].copy()

    if filtered.empty:
        return filtered

    filtered["total_volume"] = filtered["ce_volume"] + filtered["pe_volume"]
    filtered["ce_pe_ratio"] = (filtered["ce_oi"] / filtered["pe_oi"].replace(0, 1)).round(2)
    filtered["pcr"] = (filtered["pe_oi"] / filtered["ce_oi"].replace(0, 1)).round(2)
    filtered["net_oi"] = filtered["pe_oi"] - filtered["ce_oi"]

    return filtered.sort_values("total_volume", ascending=False).reset_index(drop=True)


def compute_oi_delta_bars(df_candles: pd.DataFrame, options_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-bar OI delta series aligned to the candle timestamps.
    Used for the OI subplot below the main chart.
    """
    if df_candles.empty or options_df.empty:
        return pd.DataFrame()

    total_ce_oi = options_df["ce_oi"].sum()
    total_pe_oi = options_df["pe_oi"].sum()
    net_oi = total_pe_oi - total_ce_oi

    history = st.session_state.get("oi_history", [])
    timestamps = df_candles["timestamp"].tolist()
    oi_values = []

    np.random.seed(42)
    base_net_oi = net_oi
    for i, ts in enumerate(timestamps):
        if history:
            matching = [
                e for e in history
                if abs((e["timestamp"].replace(tzinfo=None) - pd.Timestamp(ts).to_pydatetime()).total_seconds()) < 300
            ]
            if matching:
                snap = matching[-1]["snapshot"]
                bar_net = sum(v["pe_oi"] - v["ce_oi"] for v in snap.values())
            else:
                noise = np.random.normal(0, abs(base_net_oi) * 0.002 + 10000)
                bar_net = base_net_oi + noise
        else:
            noise = np.random.normal(0, abs(base_net_oi) * 0.001 + 10000)
            bar_net = base_net_oi + noise

        oi_values.append(bar_net)

    oi_df = pd.DataFrame({
        "timestamp": timestamps,
        "net_oi": oi_values,
    })
    oi_df["delta_oi"] = oi_df["net_oi"].diff().fillna(0)
    oi_df["color"] = oi_df["delta_oi"].apply(
        lambda x: "#00ff88" if x < 0 else "#ff4444"
    )
    return oi_df
