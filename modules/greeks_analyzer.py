"""
Options Greeks Analyzer for Nifty50.
Analyzes Gamma, Theta, Delta, Vega for ATM ±5 strikes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def analyze_greeks(options_df: pd.DataFrame, spot_price: float, n_strikes: int = 5) -> Dict:
    """
    Analyze options greeks for ATM ±n strikes.
    Returns summary dict with bias and tables.
    """
    if options_df is None or options_df.empty:
        return {"bias": "NEUTRAL", "summary": {}, "table": pd.DataFrame()}

    atm = round(spot_price / 50) * 50
    strikes = [atm + i * 50 for i in range(-n_strikes, n_strikes + 1)]
    chain = options_df[options_df["strike"].isin(strikes)].copy()

    if chain.empty:
        return {"bias": "NEUTRAL", "summary": {}, "table": pd.DataFrame()}

    # Aggregate greeks
    total_ce_gamma = chain["ce_gamma"].sum()
    total_pe_gamma = chain["pe_gamma"].sum()
    total_ce_theta = chain["ce_theta"].sum()
    total_pe_theta = chain["pe_theta"].sum()
    total_ce_vega = chain["ce_vega"].sum()
    total_pe_vega = chain["pe_vega"].sum()

    atm_row = chain[chain["strike"] == atm]
    atm_gamma = atm_row["ce_gamma"].values[0] if not atm_row.empty else 0
    atm_theta_ce = atm_row["ce_theta"].values[0] if not atm_row.empty else 0
    atm_theta_pe = atm_row["pe_theta"].values[0] if not atm_row.empty else 0

    # Premium skew: if CE premiums higher than PE at equidistant strikes → bullish
    otm_ce = chain[chain["strike"] > atm]["ce_ltp"].mean() if not chain[chain["strike"] > atm].empty else 0
    otm_pe = chain[chain["strike"] < atm]["pe_ltp"].mean() if not chain[chain["strike"] < atm].empty else 0
    premium_skew = "CE EXPENSIVE (BEARISH)" if otm_ce > otm_pe * 1.05 else \
                   "PE EXPENSIVE (BULLISH)" if otm_pe > otm_ce * 1.05 else "BALANCED"

    # High gamma at ATM means big move expected
    gamma_signal = "HIGH VOLATILITY EXPECTED" if atm_gamma > 0.003 else \
                   "LOW VOLATILITY" if atm_gamma < 0.001 else "MODERATE VOLATILITY"

    # Theta decay: options losing value faster → favor selling
    avg_theta = (abs(atm_theta_ce) + abs(atm_theta_pe)) / 2
    theta_signal = "HIGH DECAY - SELL OPTIONS" if avg_theta > 15 else \
                   "MODERATE DECAY" if avg_theta > 7 else "LOW DECAY - BUY OPTIONS"

    # Overall bias from multiple signals
    ce_dominance = chain["ce_oi"].sum() > chain["pe_oi"].sum()
    bias = "BEARISH" if ce_dominance else "BULLISH"

    summary = {
        "ATM Strike": int(atm),
        "ATM Gamma": round(float(atm_gamma), 6),
        "Gamma Signal": gamma_signal,
        "ATM Theta (CE)": round(float(atm_theta_ce), 2),
        "ATM Theta (PE)": round(float(atm_theta_pe), 2),
        "Theta Signal": theta_signal,
        "Premium Skew": premium_skew,
        "Overall Bias": bias,
        "Total CE Vega": round(float(total_ce_vega), 2),
        "Total PE Vega": round(float(total_pe_vega), 2),
    }

    # Build per-strike table
    rows = []
    for _, row in chain.iterrows():
        strike = row["strike"]
        label = "ATM" if strike == atm else ("ITM-CE" if strike < atm else "OTM-CE")
        rows.append({
            "Strike": f"{int(strike)} ({label})",
            "CE LTP": f"₹{row['ce_ltp']:.0f}",
            "CE Δ": f"{row['ce_delta']:.3f}",
            "CE Γ": f"{row['ce_gamma']:.5f}",
            "CE Θ": f"{row['ce_theta']:.1f}",
            "CE IV": f"{row['ce_iv']:.1f}%",
            "PE LTP": f"₹{row['pe_ltp']:.0f}",
            "PE Δ": f"{row['pe_delta']:.3f}",
            "PE Γ": f"{row['pe_gamma']:.5f}",
            "PE Θ": f"{row['pe_theta']:.1f}",
            "PE IV": f"{row['pe_iv']:.1f}%",
        })

    table = pd.DataFrame(rows)
    return {"bias": bias, "summary": summary, "table": table}


def get_gamma_exposure(options_df: pd.DataFrame, spot_price: float) -> pd.DataFrame:
    """
    Calculate gamma exposure (GEX) for each strike.
    GEX = gamma * OI * lot_size * spot^2 / 100
    Nifty lot size = 25
    """
    if options_df is None or options_df.empty:
        return pd.DataFrame()

    LOT_SIZE = 25
    df = options_df.copy()
    df["ce_gex"] = df["ce_gamma"] * df["ce_oi"] * LOT_SIZE * spot_price**2 / 100
    df["pe_gex"] = -df["pe_gamma"] * df["pe_oi"] * LOT_SIZE * spot_price**2 / 100
    df["net_gex"] = df["ce_gex"] + df["pe_gex"]

    atm = round(spot_price / 50) * 50
    filtered = df[
        (df["strike"] >= atm - 500) &
        (df["strike"] <= atm + 500)
    ][["strike", "ce_gex", "pe_gex", "net_gex"]].copy()

    filtered["ce_gex"] = (filtered["ce_gex"] / 1e6).round(2)
    filtered["pe_gex"] = (filtered["pe_gex"] / 1e6).round(2)
    filtered["net_gex"] = (filtered["net_gex"] / 1e6).round(2)
    filtered.columns = ["Strike", "CE GEX (M)", "PE GEX (M)", "Net GEX (M)"]
    return filtered.reset_index(drop=True)


def build_greeks_trend_table(options_df: pd.DataFrame, spot_price: float) -> pd.DataFrame:
    """
    Build a summary table showing which side has higher premium/risk.
    """
    if options_df is None or options_df.empty:
        return pd.DataFrame()

    atm = round(spot_price / 50) * 50
    chain = options_df[
        (options_df["strike"] >= atm - 250) &
        (options_df["strike"] <= atm + 250)
    ].copy()

    if chain.empty:
        return pd.DataFrame()

    rows = []
    for metric_name, ce_col, pe_col, bull_if_ce_higher in [
        ("IV", "ce_iv", "pe_iv", False),
        ("Gamma", "ce_gamma", "pe_gamma", True),
        ("Theta", "ce_theta", "pe_theta", True),
        ("Vega", "ce_vega", "pe_vega", True),
        ("LTP", "ce_ltp", "pe_ltp", False),
        ("OI", "ce_oi", "pe_oi", False),
        ("Volume", "ce_volume", "pe_volume", False),
    ]:
        ce_val = chain[ce_col].mean() if ce_col in chain.columns else 0
        pe_val = chain[pe_col].mean() if pe_col in chain.columns else 0
        higher = "CE" if ce_val > pe_val else "PE"
        diff_pct = abs(ce_val - pe_val) / (abs(ce_val) + 1e-9) * 100
        trend = "BEARISH" if higher == "CE" and not bull_if_ce_higher else \
                "BULLISH" if higher == "PE" and not bull_if_ce_higher else \
                "BULLISH" if higher == "CE" and bull_if_ce_higher else "BEARISH"
        rows.append({
            "Metric": metric_name,
            "CE Avg": f"{ce_val:.4f}" if ce_val < 1 else f"{ce_val:.2f}",
            "PE Avg": f"{pe_val:.4f}" if pe_val < 1 else f"{pe_val:.2f}",
            "Higher Side": higher,
            "Diff %": f"{diff_pct:.1f}%",
            "Signal": trend,
        })

    return pd.DataFrame(rows)
