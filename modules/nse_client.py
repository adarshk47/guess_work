"""
NSE (official exchange) option chain client — fallback/primary source for
the options chain when AngelOne's optionGreek/getMarketData/scrip-master
chain (see angelone_client.py) comes back empty, which has proven to happen
even during live market hours.

NSE publishes the same data retail terminals show at
https://www.nseindia.com/option-chain via a public JSON API. The API
requires session cookies obtained by first visiting the site (a plain GET
to the API endpoint returns 401) — this is the standard, widely used
approach for pulling NSE chain data without a broker subscription.

NSE does not publish option greeks (delta/gamma/theta/vega) — only OI,
volume, LTP and IV — so greeks are derived locally via Black-Scholes from
the IV NSE reports, using the underlying spot NSE reports for the same
request (so strike, spot and IV are always consistent with each other).
"""

import math
import logging
from datetime import datetime, timedelta

import pandas as pd
import pytz
import requests
import streamlit as st

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

_NSE_BASE = "https://www.nseindia.com"
_NSE_CHAIN_URL = f"{_NSE_BASE}/api/option-chain-indices?symbol=NIFTY"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{_NSE_BASE}/option-chain",
}

# India risk-free rate approximation, used only for local greek estimation.
_RISK_FREE_RATE = 0.065


def _get_session() -> requests.Session:
    """
    NSE rejects a bare GET to the API (401) unless the client first holds
    cookies from the site itself. Cache the warmed-up session for reuse.
    """
    sess = st.session_state.get("_nse_session")
    if sess is not None:
        return sess
    sess = requests.Session()
    sess.headers.update(_HEADERS)
    try:
        sess.get(_NSE_BASE, timeout=10)
        sess.get(f"{_NSE_BASE}/option-chain", timeout=10)
    except Exception as e:
        logger.debug(f"NSE session warm-up failed: {e}")
    st.session_state["_nse_session"] = sess
    return sess


def _fetch_raw(retry: bool = True) -> dict:
    sess = _get_session()
    try:
        resp = sess.get(_NSE_CHAIN_URL, timeout=10)
        st.session_state["_nse_last_status"] = resp.status_code
        if resp.status_code in (401, 403) and retry:
            # Cookies likely stale — drop the session and retry once.
            st.session_state.pop("_nse_session", None)
            return _fetch_raw(retry=False)
        resp.raise_for_status()
        st.session_state["_nse_last_error"] = ""
        return resp.json()
    except Exception as e:
        logger.debug(f"NSE option chain fetch failed: {e}")
        st.session_state["_nse_last_error"] = f"{type(e).__name__}: {e}"
        return {}


@st.cache_data(ttl=20)
def fetch_nse_chain_raw() -> dict:
    """Raw NSE option-chain-indices JSON for NIFTY, cached for 20s."""
    return _fetch_raw()


def get_nse_last_error() -> str:
    """Last exception/status from the most recent NSE fetch — for diagnostics."""
    status = st.session_state.get("_nse_last_status", "—")
    err = st.session_state.get("_nse_last_error", "")
    return f"HTTP {status}" + (f" — {err}" if err else "")


def get_nse_expiries() -> list:
    """Sorted list of expiry `date` objects NSE currently lists for NIFTY."""
    raw = fetch_nse_chain_raw()
    dates = (raw.get("records") or {}).get("expiryDates") or []
    out = []
    for d in dates:
        try:
            out.append(datetime.strptime(d, "%d-%b-%Y").date())
        except Exception:
            continue
    return sorted(out)


def get_nse_spot() -> float:
    """Underlying NIFTY spot value as reported by NSE in the same payload."""
    raw = fetch_nse_chain_raw()
    try:
        return float((raw.get("records") or {}).get("underlyingValue") or 0)
    except Exception:
        return 0.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_greeks(spot: float, strike: float, t_years: float, iv_pct: float, option_type: str) -> dict:
    """
    Black-Scholes greeks (no dividend yield) computed from NSE's reported IV.
    Returns delta/gamma/theta(per day)/vega(per 1% IV move). Best-effort
    estimate — NSE doesn't publish greeks directly, this is the same model
    every retail options terminal uses to derive them from IV.
    """
    sigma = max(iv_pct, 0.01) / 100.0
    t = max(t_years, 1.0 / (365 * 24))  # floor at ~1 hour to avoid div-by-zero
    if spot <= 0 or strike <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (_RISK_FREE_RATE + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    pdf_d1 = _norm_pdf(d1)

    gamma = pdf_d1 / (spot * sigma * sqrt_t)
    vega = spot * pdf_d1 * sqrt_t / 100.0

    if option_type == "CE":
        delta = _norm_cdf(d1)
        theta = (-spot * pdf_d1 * sigma / (2 * sqrt_t)
                 - _RISK_FREE_RATE * strike * math.exp(-_RISK_FREE_RATE * t) * _norm_cdf(d2)) / 365.0
    else:
        delta = _norm_cdf(d1) - 1.0
        theta = (-spot * pdf_d1 * sigma / (2 * sqrt_t)
                 + _RISK_FREE_RATE * strike * math.exp(-_RISK_FREE_RATE * t) * _norm_cdf(-d2)) / 365.0

    return {"delta": round(delta, 4), "gamma": round(gamma, 6),
            "theta": round(theta, 2), "vega": round(vega, 2)}


def fetch_nse_chain_df(expiry_str: str) -> pd.DataFrame:
    """
    Build a chain DataFrame for `expiry_str` (our internal 'DDMMMYYYY' format,
    e.g. '24JUN2026') sourced from NSE, with the SAME column schema as
    angelone_client.fetch_options_chain(): strike, ce_/pe_ oi, volume, ltp,
    iv, delta, gamma, theta, vega. Greeks are locally derived (see above).
    Returns an empty DataFrame if NSE is unreachable or has no matching rows.
    """
    raw = fetch_nse_chain_raw()
    records = raw.get("records") or {}
    data = records.get("data") or []
    if not data:
        return pd.DataFrame()

    try:
        expiry_dt = datetime.strptime(expiry_str, "%d%b%Y")
    except Exception:
        return pd.DataFrame()
    nse_expiry_fmt = expiry_dt.strftime("%d-%b-%Y")

    spot = get_nse_spot()
    expiry_close = IST.localize(expiry_dt.replace(hour=15, minute=30))
    now = datetime.now(IST)
    t_years = max((expiry_close - now).total_seconds(), 0) / (365 * 24 * 3600)

    rows = []
    for item in data:
        if item.get("expiryDate") != nse_expiry_fmt:
            continue
        strike = float(item.get("strikePrice", 0) or 0)
        if strike <= 0:
            continue
        row = {"strike": strike}
        for side, key in (("ce", "CE"), ("pe", "PE")):
            leg = item.get(key) or {}
            iv = float(leg.get("impliedVolatility", 0) or 0)
            ltp = float(leg.get("lastPrice", 0) or 0)
            row[f"{side}_oi"] = int(leg.get("openInterest", 0) or 0)
            row[f"{side}_volume"] = int(leg.get("totalTradedVolume", 0) or 0)
            row[f"{side}_ltp"] = ltp
            row[f"{side}_bid"] = float(leg.get("bidprice", 0) or 0)
            row[f"{side}_ask"] = float(leg.get("askPrice", 0) or 0)
            row[f"{side}_iv"] = iv
            greeks = _bs_greeks(spot or strike, strike, t_years, iv, key) if iv > 0 else \
                {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
            row[f"{side}_delta"] = greeks["delta"]
            row[f"{side}_gamma"] = greeks["gamma"]
            row[f"{side}_theta"] = greeks["theta"]
            row[f"{side}_vega"] = greeks["vega"]
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    if spot and spot > 0:
        atm = min(df["strike"], key=lambda s: abs(s - spot))
        idx = df.index[df["strike"] == atm][0]
        lo, hi = max(0, idx - 12), min(len(df), idx + 13)
        df = df.iloc[lo:hi].reset_index(drop=True)
    return df
