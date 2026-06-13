"""
AngelOne SmartAPI Client Wrapper
Handles authentication, session management, and all data fetching.
"""

import streamlit as st
import pyotp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import time
import logging

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

NIFTY_TOKEN = "99926000"
NIFTY_EXCHANGE = "NSE"

INTERVAL_MAP = {
    1: "ONE_MINUTE",
    2: "TWO_MINUTE",
    3: "THREE_MINUTE",
    5: "FIVE_MINUTE",
    10: "TEN_MINUTE",
    15: "FIFTEEN_MINUTE",
    30: "THIRTY_MINUTE",
    60: "ONE_HOUR",
}


def _read_secrets() -> dict:
    """
    Read AngelOne credentials from st.secrets, supporting either section name
    ([angel_one] or [angelone]) and either 'mpin' or 'password' for login.
    """
    section = {}
    for key in ("angel_one", "angelone", "ANGEL_ONE", "ANGELONE"):
        try:
            if key in st.secrets:
                section = st.secrets[key]
                break
        except Exception:
            continue
    # Fall back to a flat layout (keys at top level)
    if not section:
        section = st.secrets

    def g(*names):
        for n in names:
            try:
                if n in section and section[n]:
                    return str(section[n])
            except Exception:
                pass
        return ""

    return {
        "api_key": g("api_key", "apikey", "key"),
        "client_id": g("client_id", "clientid", "client_code", "clientcode"),
        # AngelOne login now uses MPIN; fall back to password for older setups
        "login_pwd": g("mpin", "pin", "password"),
        "totp_secret": g("totp_secret", "totp", "totp_key"),
    }


def get_client(force: bool = False):
    """
    Get or create an AngelOne SmartConnect client session.
    Caches the client in st.session_state. Stores the last error message in
    st.session_state['angel_error'] so the UI can show why login failed.
    Pass force=True to retry a fresh login (used by the Connect button).
    Returns the SmartConnect object or None on failure.
    """
    if force:
        st.session_state.pop("angel_client", None)
        st.session_state["angel_client_valid"] = False

    if st.session_state.get("angel_client") is not None and \
            st.session_state.get("angel_client_valid", False):
        return st.session_state["angel_client"]

    st.session_state["angel_error"] = ""
    try:
        from SmartApi import SmartConnect  # smartapi-python package
    except ImportError as e:
        st.session_state["angel_client_valid"] = False
        st.session_state["angel_error"] = (
            f"smartapi-python (or a dependency like logzero) not installed: {e}"
        )
        return None

    creds = _read_secrets()
    missing = [k for k in ("api_key", "client_id", "login_pwd", "totp_secret")
               if not creds.get(k)]
    if missing:
        st.session_state["angel_client_valid"] = False
        st.session_state["angel_error"] = (
            "Missing credentials in secrets: " + ", ".join(missing) +
            ". Expected a [angel_one] section with api_key, client_id, "
            "mpin (or password) and totp_secret."
        )
        return None

    try:
        obj = SmartConnect(api_key=creds["api_key"])
        totp = pyotp.TOTP(creds["totp_secret"]).now()
        data = obj.generateSession(creds["client_id"], creds["login_pwd"], totp)

        if data and data.get("status"):
            try:
                obj.getfeedToken()
            except Exception:
                pass
            st.session_state["angel_client"] = obj
            st.session_state["angel_client_valid"] = True
            st.session_state["angel_auth_token"] = data["data"]["jwtToken"]
            st.session_state["angel_error"] = ""
            return obj

        # Login returned a failure payload — surface the API message
        msg = ""
        if isinstance(data, dict):
            msg = data.get("message") or data.get("errorcode") or str(data)
        st.session_state["angel_client_valid"] = False
        st.session_state["angel_error"] = f"Login failed: {msg}"
        return None

    except Exception as e:
        logger.error(f"AngelOne login error: {e}")
        st.session_state["angel_client_valid"] = False
        st.session_state["angel_error"] = f"Login error: {e}"
        return None


def get_last_error() -> str:
    """Return the last connection error message (empty string if none)."""
    return st.session_state.get("angel_error", "")


def is_connected() -> bool:
    """Return True if a live authenticated AngelOne API session is active."""
    # Trigger a connection attempt if not yet tried this run
    if "angel_client_valid" not in st.session_state:
        get_client()
    return bool(st.session_state.get("angel_client_valid", False))


def get_data_source() -> str:
    """Return 'LIVE' if connected to AngelOne, otherwise 'DEMO'."""
    return "LIVE" if is_connected() else "DEMO"


def is_market_open() -> bool:
    """Check if NSE market is currently open (9:15 AM - 3:30 PM IST, Mon-Fri)."""
    now = datetime.now(IST)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def get_next_weekly_expiry() -> datetime:
    """
    Get the next weekly NIFTY expiry from the NFO scrip master (authoritative
    list of all listed option expiries). Picks the nearest upcoming expiry.
    Falls back to the cached value, else returns None so the UI shows '---'.
    """
    now = datetime.now(IST)
    today = now.date()

    # Primary source: the public NFO scrip master (no auth needed, reliable)
    try:
        expiries = _load_nifty_expiries()
        future = sorted(d for d in expiries if d >= today)
        if future:
            nearest = future[0]
            # If today is expiry and the session has closed, roll to the next
            if nearest == today:
                mkt_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
                if now > mkt_close and len(future) > 1:
                    nearest = future[1]
            expiry_dt = datetime.combine(nearest, datetime.min.time()).replace(tzinfo=IST)
            st.session_state["_last_known_expiry"] = expiry_dt
            return expiry_dt
    except Exception as e:
        logger.debug(f"Expiry fetch from scrip master failed: {e}")

    # Fallback: use last known expiry from session_state if still valid
    cached = st.session_state.get("_last_known_expiry")
    if cached is not None:
        cached_date = cached.date() if hasattr(cached, "date") else cached
        if cached_date >= today:
            return cached

    # No data available — return None so UI can show "---"
    return None


def get_expiry_string(expiry_dt) -> str:
    """Format expiry date as AngelOne API expects, e.g. '27JUN2024'. Returns '---' if None."""
    if expiry_dt is None:
        return "---"
    return expiry_dt.strftime("%d%b%Y").upper()


def get_expiry_countdown(expiry_dt) -> str:
    """Return human-readable countdown string to expiry. Returns '---' if None."""
    if expiry_dt is None:
        return "---"
    now = datetime.now(IST)
    expiry_close = expiry_dt.replace(hour=15, minute=30, second=0)
    diff = expiry_close - now
    if diff.total_seconds() <= 0:
        return "Expired"
    total_seconds = int(diff.total_seconds())
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    if days == 0:
        return f"Expiry Today! {hours}h {minutes}m remaining"
    return f"{days}d {hours}h {minutes}m"


_EMPTY_CANDLES = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


@st.cache_data(ttl=10)
def fetch_candle_data(interval_minutes: int = 1, lookback_bars: int = 200) -> pd.DataFrame:
    """
    Fetch real OHLCV candlestick data from AngelOne for NIFTY 50.
    When the market is closed, this returns the LAST trading day's session
    (the API request window is widened to bridge weekends/holidays).
    Returns an empty DataFrame if not connected — no simulated data.
    """
    obj = get_client()
    if obj is None:
        return _EMPTY_CANDLES.copy()

    try:
        interval_str = INTERVAL_MAP.get(interval_minutes, "ONE_MINUTE")
        now = datetime.now(IST)

        # Widen the window enough to always include the last completed session,
        # even across a weekend or a string of holidays (look back up to 6 days),
        # while still requesting enough history for the chosen interval.
        lookback_minutes = interval_minutes * lookback_bars
        from_dt = now - timedelta(minutes=lookback_minutes + 30)
        earliest = now - timedelta(days=6)
        if from_dt > earliest:
            from_dt = earliest

        from_str = from_dt.strftime("%Y-%m-%d %H:%M")
        to_str = now.strftime("%Y-%m-%d %H:%M")

        params = {
            "exchange": NIFTY_EXCHANGE,
            "symboltoken": NIFTY_TOKEN,
            "interval": interval_str,
            "fromdate": from_str,
            "todate": to_str,
        }
        response = obj.getCandleData(params)

        if response and response.get("status") and response.get("data"):
            raw = response["data"]
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            df = df.sort_values("timestamp").reset_index(drop=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(inplace=True)
            return df

        return _EMPTY_CANDLES.copy()

    except Exception as e:
        logger.error(f"Candle data error: {e}")
        return _EMPTY_CANDLES.copy()


_SCRIP_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/"
    "OpenAPISymbolMaster.json"
)


@st.cache_data(ttl=3600)
def _load_nifty_master_raw() -> pd.DataFrame:
    """
    Download the NFO scrip master once and return ALL NIFTY index options.
    Columns: strike, option_type (CE/PE), token, symbol, expiry (str),
    expiry_date (date). Cached for an hour. This is a public file — no auth.
    """
    import requests

    resp = requests.get(_SCRIP_MASTER_URL, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for item in data:
        if item.get("name") != "NIFTY":
            continue
        if item.get("instrumenttype") != "OPTIDX":
            continue
        symbol = item.get("symbol", "")
        opt_type = "CE" if symbol.endswith("CE") else "PE" if symbol.endswith("PE") else None
        if opt_type is None:
            continue
        exp_raw = str(item.get("expiry", "")).upper()
        try:
            exp_date = datetime.strptime(exp_raw, "%d%b%Y").date()
        except Exception:
            continue
        try:
            strike = float(item.get("strike", 0)) / 100.0  # master strike is in paise
        except Exception:
            continue
        rows.append({
            "strike": strike,
            "option_type": opt_type,
            "token": str(item.get("token", "")),
            "symbol": symbol,
            "expiry": exp_raw,
            "expiry_date": exp_date,
        })
    return pd.DataFrame(rows)


def _load_nifty_expiries() -> list:
    """Return sorted unique NIFTY option expiry dates from the scrip master."""
    raw = _load_nifty_master_raw()
    if raw.empty:
        return []
    return sorted(set(raw["expiry_date"].tolist()))


def _load_nifty_option_master(expiry_str: str) -> pd.DataFrame:
    """
    Return NIFTY index options for the given expiry from the cached master.
    Columns: strike, option_type (CE/PE), token, symbol.
    """
    raw = _load_nifty_master_raw()
    if raw.empty:
        return pd.DataFrame()
    sub = raw[raw["expiry"] == expiry_str.upper()]
    return sub[["strike", "option_type", "token", "symbol"]].reset_index(drop=True)


def _chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


@st.cache_data(ttl=10)
def fetch_options_chain(expiry_str: str = None) -> pd.DataFrame:
    """
    Fetch the NIFTY options chain (ATM ±12 strikes) from AngelOne:
      • getMarketData(FULL) → OI, volume, LTP, best bid/ask per strike
      • optionGreek         → delta, gamma, theta, vega, IV per strike
    Returns one row per strike with ce_*/pe_* columns.
    Returns an empty DataFrame if not connected — no simulated data.
    """
    try:
        if expiry_str is None:
            expiry_dt = get_next_weekly_expiry()
            expiry_str = get_expiry_string(expiry_dt)

        obj = get_client()
        if obj is None:
            return pd.DataFrame()

        master = _load_nifty_option_master(expiry_str)
        if master.empty:
            logger.error(f"No NIFTY options found in master for expiry {expiry_str}")
            return pd.DataFrame()

        # Limit to ATM ±12 strikes to stay within the 50-token market-data cap
        spot = fetch_ltp()
        all_strikes = sorted(master["strike"].unique())
        if spot and spot > 0:
            atm = min(all_strikes, key=lambda s: abs(s - spot))
            atm_idx = all_strikes.index(atm)
            lo = max(0, atm_idx - 12)
            hi = min(len(all_strikes), atm_idx + 13)
            keep = set(all_strikes[lo:hi])
            master = master[master["strike"].isin(keep)]

        token_to_meta = {
            r["token"]: (r["strike"], r["option_type"])
            for _, r in master.iterrows()
        }
        tokens = list(token_to_meta.keys())

        # ── Market data (OI / volume / LTP / depth) ──────────────────────────
        md_by_token = {}
        for batch in _chunked(tokens, 50):
            try:
                md = obj.getMarketData("FULL", {"NFO": batch})
                if md and md.get("status") and md.get("data"):
                    for item in md["data"].get("fetched", []):
                        md_by_token[str(item.get("symbolToken"))] = item
            except Exception as e:
                logger.error(f"getMarketData error: {e}")

        # ── Greeks (delta / gamma / theta / vega / IV) ───────────────────────
        greeks_by_key = {}
        try:
            gr = obj.optionGreek({"name": "NIFTY", "expirydate": expiry_str})
            if gr and gr.get("status") and gr.get("data"):
                for g in gr["data"]:
                    try:
                        k = (float(g.get("strikePrice", 0)), g.get("optionType", "").upper())
                        greeks_by_key[k] = g
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"optionGreek error: {e}")

        def _best_depth(item, side):
            try:
                lvls = item.get("depth", {}).get(side, [])
                return float(lvls[0].get("price", 0)) if lvls else 0.0
            except Exception:
                return 0.0

        # ── Assemble per-strike rows ─────────────────────────────────────────
        strikes = sorted(master["strike"].unique())
        rows = []
        for strike in strikes:
            row = {"strike": float(strike)}
            for opt in ("ce", "pe"):
                ot = opt.upper()
                sub = master[(master["strike"] == strike) & (master["option_type"] == ot)]
                md = md_by_token.get(sub["token"].iloc[0]) if not sub.empty else None
                g = greeks_by_key.get((float(strike), ot), {})
                if md:
                    row[f"{opt}_oi"] = int(float(md.get("opnInterest", 0) or 0))
                    row[f"{opt}_volume"] = int(float(md.get("tradeVolume", 0) or 0))
                    row[f"{opt}_ltp"] = float(md.get("ltp", 0) or 0)
                    row[f"{opt}_bid"] = _best_depth(md, "buy")
                    row[f"{opt}_ask"] = _best_depth(md, "sell")
                else:
                    row[f"{opt}_oi"] = 0
                    row[f"{opt}_volume"] = 0
                    row[f"{opt}_ltp"] = 0.0
                    row[f"{opt}_bid"] = 0.0
                    row[f"{opt}_ask"] = 0.0
                row[f"{opt}_iv"] = float(g.get("impliedVolatility", 0) or 0)
                row[f"{opt}_delta"] = float(g.get("delta", 0) or 0)
                row[f"{opt}_gamma"] = float(g.get("gamma", 0) or 0)
                row[f"{opt}_theta"] = float(g.get("theta", 0) or 0)
                row[f"{opt}_vega"] = float(g.get("vega", 0) or 0)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.sort_values("strike", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    except Exception as e:
        logger.error(f"Options chain error: {e}")
        return pd.DataFrame()


def get_atm_strike(spot_price: float, step: int = 50) -> int:
    """Round spot price to nearest ATM strike (multiple of step)."""
    return int(round(spot_price / step) * step)


def get_strike_range(spot_price: float, n: int = 5, step: int = 50) -> list:
    """Get ATM ± n strikes."""
    atm = get_atm_strike(spot_price, step)
    return [atm + i * step for i in range(-n, n + 1)]


def _last_trading_day(ref: datetime) -> datetime:
    """Return the most recent trading day (Mon–Fri) on or before ref's date."""
    d = ref
    # If before market open today, the latest completed session is the prior day
    if d.weekday() >= 5:  # weekend -> roll back to Friday
        d = d - timedelta(days=(d.weekday() - 4))
    elif d.hour < 9 or (d.hour == 9 and d.minute < 15):
        d = d - timedelta(days=1)
        while d.weekday() >= 5:
            d = d - timedelta(days=1)
    return d


@st.cache_data(ttl=5)
def fetch_ltp(token: str = NIFTY_TOKEN) -> float:
    """
    Fetch the Last Traded Price for NIFTY 50 from AngelOne.
    When the market is closed this returns the last traded price (last close).
    Returns 0.0 if not connected — no simulated price.
    """
    obj = get_client()
    if obj is None:
        return 0.0

    try:
        response = obj.ltpData(NIFTY_EXCHANGE, "NIFTY 50", token)
        if response and response.get("status"):
            return float(response["data"]["ltp"])
        # Fall back to the last real candle close
        candles = fetch_candle_data(1, 2)
        return float(candles["close"].iloc[-1]) if not candles.empty else 0.0
    except Exception as e:
        logger.error(f"LTP fetch error: {e}")
        candles = fetch_candle_data(1, 2)
        return float(candles["close"].iloc[-1]) if not candles.empty else 0.0
