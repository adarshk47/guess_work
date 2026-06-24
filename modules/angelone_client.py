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


def _candidate_expiry_strings(ref_date=None) -> list:
    """
    Generate candidate NIFTY expiry strings to probe (next 6 Tuesdays — NSE
    moved the weekly NIFTY expiry day from Thursday to Tuesday).
    Returns list of strings in '19JUN2026' format.
    """
    today = ref_date or datetime.now(IST).date()
    candidates = []
    # If today IS the Tuesday expiry (and still tradable), it must be probed
    # first — otherwise the off-by-one below would skip straight to next week.
    if today.weekday() == 1:
        candidates.append(today.strftime("%d%b%Y").upper())
    d = today
    for _ in range(6):
        days_ahead = (1 - d.weekday()) % 7  # next Tuesday
        if days_ahead == 0:
            days_ahead = 7
        d = d + timedelta(days=days_ahead)
        candidates.append(d.strftime("%d%b%Y").upper())
    return candidates


@st.cache_data(ttl=1800)
def _find_valid_nifty_expiry_via_api() -> str:
    """
    Try candidate expiry strings against AngelOne's optionGreek endpoint.
    The first one that returns non-empty data is the correct live expiry.
    Returns expiry string like '19JUN2026', or '' if none found / not connected.

    AngelOne rate-limits this endpoint (observed: "Access denied because of
    exceeding access rate" when probed back-to-back with no delay) — a short
    pause between candidates is required or every later candidate fails
    regardless of whether its expiry is real.
    """
    obj = get_client()
    if obj is None:
        return ""
    for i, expiry_str in enumerate(_candidate_expiry_strings()):
        if i > 0:
            time.sleep(0.35)
        try:
            gr = obj.optionGreek({"name": "NIFTY", "expirydate": expiry_str})
            # A real expiry returns at least a handful of strikes; an invalid
            # one returns an empty/zero-length list. ">5" was too strict and
            # could reject a thin but valid response — any non-empty data is
            # proof the expiry string itself is valid.
            if gr and gr.get("status") and gr.get("data"):
                logger.info(f"Valid NIFTY expiry confirmed via optionGreek: {expiry_str}")
                return expiry_str
        except Exception:
            pass
    return ""


def get_next_weekly_expiry() -> datetime:
    """
    Get the next NIFTY option expiry.

    Priority:
    1. Validate via optionGreek API (most reliable — confirmed by exchange data)
    2. AngelOne searchScrip symbol parsing
    3. Public NFO scrip master URL
    4. Session-state cache from last successful fetch
    5. Next Tuesday (last resort, may be wrong)
    """
    now = datetime.now(IST)
    today = now.date()
    mkt_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

    def _as_dt(expiry_str: str):
        try:
            d = datetime.strptime(expiry_str, "%d%b%Y").date()
            return datetime.combine(d, datetime.min.time()).replace(tzinfo=IST)
        except Exception:
            return None

    # ── 1. optionGreek validation (most accurate) ─────────────────────────────
    try:
        valid = _find_valid_nifty_expiry_via_api()
        if valid:
            expiry_dt = _as_dt(valid)
            if expiry_dt:
                st.session_state["_last_known_expiry"] = expiry_dt
                st.session_state["_expiry_source"] = "AngelOne optionGreek"
                return expiry_dt
    except Exception as e:
        logger.debug(f"optionGreek expiry validation failed: {e}")

    # ── 2. searchScrip symbol parsing ─────────────────────────────────────────
    try:
        api_expiries = _fetch_expiries_from_api()
        if api_expiries:
            future = sorted(d for d in api_expiries if d >= today)
            if future:
                nearest = future[0]
                if nearest == today and now > mkt_close and len(future) > 1:
                    nearest = future[1]
                expiry_dt = datetime.combine(nearest, datetime.min.time()).replace(tzinfo=IST)
                st.session_state["_last_known_expiry"] = expiry_dt
                st.session_state["_expiry_source"] = "AngelOne searchScrip"
                return expiry_dt
    except Exception as e:
        logger.debug(f"searchScrip expiry failed: {e}")

    # ── 3. Public scrip master ─────────────────────────────────────────────────
    try:
        master_expiries = _load_nifty_expiries()
        if master_expiries:
            future = sorted(d for d in master_expiries if d >= today)
            if future:
                nearest = future[0]
                if nearest == today and now > mkt_close and len(future) > 1:
                    nearest = future[1]
                expiry_dt = datetime.combine(nearest, datetime.min.time()).replace(tzinfo=IST)
                st.session_state["_last_known_expiry"] = expiry_dt
                st.session_state["_expiry_source"] = "scrip master"
                return expiry_dt
    except Exception as e:
        logger.debug(f"Scrip master expiry failed: {e}")

    # ── 4. Session-state cache ─────────────────────────────────────────────────
    cached = st.session_state.get("_last_known_expiry")
    if cached is not None:
        cached_date = cached.date() if hasattr(cached, "date") else cached
        if cached_date >= today:
            return cached

    # ── 5. Next Tuesday (absolute last resort) ────────────────────────────────
    days_ahead = (1 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    tuesday = today + timedelta(days=days_ahead)
    expiry_dt = datetime.combine(tuesday, datetime.min.time()).replace(tzinfo=IST)
    st.session_state["_expiry_source"] = "estimated Tuesday"
    return expiry_dt


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


import re as _re

_SCRIP_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/"
    "OpenAPIScripMaster.json"
)


@st.cache_data(ttl=1800)
def _fetch_expiries_from_api() -> list:
    """
    Parse NIFTY option expiry dates from AngelOne searchScrip symbols.
    Handles: NIFTY19JUN26CE24000 (2-digit yr) and NIFTY19JUN2026CE24000 (4-digit yr).
    """
    obj = get_client()
    if obj is None:
        return []
    try:
        result = obj.searchScrip("NFO", "NIFTY")
        if not (result and result.get("status") and result.get("data")):
            return []
        pat = _re.compile(r"^NIFTY(\d{2})([A-Z]{3})(\d{2,4})(CE|PE)(\d+)$")
        expiry_dates = set()
        for item in result["data"]:
            sym = (item.get("tradingsymbol") or item.get("symbolname") or
                   item.get("symbol") or "").upper()
            m = pat.match(sym)
            if m:
                day, mon, yr = m.group(1), m.group(2), m.group(3)
                yr = yr if len(yr) == 4 else f"20{yr}"
                try:
                    expiry_dates.add(datetime.strptime(f"{day}{mon}{yr}", "%d%b%Y").date())
                except Exception:
                    pass
        return sorted(expiry_dates)
    except Exception as e:
        logger.debug(f"searchScrip expiry fetch failed: {e}")
        return []


@st.cache_data(ttl=300)
def _load_nifty_option_master_from_api(expiry_str: str) -> pd.DataFrame:
    """
    Build option master (token, strike, option_type) via AngelOne searchScrip.
    Used when the public scrip master URL is unreachable.
    """
    obj = get_client()
    if obj is None:
        return pd.DataFrame()
    try:
        expiry_dt = datetime.strptime(expiry_str, "%d%b%Y")
        exp_short = expiry_dt.strftime("%d%b%y").upper()   # 19JUN26
        exp_long = expiry_str.upper()                       # 19JUN2026
        result = obj.searchScrip("NFO", "NIFTY")
        if not (result and result.get("status") and result.get("data")):
            return pd.DataFrame()
        pat = _re.compile(r"^NIFTY(\d{2})([A-Z]{3})(\d{2,4})(CE|PE)(\d+)$")
        rows = []
        for item in result["data"]:
            sym = (item.get("tradingsymbol") or item.get("symbolname") or
                   item.get("symbol") or "").upper()
            token = str(item.get("symboltoken") or item.get("token") or "")
            m = pat.match(sym)
            if not m or not token:
                continue
            day, mon, yr, opt_type, strike_str = m.groups()
            yr_full = yr if len(yr) == 4 else f"20{yr}"
            if f"{day}{mon}{yr_full}" not in (exp_short, exp_long):
                continue
            try:
                rows.append({"strike": float(strike_str), "option_type": opt_type,
                             "token": token, "symbol": sym})
            except Exception:
                pass
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception as e:
        logger.debug(f"API option master fetch failed: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def _load_nifty_master_raw() -> pd.DataFrame:
    """
    Download the NFO scrip master once and return ALL NIFTY index options.
    Columns: strike, option_type (CE/PE), token, symbol, expiry (str),
    expiry_date (date). Cached for an hour. This is a public file — no auth.
    """
    import requests

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
    }
    resp = requests.get(_SCRIP_MASTER_URL, headers=headers, timeout=30)
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
    Return NIFTY index options for the given expiry.
    Tries public scrip master first; falls back to AngelOne searchScrip API.
    """
    # Primary: public scrip master (cached, no auth needed)
    try:
        raw = _load_nifty_master_raw()
        if not raw.empty:
            sub = raw[raw["expiry"] == expiry_str.upper()]
            if not sub.empty:
                return sub[["strike", "option_type", "token", "symbol"]].reset_index(drop=True)
    except Exception as e:
        logger.warning(f"Scrip master option lookup failed: {e}")

    # Fallback: AngelOne searchScrip API (needs auth but no external URL)
    logger.info(f"Falling back to API-based option master for {expiry_str}")
    api_master = _load_nifty_option_master_from_api(expiry_str)
    if not api_master.empty:
        st.session_state["_option_master_source"] = "AngelOne API"
    return api_master


def _chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


@st.cache_data(ttl=30)
def fetch_options_chain(expiry_str: str = None) -> pd.DataFrame:
    """
    Fetch NIFTY options chain (ATM ±12 strikes).

    Two-stage approach so partial data is always better than no data:
      Stage 1 — optionGreek: greeks + IV for all strikes. Needs only
                expiry string (no tokens). Works even if scrip master fails.
      Stage 2 — getMarketData: OI, volume, LTP. Needs token list from
                scrip master or searchScrip.
    Returns merged DataFrame; falls back to last cached result if empty.
    """
    try:
        if expiry_str is None or expiry_str == "---":
            expiry_dt = get_next_weekly_expiry()
            expiry_str = get_expiry_string(expiry_dt)

        obj = get_client()
        spot = fetch_ltp() if obj is not None else 0.0

        # ── Stage 1: Greeks via optionGreek (no tokens needed) ───────────────
        greeks_by_key: dict = {}
        greek_strikes: set = set()
        if obj is not None:
            try:
                gr = obj.optionGreek({"name": "NIFTY", "expirydate": expiry_str})
                if gr and gr.get("status") and gr.get("data"):
                    for g in gr["data"]:
                        try:
                            strike = float(g.get("strikePrice", 0))
                            ot = g.get("optionType", "").upper()
                            greeks_by_key[(strike, ot)] = g
                            greek_strikes.add(strike)
                        except Exception:
                            pass
            except Exception as e:
                logger.error(f"optionGreek error: {e}")

        # ── Stage 2: Market data via tokens (OI / volume / LTP) ──────────────
        master = _load_nifty_option_master(expiry_str) if obj is not None else pd.DataFrame()
        strike_opt_to_md: dict = {}

        if obj is not None and not master.empty:
            all_strikes = sorted(master["strike"].unique())
            if spot and spot > 0:
                atm = min(all_strikes, key=lambda s: abs(s - spot))
                atm_idx = all_strikes.index(atm)
                lo, hi = max(0, atm_idx - 12), min(len(all_strikes), atm_idx + 13)
                keep = set(all_strikes[lo:hi])
                master = master[master["strike"].isin(keep)]

            token_to_meta = {
                r["token"]: (r["strike"], r["option_type"])
                for _, r in master.iterrows()
            }
            md_by_token: dict = {}
            for batch in _chunked(list(token_to_meta.keys()), 50):
                try:
                    md = obj.getMarketData("FULL", {"NFO": batch})
                    if md and md.get("status") and md.get("data"):
                        for item in md["data"].get("fetched", []):
                            md_by_token[str(item.get("symbolToken"))] = item
                except Exception as e:
                    logger.error(f"getMarketData error: {e}")
            for tok, item in md_by_token.items():
                meta = token_to_meta.get(tok)
                if meta:
                    strike_opt_to_md[meta] = item

        # ── Determine which strikes to include ────────────────────────────────
        all_avail = sorted(
            (master["strike"].unique().tolist() if not master.empty else []) or
            sorted(greek_strikes)
        )
        if spot and spot > 0 and all_avail:
            atm = min(all_avail, key=lambda s: abs(s - spot))
            idx = all_avail.index(atm)
            use_strikes = set(all_avail[max(0, idx - 12):idx + 13])
        else:
            use_strikes = set(all_avail[:25]) or greek_strikes

        # ── Assemble rows ─────────────────────────────────────────────────────
        def _depth(item, side):
            try:
                lvls = item.get("depth", {}).get(side, [])
                return float(lvls[0].get("price", 0)) if lvls else 0.0
            except Exception:
                return 0.0

        rows = []
        for strike in sorted(use_strikes):
            row = {"strike": float(strike)}
            for opt in ("ce", "pe"):
                ot = opt.upper()
                md = strike_opt_to_md.get((float(strike), ot))
                g = greeks_by_key.get((float(strike), ot), {})
                row[f"{opt}_oi"]     = int(float(md.get("opnInterest", 0) or 0)) if md else 0
                row[f"{opt}_volume"] = int(float(md.get("tradeVolume", 0) or 0)) if md else 0
                row[f"{opt}_ltp"]    = float(md.get("ltp", 0) or 0) if md else float(g.get("ltp", 0) or 0)
                row[f"{opt}_bid"]    = _depth(md, "buy") if md else 0.0
                row[f"{opt}_ask"]    = _depth(md, "sell") if md else 0.0
                row[f"{opt}_iv"]     = float(g.get("impliedVolatility", 0) or 0)
                row[f"{opt}_delta"]  = float(g.get("delta", 0) or 0)
                row[f"{opt}_gamma"]  = float(g.get("gamma", 0) or 0)
                row[f"{opt}_theta"]  = float(g.get("theta", 0) or 0)
                row[f"{opt}_vega"]   = float(g.get("vega", 0) or 0)
            rows.append(row)

        ao_df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True) if rows else pd.DataFrame()
        ao_has_oi = not ao_df.empty and (ao_df["ce_oi"].sum() + ao_df["pe_oi"].sum()) > 0

        # ── Fallback / supplement: NSE's own public option chain ─────────────
        # AngelOne's getMarketData (OI/volume) stage depends on token lookups
        # (scrip master / searchScrip) that have repeatedly failed even during
        # live market hours. NSE publishes the same chain directly — use it
        # whenever AngelOne has no OI data, merging in AngelOne's greeks when
        # we do have them.
        df = ao_df
        source = "AngelOne" if ao_has_oi else ""
        if not ao_has_oi:
            try:
                from modules.nse_client import fetch_nse_chain_df
                nse_df = fetch_nse_chain_df(expiry_str)
            except Exception as e:
                logger.error(f"NSE chain fallback error: {e}")
                nse_df = pd.DataFrame()

            if not nse_df.empty:
                if not ao_df.empty:
                    # Keep AngelOne's greeks (Stage 1 optionGreek) where present,
                    # otherwise fall back to NSE's Black-Scholes-derived greeks.
                    merged = nse_df.merge(
                        ao_df[["strike", "ce_delta", "ce_gamma", "ce_theta", "ce_vega",
                               "pe_delta", "pe_gamma", "pe_theta", "pe_vega"]],
                        on="strike", how="left", suffixes=("", "_ao"))
                    for opt in ("ce", "pe"):
                        for g in ("delta", "gamma", "theta", "vega"):
                            ao_col = f"{opt}_{g}_ao"
                            if ao_col in merged.columns:
                                use_ao = merged[ao_col].abs() > 0
                                merged.loc[use_ao, f"{opt}_{g}"] = merged.loc[use_ao, ao_col]
                                merged.drop(columns=[ao_col], inplace=True)
                    df = merged
                    source = "AngelOne+NSE"
                else:
                    df = nse_df
                    source = "NSE"

        st.session_state["_chain_source"] = source or "none"

        if df is None or df.empty:
            # Nothing live from either source — return last session's cache.
            cached = st.session_state.get("_last_options_df", pd.DataFrame())
            if not cached.empty:
                logger.info("Options chain empty (AngelOne + NSE) — returning last cached data")
            return cached

        # Cache so next refresh can fall back to this if live fetch fails
        st.session_state["_last_options_df"] = df.copy()
        st.session_state["_last_options_expiry"] = expiry_str
        return df

    except Exception as e:
        logger.error(f"Options chain error: {e}")
        return st.session_state.get("_last_options_df", pd.DataFrame())


def get_options_diagnostics(expiry_str: str = None) -> dict:
    """
    Run each option-data step individually and report what happened.
    Used by the UI debug panel to pinpoint why the options chain is empty.
    Returns a dict of human-readable status strings.
    """
    diag = {}
    obj = get_client()
    diag["connected"] = obj is not None
    diag["last_error"] = get_last_error() or "—"

    if expiry_str is None or expiry_str == "---":
        try:
            expiry_str = get_expiry_string(get_next_weekly_expiry())
        except Exception:
            expiry_str = "?"
    diag["expiry"] = expiry_str
    diag["expiry_source"] = st.session_state.get("_expiry_source", "?")

    if obj is None:
        diag["summary"] = "NOT connected to AngelOne — chart/AngelOne-side data unavailable. Checking NSE fallback only."
    else:
        # 1) optionGreek — greeks for all strikes (only needs expiry string)
        try:
            gr = obj.optionGreek({"name": "NIFTY", "expirydate": expiry_str})
            if gr and gr.get("status"):
                n = len(gr.get("data") or [])
                diag["optionGreek"] = f"OK — {n} rows"
            else:
                msg = gr.get("message") if isinstance(gr, dict) else str(gr)
                diag["optionGreek"] = f"FAIL — {msg}"
        except Exception as e:
            diag["optionGreek"] = f"ERROR — {type(e).__name__}: {e}"

        # 1b) per-candidate expiry probe — shows exactly which Tuesday (if any)
        # AngelOne actually confirms, and what each attempt returned/raised.
        # Paced with a delay — calling this back-to-back is what triggers
        # AngelOne's "exceeding access rate" errors in the first place.
        time.sleep(0.35)
        probe = {}
        for i, cand in enumerate(_candidate_expiry_strings()):
            if i > 0:
                time.sleep(0.35)
            try:
                gr = obj.optionGreek({"name": "NIFTY", "expirydate": cand})
                if gr and gr.get("status"):
                    probe[cand] = f"OK — {len(gr.get('data') or [])} rows"
                else:
                    msg = gr.get("message") if isinstance(gr, dict) else str(gr)
                    probe[cand] = f"FAIL — {msg}"
            except Exception as e:
                probe[cand] = f"ERROR — {type(e).__name__}: {e}"
        diag["expiry_candidate_probe"] = probe

        # 2) scrip master URL (public) — token source
        try:
            raw = _load_nifty_master_raw()
            diag["scrip_master_url"] = f"OK — {len(raw)} NIFTY option rows" if not raw.empty \
                else "EMPTY — URL reachable but no rows parsed"
        except Exception as e:
            diag["scrip_master_url"] = f"ERROR — {type(e).__name__}: {e}"

        # 3) searchScrip API — alternate token source
        time.sleep(0.35)
        try:
            res = obj.searchScrip("NFO", "NIFTY")
            if res and res.get("status"):
                diag["searchScrip"] = f"OK — {len(res.get('data') or [])} symbols"
            else:
                diag["searchScrip"] = f"FAIL — {res.get('message') if isinstance(res, dict) else res}"
        except Exception as e:
            diag["searchScrip"] = f"ERROR — {type(e).__name__}: {e}"

    # 4) NSE public chain — independent of AngelOne entirely
    try:
        from modules.nse_client import (fetch_nse_chain_df, get_nse_expiries,
                                         get_nse_spot, get_nse_last_error)
        nse_expiries = get_nse_expiries()
        diag["nse_expiries"] = [d.strftime("%d-%b-%Y") for d in nse_expiries[:3]]
        diag["nse_spot"] = get_nse_spot()
        diag["nse_last_status"] = get_nse_last_error()
        nse_df = fetch_nse_chain_df(expiry_str)
        diag["nse_chain_rows"] = 0 if nse_df is None or nse_df.empty else len(nse_df)
    except Exception as e:
        diag["nse_chain_rows"] = f"ERROR — {type(e).__name__}: {e}"

    # 5) final assembled chain (AngelOne + NSE merge, whichever produced data)
    if obj is not None:
        time.sleep(0.35)
    df = fetch_options_chain(expiry_str)
    diag["chain_source"] = st.session_state.get("_chain_source", "?")
    diag["chain_rows"] = 0 if df is None or df.empty else len(df)
    if diag["chain_rows"]:
        oi_total = int(df["ce_oi"].sum() + df["pe_oi"].sum())
        diag["chain_has_oi"] = oi_total > 0
        diag["chain_has_greeks"] = bool((df["ce_delta"].abs().sum() +
                                         df["pe_delta"].abs().sum()) > 0)
    return diag


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
