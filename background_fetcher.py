#!/usr/bin/env python3
"""
Headless 24/7 data fetcher — NOT a Streamlit app.

Streamlit Cloud only runs app code while a browser tab has it open, and
st.session_state is per-browser, so data never persisted across reboots or
between visitors. This script is meant to run independently on a schedule
(see .github/workflows/background-fetch.yml) so the option chain, OI history
and paper trades in Firestore stay fresh even when nobody has the app open.

It installs a minimal `streamlit` shim into sys.modules *before* importing
the existing app modules. Those modules only touch a handful of Streamlit
primitives — session_state, secrets and cache_data — all trivially replaced
with plain Python, so the exact same AngelOne/NSE fetch and pattern/paper-
trade/OI logic the live app uses runs here unmodified, with nothing
duplicated to drift out of sync.

Credentials come from environment variables (GitHub Actions secrets), not
Streamlit secrets:
    ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_MPIN, ANGEL_TOTP_SECRET
    FIREBASE_SERVICE_ACCOUNT_JSON  (the full service-account JSON, as one line)
"""

import os
import sys
import json
import types
import logging

import pytz
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("background_fetcher")

IST = pytz.timezone("Asia/Kolkata")


def _install_streamlit_shim():
    shim = types.ModuleType("streamlit")
    shim.session_state = {}
    shim.secrets = {
        "angel_one": {
            "api_key": os.environ.get("ANGEL_API_KEY", ""),
            "client_id": os.environ.get("ANGEL_CLIENT_ID", ""),
            "mpin": os.environ.get("ANGEL_MPIN", ""),
            "totp_secret": os.environ.get("ANGEL_TOTP_SECRET", ""),
        },
        "firebase": json.loads(os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON") or "{}"),
    }

    def cache_data(*args, **kwargs):
        # No-op: this process runs once per cron invocation and exits, so
        # caching across calls has no benefit and skipping it keeps every
        # run honest about fetching live data.
        if args and callable(args[0]):
            return args[0]

        def deco(fn):
            return fn
        return deco
    shim.cache_data = cache_data

    def _noop(*args, **kwargs):
        return None
    for name in ("error", "warning", "info", "success", "spinner"):
        setattr(shim, name, _noop)

    sys.modules["streamlit"] = shim


_install_streamlit_shim()

from modules.angelone_client import (  # noqa: E402
    fetch_candle_data, fetch_options_chain, get_next_weekly_expiry, get_expiry_string,
)
from modules.pattern_detector import detect_all_patterns  # noqa: E402
from modules.paper_trader import (  # noqa: E402
    init_paper_trades, should_add_new_trade, add_paper_trade,
    update_paper_trades, get_atm_option_quote,
)
from modules.oi_analyzer import store_oi_snapshot  # noqa: E402
from modules import firebase_client  # noqa: E402


def run_once():
    now = datetime.now(IST)
    logger.info(f"Background fetch starting at {now.strftime('%Y-%m-%d %H:%M:%S')} IST")

    if not firebase_client.is_configured():
        logger.error(
            "Firestore is not configured (FIREBASE_SERVICE_ACCOUNT_JSON missing/invalid) "
            "— nothing to persist, exiting."
        )
        return

    candles = fetch_candle_data(1, 80)
    if candles is None or candles.empty:
        logger.warning("No candle data — AngelOne login likely failed. Exiting.")
        return
    spot = float(candles["close"].iloc[-1])

    expiry_dt = get_next_weekly_expiry()
    expiry_str = get_expiry_string(expiry_dt)
    options_df = fetch_options_chain(expiry_str)
    chain_rows = 0 if options_df is None else len(options_df)
    logger.info(f"spot={spot} expiry={expiry_str} chain_rows={chain_rows}")

    if options_df is not None and not options_df.empty:
        snapshot = {}
        for _, row in options_df.iterrows():
            strike = int(row["strike"])
            snapshot[strike] = {
                "ce_oi": int(row.get("ce_oi", 0)),
                "pe_oi": int(row.get("pe_oi", 0)),
                "ce_volume": int(row.get("ce_volume", 0)),
                "pe_volume": int(row.get("pe_volume", 0)),
                "net_oi": int(row.get("pe_oi", 0)) - int(row.get("ce_oi", 0)),
            }
        store_oi_snapshot(snapshot)
        logger.info(f"Saved OI snapshot ({len(snapshot)} strikes)")

    signals = detect_all_patterns(candles)
    init_paper_trades()
    new_count = 0
    for signal in signals[-5:]:  # only the most recent few each run
        if not should_add_new_trade(signal.pattern, signal.signal, simulated=False):
            continue
        option_type = "CE" if signal.signal == "BUY" else "PE"
        premium, delta = get_atm_option_quote(
            options_df, spot, option_type, idx_entry=signal.entry
        )
        add_paper_trade(
            signal, signal.pattern, spot,
            source="BACKGROUND", option_premium=premium, option_delta=delta,
        )
        new_count += 1
    update_paper_trades(spot)
    logger.info(f"Run complete: {new_count} new trade(s).")


if __name__ == "__main__":
    run_once()
