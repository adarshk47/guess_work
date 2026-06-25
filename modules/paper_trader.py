"""
Paper Trading Module for Nifty50.
Auto-generates and tracks paper trades based on pattern signals.
Works both during live market AND in post-market simulation mode.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import pytz
from typing import Dict

IST = pytz.timezone("Asia/Kolkata")


def is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def init_paper_trades():
    """
    First call in a session seeds today's trades from Firestore (if
    configured), so a brand-new visitor immediately sees what the background
    fetcher — or another visitor's session — already recorded today.
    """
    if "paper_trades" not in st.session_state:
        from modules.firebase_client import get_todays_trades
        try:
            seeded = get_todays_trades()
        except Exception:
            seeded = []
        st.session_state["paper_trades"] = seeded
    if "paper_trade_counter" not in st.session_state:
        existing_ids = [t.get("id", 0) for t in st.session_state["paper_trades"]]
        st.session_state["paper_trade_counter"] = max(existing_ids) if existing_ids else 0


# Default ATM option delta used when the live chain has no greek data.
DEFAULT_ATM_DELTA = 0.5


def get_atm_option_quote(options_df, spot_price: float, option_type: str,
                         idx_entry: float = None):
    """
    Look up the ATM option's premium (LTP) and delta from the options chain.
    option_type is 'CE' or 'PE'. Returns (premium, delta); (0.0, 0.0) if the
    chain is unavailable.

    If idx_entry is given (the index level when the pattern formed), the
    current ATM premium is back-adjusted to estimate what it was at that time:
        adjusted_prem = current_prem + delta × (idx_entry − current_spot)
    This removes the bias of using today's live LTP for all historical patterns.
    """
    try:
        if options_df is None or options_df.empty or not spot_price:
            return 0.0, 0.0
        atm = round(spot_price / 50) * 50
        idx = (options_df["strike"] - atm).abs().idxmin()
        row = options_df.loc[idx]
        opt = option_type.lower()
        prem = float(row.get(f"{opt}_ltp", 0) or 0)
        delta = float(row.get(f"{opt}_delta", 0) or 0)
        # Adjust for the spot level at pattern formation time
        if idx_entry is not None and spot_price and delta and prem:
            prem = prem + delta * (idx_entry - spot_price)
            prem = max(round(prem, 2), 0.05)
        return prem, delta
    except Exception:
        return 0.0, 0.0


def _premium_levels(signal, option_premium: float, option_delta: float,
                    spot_price: float):
    """
    Convert the pattern's index-level entry/SL/target into OPTION-PREMIUM space.

    We always BUY the option (CE for a bullish signal, PE for a bearish one),
    so a profit happens whenever the premium RISES. The pattern's 'target' is
    therefore the favourable side and its 'stop loss' is the adverse side,
    regardless of CE/PE.

        premium move ≈ |delta| × (index points moved)

    Returns (entry_premium, sl_premium, target_premium).
    """
    idx_entry = float(signal.entry)
    idx_sl = float(signal.stop_loss)
    idx_target = float(signal.target)

    delta = abs(float(option_delta or 0))
    if delta <= 0:
        delta = DEFAULT_ATM_DELTA

    entry_prem = float(option_premium or 0)
    if entry_prem <= 0:
        # Fallback when the live chain has no premium (e.g. not connected):
        # a rough ATM weekly premium ≈ 0.5% of spot. Approximate only.
        entry_prem = max(float(spot_price or 0) * 0.005, 1.0)

    fav_move = abs(idx_target - idx_entry)   # favourable points → premium up
    adv_move = abs(idx_entry - idx_sl)       # adverse points    → premium down

    entry_prem = round(entry_prem, 2)
    target_prem = round(entry_prem + delta * fav_move, 2)
    sl_prem = round(max(entry_prem - delta * adv_move, 0.05), 2)
    return entry_prem, sl_prem, target_prem


def add_paper_trade(signal, pattern_name: str, spot_price: float,
                    source: str = "AUTO", simulated: bool = False,
                    trade_time: datetime = None,
                    option_premium: float = 0.0, option_delta: float = 0.0):
    """
    Add a new paper trade from a pattern signal (live or simulated).

    We only ever BUY options: a bullish signal buys a CE, a bearish signal
    buys a PE. Entry/SL/Target are tracked in OPTION-PREMIUM (₹) terms — not
    index/futures points — using the option's delta to convert.
    """
    init_paper_trades()
    now = trade_time if trade_time is not None else datetime.now(IST)
    if now.tzinfo is None:
        now = IST.localize(now)
    st.session_state["paper_trade_counter"] += 1
    trade_id = st.session_state["paper_trade_counter"]

    atm = round(spot_price / 50) * 50
    # Bullish pattern → buy a CALL; bearish pattern → buy a PUT. Either way the
    # action is BUY (we never sell/write an option).
    direction = signal.signal  # "BUY" (bullish) or "SELL" (bearish)
    option_type = "CE" if direction == "BUY" else "PE"

    entry_prem, sl_prem, target_prem = _premium_levels(
        signal, option_premium, option_delta, spot_price)

    # In simulation, immediately resolve the trade.
    status = "OPEN"
    exit_price = None
    exit_time = None
    pnl = 0.0
    pnl_pct = 0.0
    if simulated:
        rr = float(signal.risk_reward or 1.0)
        if rr >= 1.5:
            status = "PROFIT"
            exit_price = target_prem
        else:
            status = "LOSS"
            exit_price = sl_prem
        pnl = round(exit_price - entry_prem, 2)
        pnl_pct = round(pnl / entry_prem * 100, 2) if entry_prem else 0.0
        exit_time = now.strftime("%H:%M:%S")

    trade = {
        "id": trade_id,
        "time": now.strftime("%H:%M:%S"),
        "date": now.strftime("%d-%b-%Y"),
        "pattern": pattern_name,
        "signal": "BUY",            # we always BUY the option (display)
        "direction": direction,     # underlying bias (internal: BUY/SELL)
        "option": f"NIFTY {int(atm)} {option_type}",
        "entry_spot": round(spot_price, 2),
        # Premium-space levels (what the user trades on)
        "entry": entry_prem,
        "stop_loss": sl_prem,
        "target": target_prem,
        # Underlying index levels — kept for SL/target hit detection only
        "idx_entry": round(float(signal.entry), 2),
        "idx_sl": round(float(signal.stop_loss), 2),
        "idx_target": round(float(signal.target), 2),
        "rr": signal.risk_reward,
        "status": status,
        "exit_price": exit_price,
        "exit_time": exit_time,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "source": "SIM" if simulated else source,
        "confidence": getattr(signal, "confidence", "MEDIUM"),
    }
    st.session_state["paper_trades"].append(trade)
    from modules.firebase_client import save_trade
    try:
        save_trade(trade)
    except Exception:
        pass
    return trade_id


def update_paper_trades(current_spot: float):
    """
    Check open trades and mark them complete if SL or target is hit.

    Hit detection runs on the underlying INDEX level (idx_*), but the recorded
    exit price and P&L are in OPTION-PREMIUM (₹) terms — buying an option means
    a profit when the premium rises (target side) and a loss when it falls
    (stop-loss side), for both CE and PE.
    """
    init_paper_trades()
    now = datetime.now(IST)
    for trade in st.session_state["paper_trades"]:
        if trade["status"] != "OPEN":
            continue
        if not current_spot:
            continue
        entry_prem = trade["entry"]
        sl_prem = trade["stop_loss"]
        target_prem = trade["target"]
        # Underlying index thresholds (fall back to legacy keys if absent)
        idx_sl = trade.get("idx_sl", trade["stop_loss"])
        idx_target = trade.get("idx_target", trade["target"])
        direction = trade.get("direction", trade["signal"])

        def _close(status, exit_prem):
            trade.update(
                status=status, exit_price=round(exit_prem, 2),
                exit_time=now.strftime("%H:%M:%S"),
                pnl=round(exit_prem - entry_prem, 2),
                pnl_pct=round((exit_prem - entry_prem) / entry_prem * 100, 2)
                if entry_prem else 0.0,
            )
            from modules.firebase_client import save_trade
            try:
                save_trade(trade)
            except Exception:
                pass

        if direction == "BUY":          # bullish → bought a CE
            if current_spot <= idx_sl:
                _close("LOSS", sl_prem)
            elif current_spot >= idx_target:
                _close("PROFIT", target_prem)
        else:                            # bearish → bought a PE
            if current_spot >= idx_sl:
                _close("LOSS", sl_prem)
            elif current_spot <= idx_target:
                _close("PROFIT", target_prem)


def get_trades_df() -> pd.DataFrame:
    init_paper_trades()
    if not st.session_state["paper_trades"]:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state["paper_trades"])


def get_paper_trade_summary() -> Dict:
    df = get_trades_df()
    if df.empty:
        return {"total": 0, "open": 0, "profit": 0, "loss": 0,
                "total_pnl": 0.0, "win_rate": 0.0}
    profits = (df["status"] == "PROFIT").sum()
    losses = (df["status"] == "LOSS").sum()
    total_closed = profits + losses
    win_rate = profits / total_closed * 100 if total_closed > 0 else 0
    return {
        "total": len(df),
        "open": (df["status"] == "OPEN").sum(),
        "profit": int(profits),
        "loss": int(losses),
        "total_pnl": round(df["pnl"].sum(), 2),
        "win_rate": round(win_rate, 1),
    }


def should_add_new_trade(pattern_name: str, signal_type: str,
                         simulated: bool = False) -> bool:
    """Avoid duplicate trades for same pattern."""
    init_paper_trades()
    source = "SIM" if simulated else "AUTO"
    recent = [
        t for t in st.session_state["paper_trades"]
        if t["pattern"] == pattern_name
        and t.get("direction", t["signal"]) == signal_type
        and t["source"] == source
    ]
    return len(recent) == 0


def clear_all_trades():
    st.session_state["paper_trades"] = []
    st.session_state["paper_trade_counter"] = 0
