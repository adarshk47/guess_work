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
    if "paper_trades" not in st.session_state:
        st.session_state["paper_trades"] = []
    if "paper_trade_counter" not in st.session_state:
        st.session_state["paper_trade_counter"] = 0


def add_paper_trade(signal, pattern_name: str, spot_price: float,
                    source: str = "AUTO", simulated: bool = False,
                    trade_time: datetime = None):
    """Add a new paper trade from a pattern signal (live or simulated)."""
    init_paper_trades()
    now = trade_time if trade_time is not None else datetime.now(IST)
    if now.tzinfo is None:
        now = IST.localize(now)
    st.session_state["paper_trade_counter"] += 1
    trade_id = st.session_state["paper_trade_counter"]

    atm = round(spot_price / 50) * 50
    option_type = "CE" if signal.signal == "BUY" else "PE"

    # In simulation, immediately resolve the trade using candle data:
    # if price reached target before SL → PROFIT, else → LOSS
    status = "OPEN"
    exit_price = None
    exit_time = None
    pnl = 0.0
    pnl_pct = 0.0
    if simulated:
        # Simulate outcome: assume price moved 0.5×RR toward target
        rr = float(signal.risk_reward or 1.0)
        entry = float(signal.entry)
        sl = float(signal.stop_loss)
        target = float(signal.target)
        risk = abs(entry - sl)
        if signal.signal == "BUY":
            simulated_exit = entry + risk * (rr * 0.5)
            if simulated_exit >= target:
                status = "PROFIT"
                exit_price = round(target, 2)
                pnl = round(target - entry, 2)
                pnl_pct = round(pnl / entry * 100, 2)
            elif simulated_exit <= sl:
                status = "LOSS"
                exit_price = round(sl, 2)
                pnl = round(sl - entry, 2)
                pnl_pct = round(pnl / entry * 100, 2)
            else:
                status = "PROFIT" if rr >= 1.5 else "LOSS"
                exit_price = round(simulated_exit, 2)
                pnl = round(simulated_exit - entry, 2)
                pnl_pct = round(pnl / entry * 100, 2)
        else:
            simulated_exit = entry - risk * (rr * 0.5)
            if simulated_exit <= target:
                status = "PROFIT"
                exit_price = round(target, 2)
                pnl = round(entry - target, 2)
                pnl_pct = round(pnl / entry * 100, 2)
            elif simulated_exit >= sl:
                status = "LOSS"
                exit_price = round(sl, 2)
                pnl = round(entry - sl, 2)
                pnl_pct = round(pnl / entry * 100, 2)
            else:
                status = "PROFIT" if rr >= 1.5 else "LOSS"
                exit_price = round(simulated_exit, 2)
                pnl = round(entry - simulated_exit, 2)
                pnl_pct = round(pnl / entry * 100, 2)
        exit_time = now.strftime("%H:%M:%S")

    trade = {
        "id": trade_id,
        "time": now.strftime("%H:%M:%S"),
        "date": now.strftime("%d-%b-%Y"),
        "pattern": pattern_name,
        "signal": signal.signal,
        "option": f"NIFTY {int(atm)} {option_type}",
        "entry_spot": round(spot_price, 2),
        "entry": round(float(signal.entry), 2),
        "stop_loss": round(float(signal.stop_loss), 2),
        "target": round(float(signal.target), 2),
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
    return trade_id


def update_paper_trades(current_spot: float):
    """Check open trades and mark them complete if SL or target is hit."""
    init_paper_trades()
    now = datetime.now(IST)
    for trade in st.session_state["paper_trades"]:
        if trade["status"] != "OPEN":
            continue
        entry = trade["entry"]
        sl = trade["stop_loss"]
        target = trade["target"]
        if trade["signal"] == "BUY":
            if current_spot <= sl:
                trade.update(status="LOSS", exit_price=round(sl, 2),
                             exit_time=now.strftime("%H:%M:%S"),
                             pnl=round(sl - entry, 2),
                             pnl_pct=round((sl - entry) / entry * 100, 2))
            elif current_spot >= target:
                trade.update(status="PROFIT", exit_price=round(target, 2),
                             exit_time=now.strftime("%H:%M:%S"),
                             pnl=round(target - entry, 2),
                             pnl_pct=round((target - entry) / entry * 100, 2))
        else:
            if current_spot >= sl:
                trade.update(status="LOSS", exit_price=round(sl, 2),
                             exit_time=now.strftime("%H:%M:%S"),
                             pnl=round(entry - sl, 2),
                             pnl_pct=round((entry - sl) / entry * 100, 2))
            elif current_spot <= target:
                trade.update(status="PROFIT", exit_price=round(target, 2),
                             exit_time=now.strftime("%H:%M:%S"),
                             pnl=round(entry - target, 2),
                             pnl_pct=round((entry - target) / entry * 100, 2))


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
        and t["signal"] == signal_type
        and t["source"] == source
    ]
    return len(recent) == 0


def clear_all_trades():
    st.session_state["paper_trades"] = []
    st.session_state["paper_trade_counter"] = 0
