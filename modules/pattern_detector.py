"""
Chart Pattern Detection Module
Detects candlestick and chart patterns on OHLCV data.
Returns signals with entry, stop_loss, target, and risk/reward ratio.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PatternSignal:
    pattern: str
    signal: str  # "BUY" or "SELL"
    index: int
    timestamp: object
    entry: float
    stop_loss: float
    target: float
    risk_reward: float
    confidence: float  # 0-1
    description: str
    color: str = "green"  # for chart annotation

    def __post_init__(self):
        self.color = "green" if self.signal == "BUY" else "red"
        risk = abs(self.entry - self.stop_loss)
        reward = abs(self.target - self.entry)
        if risk > 0:
            self.risk_reward = round(reward / risk, 2)


def detect_all_patterns(df: pd.DataFrame) -> List[PatternSignal]:
    """
    Run all pattern detectors on OHLCV dataframe.
    Returns list of PatternSignal objects sorted by index.
    """
    if df is None or len(df) < 10:
        return []

    signals = []

    try:
        signals += detect_hammer(df)
        signals += detect_shooting_star(df)
        signals += detect_inverted_hammer(df)
        signals += detect_hanging_man(df)
        signals += detect_doji(df)
        signals += detect_dragonfly_doji(df)
        signals += detect_gravestone_doji(df)
        signals += detect_engulfing(df)
        signals += detect_morning_star(df)
        signals += detect_evening_star(df)
        signals += detect_inside_bar(df)
        signals += detect_pin_bar(df)
        signals += detect_double_top(df)
        signals += detect_double_bottom(df)
        signals += detect_head_and_shoulders(df)
        signals += detect_inverse_head_and_shoulders(df)
        signals += detect_ascending_triangle(df)
        signals += detect_descending_triangle(df)
        signals += detect_bull_flag(df)
        signals += detect_bear_flag(df)
    except Exception as e:
        logger.error(f"Pattern detection error: {e}")

    # Deduplicate: keep highest confidence per index
    seen_indices = {}
    for sig in signals:
        if sig.index not in seen_indices or sig.confidence > seen_indices[sig.index].confidence:
            seen_indices[sig.index] = sig

    return sorted(seen_indices.values(), key=lambda x: x.index)


# ─── Single-Candle Patterns ────────────────────────────────────────────────────

def _body_size(row) -> float:
    return abs(row["close"] - row["open"])


def _candle_range(row) -> float:
    return row["high"] - row["low"]


def _upper_shadow(row) -> float:
    return row["high"] - max(row["open"], row["close"])


def _lower_shadow(row) -> float:
    return min(row["open"], row["close"]) - row["low"]


def detect_hammer(df: pd.DataFrame) -> List[PatternSignal]:
    """
    Hammer: small body at top, long lower shadow (>=2x body), small upper shadow.
    Appears in downtrend. Bullish reversal.
    """
    signals = []
    for i in range(5, len(df)):
        row = df.iloc[i]
        body = _body_size(row)
        total = _candle_range(row)
        lower = _lower_shadow(row)
        upper = _upper_shadow(row)

        if total < 1:
            continue

        # Prior trend must be down
        prior_closes = df["close"].iloc[i - 5:i]
        in_downtrend = prior_closes.iloc[-1] < prior_closes.iloc[0]

        if (
            in_downtrend
            and body > 0
            and lower >= 2 * body
            and upper <= 0.3 * body + 0.1
            and body <= 0.35 * total
        ):
            entry = row["high"] + 0.5
            sl = row["low"] - 5
            target = entry + 2 * (entry - sl)
            signals.append(PatternSignal(
                pattern="Hammer",
                signal="BUY",
                index=i,
                timestamp=df["timestamp"].iloc[i],
                entry=entry,
                stop_loss=sl,
                target=target,
                risk_reward=0,
                confidence=0.72,
                description="Hammer - Bullish reversal after downtrend",
            ))
    return signals


def detect_shooting_star(df: pd.DataFrame) -> List[PatternSignal]:
    """
    Shooting Star: small body at bottom, long upper shadow (>=2x body), small lower shadow.
    Appears in uptrend. Bearish reversal.
    """
    signals = []
    for i in range(5, len(df)):
        row = df.iloc[i]
        body = _body_size(row)
        total = _candle_range(row)
        upper = _upper_shadow(row)
        lower = _lower_shadow(row)

        if total < 1:
            continue

        prior_closes = df["close"].iloc[i - 5:i]
        in_uptrend = prior_closes.iloc[-1] > prior_closes.iloc[0]

        if (
            in_uptrend
            and body > 0
            and upper >= 2 * body
            and lower <= 0.3 * body + 0.1
            and body <= 0.35 * total
        ):
            entry = row["low"] - 0.5
            sl = row["high"] + 5
            target = entry - 2 * (sl - entry)
            signals.append(PatternSignal(
                pattern="Shooting Star",
                signal="SELL",
                index=i,
                timestamp=df["timestamp"].iloc[i],
                entry=entry,
                stop_loss=sl,
                target=target,
                risk_reward=0,
                confidence=0.70,
                description="Shooting Star - Bearish reversal after uptrend",
            ))
    return signals


def detect_inverted_hammer(df: pd.DataFrame) -> List[PatternSignal]:
    """
    Inverted Hammer: appears in downtrend. Small body at bottom, long upper shadow.
    Potential bullish reversal.
    """
    signals = []
    for i in range(5, len(df)):
        row = df.iloc[i]
        body = _body_size(row)
        total = _candle_range(row)
        upper = _upper_shadow(row)
        lower = _lower_shadow(row)

        if total < 1:
            continue

        prior_closes = df["close"].iloc[i - 5:i]
        in_downtrend = prior_closes.iloc[-1] < prior_closes.iloc[0]

        if (
            in_downtrend
            and body > 0
            and upper >= 2 * body
            and lower <= 0.2 * total
            and body <= 0.35 * total
        ):
            entry = row["high"] + 0.5
            sl = row["low"] - 5
            target = entry + 1.5 * (entry - sl)
            signals.append(PatternSignal(
                pattern="Inverted Hammer",
                signal="BUY",
                index=i,
                timestamp=df["timestamp"].iloc[i],
                entry=entry,
                stop_loss=sl,
                target=target,
                risk_reward=0,
                confidence=0.62,
                description="Inverted Hammer - Potential bullish reversal",
            ))
    return signals


def detect_hanging_man(df: pd.DataFrame) -> List[PatternSignal]:
    """
    Hanging Man: same shape as hammer but appears in uptrend. Bearish reversal.
    """
    signals = []
    for i in range(5, len(df)):
        row = df.iloc[i]
        body = _body_size(row)
        total = _candle_range(row)
        lower = _lower_shadow(row)
        upper = _upper_shadow(row)

        if total < 1:
            continue

        prior_closes = df["close"].iloc[i - 5:i]
        in_uptrend = prior_closes.iloc[-1] > prior_closes.iloc[0]

        if (
            in_uptrend
            and body > 0
            and lower >= 2 * body
            and upper <= 0.3 * body + 0.1
            and body <= 0.35 * total
        ):
            entry = row["low"] - 0.5
            sl = row["high"] + 5
            target = entry - 1.5 * (sl - entry)
            signals.append(PatternSignal(
                pattern="Hanging Man",
                signal="SELL",
                index=i,
                timestamp=df["timestamp"].iloc[i],
                entry=entry,
                stop_loss=sl,
                target=target,
                risk_reward=0,
                confidence=0.65,
                description="Hanging Man - Bearish reversal after uptrend",
            ))
    return signals


def detect_doji(df: pd.DataFrame) -> List[PatternSignal]:
    """
    Doji: open ≈ close, can signal reversal.
    """
    signals = []
    for i in range(3, len(df)):
        row = df.iloc[i]
        body = _body_size(row)
        total = _candle_range(row)

        if total < 1:
            continue

        if body / total < 0.05:  # Very small body
            prior_closes = df["close"].iloc[i - 3:i]
            in_uptrend = prior_closes.iloc[-1] > prior_closes.iloc[0]

            if in_uptrend:
                entry = row["low"] - 0.5
                sl = row["high"] + 5
                target = entry - 1.5 * (sl - entry)
                signals.append(PatternSignal(
                    pattern="Doji",
                    signal="SELL",
                    index=i,
                    timestamp=df["timestamp"].iloc[i],
                    entry=entry,
                    stop_loss=sl,
                    target=target,
                    risk_reward=0,
                    confidence=0.55,
                    description="Doji - Indecision, potential reversal",
                ))
            else:
                entry = row["high"] + 0.5
                sl = row["low"] - 5
                target = entry + 1.5 * (entry - sl)
                signals.append(PatternSignal(
                    pattern="Doji",
                    signal="BUY",
                    index=i,
                    timestamp=df["timestamp"].iloc[i],
                    entry=entry,
                    stop_loss=sl,
                    target=target,
                    risk_reward=0,
                    confidence=0.55,
                    description="Doji - Indecision, potential reversal",
                ))
    return signals


def detect_dragonfly_doji(df: pd.DataFrame) -> List[PatternSignal]:
    """Dragonfly Doji: open=high=close, long lower shadow. Bullish."""
    signals = []
    for i in range(3, len(df)):
        row = df.iloc[i]
        body = _body_size(row)
        upper = _upper_shadow(row)
        lower = _lower_shadow(row)
        total = _candle_range(row)

        if total < 1:
            continue

        if (
            body / total < 0.07
            and upper / total < 0.05
            and lower / total > 0.7
        ):
            entry = row["close"] + 0.5
            sl = row["low"] - 5
            target = entry + 2 * (entry - sl)
            signals.append(PatternSignal(
                pattern="Dragonfly Doji",
                signal="BUY",
                index=i,
                timestamp=df["timestamp"].iloc[i],
                entry=entry,
                stop_loss=sl,
                target=target,
                risk_reward=0,
                confidence=0.68,
                description="Dragonfly Doji - Strong bullish reversal",
            ))
    return signals


def detect_gravestone_doji(df: pd.DataFrame) -> List[PatternSignal]:
    """Gravestone Doji: open=low=close, long upper shadow. Bearish."""
    signals = []
    for i in range(3, len(df)):
        row = df.iloc[i]
        body = _body_size(row)
        upper = _upper_shadow(row)
        lower = _lower_shadow(row)
        total = _candle_range(row)

        if total < 1:
            continue

        if (
            body / total < 0.07
            and lower / total < 0.05
            and upper / total > 0.7
        ):
            entry = row["close"] - 0.5
            sl = row["high"] + 5
            target = entry - 2 * (sl - entry)
            signals.append(PatternSignal(
                pattern="Gravestone Doji",
                signal="SELL",
                index=i,
                timestamp=df["timestamp"].iloc[i],
                entry=entry,
                stop_loss=sl,
                target=target,
                risk_reward=0,
                confidence=0.68,
                description="Gravestone Doji - Strong bearish reversal",
            ))
    return signals


def detect_pin_bar(df: pd.DataFrame) -> List[PatternSignal]:
    """
    Pin Bar: long wick (at least 2/3 of total range), small body on opposite end.
    Context-dependent direction.
    """
    signals = []
    for i in range(5, len(df)):
        row = df.iloc[i]
        body = _body_size(row)
        total = _candle_range(row)
        upper = _upper_shadow(row)
        lower = _lower_shadow(row)

        if total < 2:
            continue

        prior_closes = df["close"].iloc[i - 5:i]
        in_uptrend = prior_closes.iloc[-1] > prior_closes.iloc[0]

        # Bearish pin bar (upper wick dominant)
        if (
            in_uptrend
            and upper >= 0.65 * total
            and body <= 0.25 * total
            and lower <= 0.2 * total
        ):
            entry = row["low"] - 0.5
            sl = row["high"] + 5
            target = entry - 2.5 * (sl - entry)
            signals.append(PatternSignal(
                pattern="Pin Bar (Bearish)",
                signal="SELL",
                index=i,
                timestamp=df["timestamp"].iloc[i],
                entry=entry,
                stop_loss=sl,
                target=target,
                risk_reward=0,
                confidence=0.73,
                description="Bearish Pin Bar - Rejection of highs",
            ))
        # Bullish pin bar (lower wick dominant)
        elif (
            not in_uptrend
            and lower >= 0.65 * total
            and body <= 0.25 * total
            and upper <= 0.2 * total
        ):
            entry = row["high"] + 0.5
            sl = row["low"] - 5
            target = entry + 2.5 * (entry - sl)
            signals.append(PatternSignal(
                pattern="Pin Bar (Bullish)",
                signal="BUY",
                index=i,
                timestamp=df["timestamp"].iloc[i],
                entry=entry,
                stop_loss=sl,
                target=target,
                risk_reward=0,
                confidence=0.73,
                description="Bullish Pin Bar - Rejection of lows",
            ))
    return signals


# ─── Two-Candle Patterns ──────────────────────────────────────────────────────

def detect_engulfing(df: pd.DataFrame) -> List[PatternSignal]:
    """Bullish and Bearish Engulfing patterns."""
    signals = []
    for i in range(6, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]

        curr_body = _body_size(curr)
        prev_body = _body_size(prev)

        if prev_body < 1:
            continue

        prior_closes = df["close"].iloc[i - 6:i - 1]
        in_downtrend = prior_closes.iloc[-1] < prior_closes.iloc[0]
        in_uptrend = prior_closes.iloc[-1] > prior_closes.iloc[0]

        # Bullish Engulfing
        if (
            in_downtrend
            and prev["close"] < prev["open"]  # prev is bearish
            and curr["close"] > curr["open"]  # curr is bullish
            and curr["open"] < prev["close"]
            and curr["close"] > prev["open"]
            and curr_body > prev_body
        ):
            entry = curr["close"] + 0.5
            sl = min(curr["low"], prev["low"]) - 5
            target = entry + 2 * (entry - sl)
            signals.append(PatternSignal(
                pattern="Bullish Engulfing",
                signal="BUY",
                index=i,
                timestamp=df["timestamp"].iloc[i],
                entry=entry,
                stop_loss=sl,
                target=target,
                risk_reward=0,
                confidence=0.78,
                description="Bullish Engulfing - Strong reversal signal",
            ))

        # Bearish Engulfing
        elif (
            in_uptrend
            and prev["close"] > prev["open"]  # prev is bullish
            and curr["close"] < curr["open"]  # curr is bearish
            and curr["open"] > prev["close"]
            and curr["close"] < prev["open"]
            and curr_body > prev_body
        ):
            entry = curr["close"] - 0.5
            sl = max(curr["high"], prev["high"]) + 5
            target = entry - 2 * (sl - entry)
            signals.append(PatternSignal(
                pattern="Bearish Engulfing",
                signal="SELL",
                index=i,
                timestamp=df["timestamp"].iloc[i],
                entry=entry,
                stop_loss=sl,
                target=target,
                risk_reward=0,
                confidence=0.78,
                description="Bearish Engulfing - Strong reversal signal",
            ))
    return signals


def detect_inside_bar(df: pd.DataFrame) -> List[PatternSignal]:
    """
    Inside Bar: second candle's high/low is inside first candle's range.
    Breakout trade.
    """
    signals = []
    for i in range(5, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]

        if (
            curr["high"] < prev["high"]
            and curr["low"] > prev["low"]
            and _candle_range(prev) > 2
        ):
            prior_closes = df["close"].iloc[i - 5:i - 1]
            in_uptrend = prior_closes.iloc[-1] > prior_closes.iloc[0]

            if in_uptrend:
                entry = prev["high"] + 0.5
                sl = prev["low"] - 5
                target = entry + 2 * (entry - sl)
                signals.append(PatternSignal(
                    pattern="Inside Bar (Bullish)",
                    signal="BUY",
                    index=i,
                    timestamp=df["timestamp"].iloc[i],
                    entry=entry,
                    stop_loss=sl,
                    target=target,
                    risk_reward=0,
                    confidence=0.65,
                    description="Inside Bar - Bullish breakout pending",
                ))
            else:
                entry = prev["low"] - 0.5
                sl = prev["high"] + 5
                target = entry - 2 * (sl - entry)
                signals.append(PatternSignal(
                    pattern="Inside Bar (Bearish)",
                    signal="SELL",
                    index=i,
                    timestamp=df["timestamp"].iloc[i],
                    entry=entry,
                    stop_loss=sl,
                    target=target,
                    risk_reward=0,
                    confidence=0.65,
                    description="Inside Bar - Bearish breakout pending",
                ))
    return signals


# ─── Three-Candle Patterns ─────────────────────────────────────────────────────

def detect_morning_star(df: pd.DataFrame) -> List[PatternSignal]:
    """Morning Star: downtrend + big bearish + small body + big bullish. Bullish reversal."""
    signals = []
    for i in range(7, len(df)):
        c1 = df.iloc[i - 2]
        c2 = df.iloc[i - 1]
        c3 = df.iloc[i]

        prior = df["close"].iloc[i - 7:i - 2]
        in_downtrend = prior.iloc[-1] < prior.iloc[0]

        if not in_downtrend:
            continue

        c1_bearish = c1["close"] < c1["open"]
        c3_bullish = c3["close"] > c3["open"]
        c1_body = _body_size(c1)
        c2_body = _body_size(c2)
        c3_body = _body_size(c3)

        if (
            c1_bearish
            and c3_bullish
            and c2_body < 0.4 * c1_body
            and c3_body > 0.5 * c1_body
            and c3["close"] > c1["open"] + (c1["close"] - c1["open"]) * 0.3
        ):
            entry = c3["close"] + 0.5
            sl = c2["low"] - 5
            target = entry + 2.5 * (entry - sl)
            signals.append(PatternSignal(
                pattern="Morning Star",
                signal="BUY",
                index=i,
                timestamp=df["timestamp"].iloc[i],
                entry=entry,
                stop_loss=sl,
                target=target,
                risk_reward=0,
                confidence=0.80,
                description="Morning Star - Strong bullish reversal",
            ))
    return signals


def detect_evening_star(df: pd.DataFrame) -> List[PatternSignal]:
    """Evening Star: uptrend + big bullish + small body + big bearish. Bearish reversal."""
    signals = []
    for i in range(7, len(df)):
        c1 = df.iloc[i - 2]
        c2 = df.iloc[i - 1]
        c3 = df.iloc[i]

        prior = df["close"].iloc[i - 7:i - 2]
        in_uptrend = prior.iloc[-1] > prior.iloc[0]

        if not in_uptrend:
            continue

        c1_bullish = c1["close"] > c1["open"]
        c3_bearish = c3["close"] < c3["open"]
        c1_body = _body_size(c1)
        c2_body = _body_size(c2)
        c3_body = _body_size(c3)

        if (
            c1_bullish
            and c3_bearish
            and c2_body < 0.4 * c1_body
            and c3_body > 0.5 * c1_body
            and c3["close"] < c1["open"] + (c1["close"] - c1["open"]) * 0.3
        ):
            entry = c3["close"] - 0.5
            sl = c2["high"] + 5
            target = entry - 2.5 * (sl - entry)
            signals.append(PatternSignal(
                pattern="Evening Star",
                signal="SELL",
                index=i,
                timestamp=df["timestamp"].iloc[i],
                entry=entry,
                stop_loss=sl,
                target=target,
                risk_reward=0,
                confidence=0.80,
                description="Evening Star - Strong bearish reversal",
            ))
    return signals


# ─── Multi-Candle / Chart Patterns ────────────────────────────────────────────

def detect_double_top(df: pd.DataFrame, window: int = 30, tolerance: float = 0.003) -> List[PatternSignal]:
    """Double Top: two similar highs with a valley. Bearish reversal."""
    signals = []
    if len(df) < window:
        return signals

    for i in range(window, len(df)):
        segment = df.iloc[i - window:i]
        highs = segment["high"].values
        lows = segment["low"].values

        # Find two prominent highs
        peak_indices = _find_peaks(highs, min_dist=5)
        if len(peak_indices) < 2:
            continue

        p1_idx = peak_indices[-2]
        p2_idx = peak_indices[-1]
        p1_val = highs[p1_idx]
        p2_val = highs[p2_idx]

        # Check peaks are similar height
        if abs(p1_val - p2_val) / p1_val > tolerance:
            continue

        # Valley between peaks
        valley_lows = lows[p1_idx:p2_idx]
        if len(valley_lows) == 0:
            continue
        neckline = np.min(valley_lows)

        # Current price must be breaking below neckline
        current_close = df["close"].iloc[i - 1]
        if current_close < neckline:
            double_top_high = max(p1_val, p2_val)
            pattern_height = double_top_high - neckline
            entry = neckline - 0.5
            sl = double_top_high + 5
            target = neckline - pattern_height
            signals.append(PatternSignal(
                pattern="Double Top",
                signal="SELL",
                index=i - 1,
                timestamp=df["timestamp"].iloc[i - 1],
                entry=entry,
                stop_loss=sl,
                target=target,
                risk_reward=0,
                confidence=0.75,
                description=f"Double Top at {double_top_high:.0f} - Bearish reversal",
            ))
    return signals


def detect_double_bottom(df: pd.DataFrame, window: int = 30, tolerance: float = 0.003) -> List[PatternSignal]:
    """Double Bottom: two similar lows with a peak. Bullish reversal."""
    signals = []
    if len(df) < window:
        return signals

    for i in range(window, len(df)):
        segment = df.iloc[i - window:i]
        lows = segment["low"].values
        highs = segment["high"].values

        trough_indices = _find_troughs(lows, min_dist=5)
        if len(trough_indices) < 2:
            continue

        t1_idx = trough_indices[-2]
        t2_idx = trough_indices[-1]
        t1_val = lows[t1_idx]
        t2_val = lows[t2_idx]

        if abs(t1_val - t2_val) / t1_val > tolerance:
            continue

        peak_highs = highs[t1_idx:t2_idx]
        if len(peak_highs) == 0:
            continue
        neckline = np.max(peak_highs)

        current_close = df["close"].iloc[i - 1]
        if current_close > neckline:
            double_bottom_low = min(t1_val, t2_val)
            pattern_height = neckline - double_bottom_low
            entry = neckline + 0.5
            sl = double_bottom_low - 5
            target = neckline + pattern_height
            signals.append(PatternSignal(
                pattern="Double Bottom",
                signal="BUY",
                index=i - 1,
                timestamp=df["timestamp"].iloc[i - 1],
                entry=entry,
                stop_loss=sl,
                target=target,
                risk_reward=0,
                confidence=0.75,
                description=f"Double Bottom at {double_bottom_low:.0f} - Bullish reversal",
            ))
    return signals


def detect_head_and_shoulders(df: pd.DataFrame, window: int = 40) -> List[PatternSignal]:
    """Head & Shoulders: left shoulder, head (higher), right shoulder. Bearish."""
    signals = []
    if len(df) < window:
        return signals

    for i in range(window, len(df)):
        segment = df.iloc[i - window:i]
        highs = segment["high"].values
        lows = segment["low"].values

        peaks = _find_peaks(highs, min_dist=4)
        if len(peaks) < 3:
            continue

        ls, head, rs = peaks[-3], peaks[-2], peaks[-1]
        ls_val, head_val, rs_val = highs[ls], highs[head], highs[rs]

        # Head must be highest, shoulders roughly equal
        if not (head_val > ls_val and head_val > rs_val):
            continue
        if abs(ls_val - rs_val) / ls_val > 0.015:
            continue

        # Neckline from troughs between shoulders and head
        troughs = _find_troughs(lows[ls:rs + 1], min_dist=2)
        if len(troughs) < 2:
            continue
        neckline = np.mean([lows[ls + troughs[0]], lows[ls + troughs[-1]]])

        current_close = df["close"].iloc[i - 1]
        if current_close < neckline:
            pattern_height = head_val - neckline
            entry = neckline - 0.5
            sl = rs_val + 5
            target = neckline - pattern_height
            signals.append(PatternSignal(
                pattern="Head & Shoulders",
                signal="SELL",
                index=i - 1,
                timestamp=df["timestamp"].iloc[i - 1],
                entry=entry,
                stop_loss=sl,
                target=target,
                risk_reward=0,
                confidence=0.82,
                description=f"Head & Shoulders - Breakdown below neckline {neckline:.0f}",
            ))
    return signals


def detect_inverse_head_and_shoulders(df: pd.DataFrame, window: int = 40) -> List[PatternSignal]:
    """Inverse H&S: bullish reversal pattern."""
    signals = []
    if len(df) < window:
        return signals

    for i in range(window, len(df)):
        segment = df.iloc[i - window:i]
        highs = segment["high"].values
        lows = segment["low"].values

        troughs = _find_troughs(lows, min_dist=4)
        if len(troughs) < 3:
            continue

        ls, head, rs = troughs[-3], troughs[-2], troughs[-1]
        ls_val, head_val, rs_val = lows[ls], lows[head], lows[rs]

        if not (head_val < ls_val and head_val < rs_val):
            continue
        if abs(ls_val - rs_val) / ls_val > 0.015:
            continue

        peaks = _find_peaks(highs[ls:rs + 1], min_dist=2)
        if len(peaks) < 2:
            continue
        neckline = np.mean([highs[ls + peaks[0]], highs[ls + peaks[-1]]])

        current_close = df["close"].iloc[i - 1]
        if current_close > neckline:
            pattern_height = neckline - head_val
            entry = neckline + 0.5
            sl = rs_val - 5
            target = neckline + pattern_height
            signals.append(PatternSignal(
                pattern="Inv. Head & Shoulders",
                signal="BUY",
                index=i - 1,
                timestamp=df["timestamp"].iloc[i - 1],
                entry=entry,
                stop_loss=sl,
                target=target,
                risk_reward=0,
                confidence=0.82,
                description=f"Inv. H&S - Breakout above neckline {neckline:.0f}",
            ))
    return signals


def detect_ascending_triangle(df: pd.DataFrame, window: int = 30) -> List[PatternSignal]:
    """Ascending Triangle: flat resistance + rising support. Bullish breakout."""
    signals = []
    if len(df) < window:
        return signals

    for i in range(window, len(df)):
        segment = df.iloc[i - window:i]
        highs = segment["high"].values
        lows = segment["low"].values

        # Resistance: flat highs
        recent_highs = highs[-10:]
        resistance = np.mean(recent_highs)
        high_std = np.std(recent_highs)

        # Support: rising lows
        x = np.arange(len(lows))
        coeffs = np.polyfit(x, lows, 1)
        slope = coeffs[0]

        if slope > 0 and high_std / resistance < 0.005:
            current_close = df["close"].iloc[i - 1]
            if current_close > resistance:
                sl = lows[-1] - 5
                target = resistance + (resistance - np.min(lows[-window:]))
                signals.append(PatternSignal(
                    pattern="Ascending Triangle",
                    signal="BUY",
                    index=i - 1,
                    timestamp=df["timestamp"].iloc[i - 1],
                    entry=resistance + 0.5,
                    stop_loss=sl,
                    target=target,
                    risk_reward=0,
                    confidence=0.73,
                    description=f"Ascending Triangle breakout above {resistance:.0f}",
                ))
    return signals


def detect_descending_triangle(df: pd.DataFrame, window: int = 30) -> List[PatternSignal]:
    """Descending Triangle: flat support + falling resistance. Bearish breakdown."""
    signals = []
    if len(df) < window:
        return signals

    for i in range(window, len(df)):
        segment = df.iloc[i - window:i]
        highs = segment["high"].values
        lows = segment["low"].values

        recent_lows = lows[-10:]
        support = np.mean(recent_lows)
        low_std = np.std(recent_lows)

        x = np.arange(len(highs))
        coeffs = np.polyfit(x, highs, 1)
        slope = coeffs[0]

        if slope < 0 and low_std / support < 0.005:
            current_close = df["close"].iloc[i - 1]
            if current_close < support:
                sl = highs[-1] + 5
                target = support - (np.max(highs[-window:]) - support)
                signals.append(PatternSignal(
                    pattern="Descending Triangle",
                    signal="SELL",
                    index=i - 1,
                    timestamp=df["timestamp"].iloc[i - 1],
                    entry=support - 0.5,
                    stop_loss=sl,
                    target=target,
                    risk_reward=0,
                    confidence=0.73,
                    description=f"Descending Triangle breakdown below {support:.0f}",
                ))
    return signals


def detect_bull_flag(df: pd.DataFrame, pole_bars: int = 10, flag_bars: int = 10) -> List[PatternSignal]:
    """Bull Flag: strong up move (pole) + consolidation (flag). Bullish continuation."""
    signals = []
    if len(df) < pole_bars + flag_bars + 5:
        return signals

    for i in range(pole_bars + flag_bars, len(df)):
        pole = df.iloc[i - pole_bars - flag_bars:i - flag_bars]
        flag = df.iloc[i - flag_bars:i]

        pole_gain = (pole["close"].iloc[-1] - pole["close"].iloc[0]) / pole["close"].iloc[0]
        if pole_gain < 0.01:
            continue

        flag_low = flag["low"].min()
        flag_high = flag["high"].max()
        flag_range = flag_high - flag_low

        # Flag should retrace less than 50% of pole
        pole_range = pole["high"].max() - pole["low"].min()
        if flag_range > 0.5 * pole_range:
            continue

        # Slight downward or sideways flag
        flag_slope = np.polyfit(range(len(flag)), flag["close"].values, 1)[0]
        if flag_slope > 0.2:
            continue

        current_close = df["close"].iloc[i - 1]
        if current_close > flag_high:
            sl = flag_low - 5
            target = flag_high + pole_range
            signals.append(PatternSignal(
                pattern="Bull Flag",
                signal="BUY",
                index=i - 1,
                timestamp=df["timestamp"].iloc[i - 1],
                entry=flag_high + 0.5,
                stop_loss=sl,
                target=target,
                risk_reward=0,
                confidence=0.76,
                description="Bull Flag - Bullish continuation breakout",
            ))
    return signals


def detect_bear_flag(df: pd.DataFrame, pole_bars: int = 10, flag_bars: int = 10) -> List[PatternSignal]:
    """Bear Flag: strong down move (pole) + consolidation (flag). Bearish continuation."""
    signals = []
    if len(df) < pole_bars + flag_bars + 5:
        return signals

    for i in range(pole_bars + flag_bars, len(df)):
        pole = df.iloc[i - pole_bars - flag_bars:i - flag_bars]
        flag = df.iloc[i - flag_bars:i]

        pole_loss = (pole["close"].iloc[0] - pole["close"].iloc[-1]) / pole["close"].iloc[0]
        if pole_loss < 0.01:
            continue

        flag_low = flag["low"].min()
        flag_high = flag["high"].max()
        flag_range = flag_high - flag_low

        pole_range = pole["high"].max() - pole["low"].min()
        if flag_range > 0.5 * pole_range:
            continue

        # Slight upward or sideways flag
        flag_slope = np.polyfit(range(len(flag)), flag["close"].values, 1)[0]
        if flag_slope < -0.2:
            continue

        current_close = df["close"].iloc[i - 1]
        if current_close < flag_low:
            sl = flag_high + 5
            target = flag_low - pole_range
            signals.append(PatternSignal(
                pattern="Bear Flag",
                signal="SELL",
                index=i - 1,
                timestamp=df["timestamp"].iloc[i - 1],
                entry=flag_low - 0.5,
                stop_loss=sl,
                target=target,
                risk_reward=0,
                confidence=0.76,
                description="Bear Flag - Bearish continuation breakdown",
            ))
    return signals


# ─── Helper utilities ─────────────────────────────────────────────────────────

def _find_peaks(arr: np.ndarray, min_dist: int = 3) -> List[int]:
    """Find local maxima indices with minimum distance between them."""
    peaks = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_dist:
                peaks.append(i)
            elif arr[i] > arr[peaks[-1]]:
                peaks[-1] = i
    return peaks


def _find_troughs(arr: np.ndarray, min_dist: int = 3) -> List[int]:
    """Find local minima indices with minimum distance between them."""
    troughs = []
    for i in range(1, len(arr) - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            if not troughs or (i - troughs[-1]) >= min_dist:
                troughs.append(i)
            elif arr[i] < arr[troughs[-1]]:
                troughs[-1] = i
    return troughs
