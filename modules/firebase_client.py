"""
Firestore persistence layer — makes paper trades and OI history survive
across Streamlit Cloud reboots and be shared across every visitor, instead
of living only in one browser's st.session_state.

Both the Streamlit app and the headless background_fetcher.py read
credentials the same way: via st.secrets["firebase"]. background_fetcher.py
installs a small `streamlit` shim (session_state/secrets/cache_data) before
importing any app module, populating st.secrets["firebase"] from the
FIREBASE_SERVICE_ACCOUNT_JSON environment variable — so this module never
needs to know which context it's running in.

Without a [firebase] secrets section configured, every function here is a
silent no-op (returns False/empty), so the app keeps working exactly as
before Firebase was wired in — it just won't persist across sessions.
"""

import logging
from datetime import datetime, timedelta

import streamlit as st
import pytz

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

_TRADES_COLLECTION = "paper_trades"
_OI_COLLECTION = "oi_snapshots"


def _read_service_account() -> dict:
    try:
        if "firebase" in st.secrets:
            section = st.secrets["firebase"]
            return {k: section[k] for k in section.keys()}
    except Exception:
        pass
    return {}


def get_db():
    """Lazily create and cache the Firestore client. Returns None if unconfigured."""
    cached = st.session_state.get("_firestore_db")
    if cached is not None:
        return cached
    if st.session_state.get("_firestore_unavailable"):
        return None

    sa_info = _read_service_account()
    if not sa_info:
        st.session_state["_firestore_unavailable"] = True
        return None

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            cred = credentials.Certificate(sa_info)
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        st.session_state["_firestore_db"] = db
        return db
    except Exception as e:
        logger.warning(f"Firestore init failed: {e}")
        st.session_state["_firestore_unavailable"] = True
        return None


def is_configured() -> bool:
    return get_db() is not None


def _trade_doc_id(trade: dict) -> str:
    return f"{trade.get('date', '')}_{trade.get('id', '')}"


def save_trade(trade: dict):
    """Upsert one paper trade into Firestore. Best-effort — failures are swallowed."""
    db = get_db()
    if db is None:
        return
    try:
        payload = dict(trade)
        payload["_saved_at"] = datetime.now(IST).isoformat()
        db.collection(_TRADES_COLLECTION).document(_trade_doc_id(trade)).set(payload)
    except Exception as e:
        logger.debug(f"Firestore save_trade failed: {e}")


def get_recent_trades(minutes: int = 5) -> list:
    """Trades saved within the last `minutes` minutes, newest first."""
    db = get_db()
    if db is None:
        return []
    try:
        cutoff = (datetime.now(IST) - timedelta(minutes=minutes)).isoformat()
        docs = (
            db.collection(_TRADES_COLLECTION)
            .where("_saved_at", ">=", cutoff)
            .order_by("_saved_at", direction="DESCENDING")
            .limit(200)
            .stream()
        )
        return [d.to_dict() for d in docs]
    except Exception as e:
        logger.debug(f"Firestore get_recent_trades failed: {e}")
        return []


def get_todays_trades() -> list:
    """All trades saved today (IST calendar date), oldest first — seeds a fresh session."""
    db = get_db()
    if db is None:
        return []
    try:
        today = datetime.now(IST).strftime("%d-%b-%Y")
        docs = db.collection(_TRADES_COLLECTION).where("date", "==", today).stream()
        trades = [d.to_dict() for d in docs]
        trades.sort(key=lambda t: t.get("id", 0))
        return trades
    except Exception as e:
        logger.debug(f"Firestore get_todays_trades failed: {e}")
        return []


def save_oi_snapshot(timestamp: datetime, snapshot: dict):
    db = get_db()
    if db is None:
        return
    try:
        doc_id = timestamp.strftime("%Y%m%d_%H%M%S")
        db.collection(_OI_COLLECTION).document(doc_id).set({
            "timestamp": timestamp.isoformat(),
            "snapshot": {str(k): v for k, v in snapshot.items()},
        })
    except Exception as e:
        logger.debug(f"Firestore save_oi_snapshot failed: {e}")


def get_oi_history(hours: float = 2) -> list:
    """OI history entries from the last `hours` hours, oldest first."""
    db = get_db()
    if db is None:
        return []
    try:
        cutoff = (datetime.now(IST) - timedelta(hours=hours)).isoformat()
        docs = (
            db.collection(_OI_COLLECTION)
            .where("timestamp", ">=", cutoff)
            .order_by("timestamp")
            .stream()
        )
        out = []
        for d in docs:
            data = d.to_dict()
            ts = datetime.fromisoformat(data["timestamp"])
            snapshot = {int(k): v for k, v in (data.get("snapshot") or {}).items()}
            out.append({"timestamp": ts, "snapshot": snapshot})
        return out
    except Exception as e:
        logger.debug(f"Firestore get_oi_history failed: {e}")
        return []
