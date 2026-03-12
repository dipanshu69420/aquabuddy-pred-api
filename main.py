"""
Shrimp Feed Prediction API  — with automatic background retraining
=================================================================
Start:  uvicorn main:app --host 0.0.0.0 --port 8000

Auto-retraining fires when EITHER:
  • NEW_RECORDS_THRESHOLD new records have been added since last training, OR
  • RETRAIN_EVERY_HOURS hours have elapsed (nightly by default)

The model is hot-swapped in memory without restarting the server.
"""

import os, json, logging, threading
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ── Retraining imports (firebase_admin is optional; Firestore export is below) ──
try:
    import firebase_admin
    from firebase_admin import credentials, firestore as fs_admin
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

# ── Sklearn (always available) ────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("shrimp_feed")

# =============================================================================
# CONFIG  (override via environment variables)
# =============================================================================
MODEL_PATH       = os.getenv("MODEL_PATH",       "shrimp_feed_model.pkl")
FEATURES_PATH    = os.getenv("FEATURES_PATH",    "model_features.json")
META_PATH        = os.getenv("META_PATH",        "training_meta.json")
FIREBASE_CRED    = os.getenv("FIREBASE_CRED",    "serviceAccountKey.json")

# Retraining thresholds
NEW_RECORDS_THRESHOLD = int(os.getenv("NEW_RECORDS_THRESHOLD", "200"))   # retrain after N new records
RETRAIN_EVERY_HOURS   = int(os.getenv("RETRAIN_EVERY_HOURS",   "24"))    # retrain at least every N hrs
MIN_TRAINING_ROWS     = int(os.getenv("MIN_TRAINING_ROWS",     "500"))   # don't retrain below this

FEATURES = [
    'doc', 'slot_num', 'current_mbw_g', 'current_est_biomass_kg',
    'standard_guideline_feed_g', 'feed_quantity_g', 'prev_feed_quantity',
    'prev_waste_g', 'prev_waste_pct', 'check_tray_avg', 'tray_waste_pct',
    'tithi_score', 'is_molting_day', 'humidity', 'min_temp',
    'yesterday_same_slot_feed',
    # v2 environmental features (filled with 0 for old rows)
    'is_raining', 'precipitation_mm', 'is_high_tide', 'area_feed_reduction_pct',
]
SLOT_MAP = {'07:00': 1, '11:00': 2, '15:00': 3, '19:00': 4}

# =============================================================================
# GLOBAL MODEL STATE  (protected by a threading.RLock for hot-swap safety)
# =============================================================================
_model_lock = threading.RLock()
_model      = None  # loaded on startup

def _load_model():
    global _model
    with _model_lock:
        _model = joblib.load(MODEL_PATH)
    log.info("Model loaded from %s", MODEL_PATH)

def _swap_model(new_model):
    global _model
    with _model_lock:
        _model = new_model
    log.info("Model hot-swapped in memory.")

def _predict(features: np.ndarray) -> float:
    with _model_lock:
        return float(_model.predict(features)[0])

# =============================================================================
# TRAINING METADATA  (tracks when/how the model was last trained)
# =============================================================================
def _load_meta() -> dict:
    if Path(META_PATH).exists():
        with open(META_PATH) as f:
            return json.load(f)
    return {"trained_at": None, "trained_on_rows": 0, "mae": None, "r2": None, "version": 0}

def _save_meta(meta: dict):
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

# =============================================================================
# FIRESTORE EXPORTER
# Pulls completed, non-blind feed_logs from Firestore and returns a DataFrame
# =============================================================================
def _export_from_firestore() -> Optional[pd.DataFrame]:
    if not FIREBASE_AVAILABLE:
        log.warning("firebase_admin not installed — cannot export from Firestore.")
        return None

    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_CRED)
            firebase_admin.initialize_app(cred)

        db = fs_admin.client()
        docs = (
            db.collection("feed_logs")
              .where("feedingType", "!=", "blind")
              .where("status",      "==", "completed")
              .stream()
        )

        rows = []
        for doc in docs:
            d = doc.to_dict()
            # Only rows that have a tray reading AND a known next feed
            if d.get("checkTrayAvg") is None:
                continue
            if d.get("mlPredictedNextFeedG") is None and d.get("feedQuantityG") is None:
                continue
            rows.append(d)

        if not rows:
            log.warning("Firestore export returned 0 usable rows.")
            return None

        df = pd.DataFrame(rows)
        log.info("Firestore export: %d rows", len(df))
        return df

    except Exception as e:
        log.error("Firestore export failed: %s", e)
        return None

# =============================================================================
# FEATURE ENGINEERING  (same logic as original retrain.py)
# =============================================================================
def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ['prev_feed_quantity', 'feed_quantity_g', 'check_tray_avg',
                'check_tray_total', 'prev_waste_g', 'standard_guideline_feed_g',
                'yesterday_same_slot_feed']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0.0

    # Slot time → numeric
    if 'slot_time' in df.columns:
        df['slot_num'] = df['slot_time'].map(SLOT_MAP).fillna(1).astype(int)
    else:
        df['slot_num'] = 1

    df['prev_waste_pct'] = np.where(
        df['prev_feed_quantity'] > 0,
        df['prev_waste_g'] / df['prev_feed_quantity'], 0)

    df['tray_waste_pct'] = np.where(
        df['feed_quantity_g'] > 0,
        (df['check_tray_avg'] * df.get('check_tray_total', 6)) / df['feed_quantity_g'], 0)

    # Target: use existing target_next_feed if available, else next feed given
    if 'target_next_feed' not in df.columns:
        df['target_next_feed'] = df['feed_quantity_g']

    return df

# =============================================================================
# RETRAINING LOGIC
# =============================================================================
_retrain_lock = threading.Lock()   # prevent concurrent retrains

def _should_retrain(meta: dict, current_row_count: int) -> tuple[bool, str]:
    trained_on = meta.get("trained_on_rows", 0)
    trained_at = meta.get("trained_at")

    if trained_at is not None:
        last_trained = datetime.fromisoformat(trained_at)
        hours_since  = (datetime.utcnow() - last_trained).total_seconds() / 3600
        if hours_since >= RETRAIN_EVERY_HOURS:
            return True, f"scheduled ({hours_since:.1f}h since last train)"

    new_records = current_row_count - trained_on
    if new_records >= NEW_RECORDS_THRESHOLD:
        return True, f"{new_records} new records since last train"

    return False, "no retrain needed"


def retrain(source_csv: Optional[str] = None) -> dict:
    """
    Core retraining function. Can be called:
      - Automatically by the scheduler (no args → pulls from Firestore)
      - Via the /admin/retrain endpoint (no args → Firestore, or pass CSV path)
      - From retrain.py CLI (passes CSV path)
    """
    if not _retrain_lock.acquire(blocking=False):
        log.info("Retrain already in progress, skipping.")
        return {"status": "skipped", "reason": "already_running"}

    try:
        log.info("Starting retraining…")

        # 1. Load data
        if source_csv:
            df = pd.read_csv(source_csv)
            df = df[df['is_blind_feeding'] == 0]
        else:
            df = _export_from_firestore()
            if df is None:
                return {"status": "skipped", "reason": "no_data"}

        df = _prepare(df)

        X = df[FEATURES].fillna(0)
        y = pd.to_numeric(df['target_next_feed'], errors='coerce').fillna(0)

        if len(X) < MIN_TRAINING_ROWS:
            log.warning("Only %d rows — skipping retrain (need %d).", len(X), MIN_TRAINING_ROWS)
            return {"status": "skipped", "reason": f"insufficient_data ({len(X)} rows)"}

        # 2. Train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(
            n_estimators=200, max_depth=15,
            min_samples_leaf=3, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        mae = mean_absolute_error(y_test, rf.predict(X_test))
        r2  = r2_score(y_test,  rf.predict(X_test))
        log.info("Retrain complete — MAE: %.1fg | R²: %.4f | Rows: %d", mae, r2, len(X))

        # 3. Persist model
        joblib.dump(rf, MODEL_PATH)

        # 4. Hot-swap in memory (no restart needed)
        _swap_model(rf)

        # 5. Update metadata
        meta = _load_meta()
        meta["trained_at"]      = datetime.utcnow().isoformat()
        meta["trained_on_rows"] = len(X)
        meta["mae"]             = round(mae, 2)
        meta["r2"]              = round(r2, 4)
        meta["version"]         = meta.get("version", 0) + 1
        _save_meta(meta)

        return {
            "status": "success",
            "rows":   len(X),
            "mae":    round(mae, 2),
            "r2":     round(r2, 4),
            "version": meta["version"],
        }

    except Exception as e:
        log.error("Retrain failed: %s", e)
        return {"status": "error", "reason": str(e)}
    finally:
        _retrain_lock.release()


# =============================================================================
# BACKGROUND SCHEDULER  (runs in a daemon thread)
# =============================================================================
def _scheduler_loop():
    """Runs every 30 minutes; fires retrain when thresholds are met."""
    import time
    CHECK_INTERVAL_S = 30 * 60  # check every 30 min

    while True:
        time.sleep(CHECK_INTERVAL_S)
        try:
            meta = _load_meta()

            # Quick row count from Firestore without full export
            row_count = _quick_row_count()
            should, reason = _should_retrain(meta, row_count)

            if should:
                log.info("Auto-retrain triggered: %s", reason)
                result = retrain()
                log.info("Auto-retrain result: %s", result)
            else:
                log.debug("Scheduler check: %s", reason)
        except Exception as e:
            log.error("Scheduler error: %s", e)


def _quick_row_count() -> int:
    """Returns approximate count of completed non-blind feed_logs in Firestore."""
    if not FIREBASE_AVAILABLE:
        return 0
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_CRED)
            firebase_admin.initialize_app(cred)
        db   = fs_admin.client()
        docs = (db.collection("feed_logs")
                  .where("feedingType", "!=", "blind")
                  .where("status", "==", "completed")
                  .select([])         # fetch only doc IDs, no field data
                  .stream())
        return sum(1 for _ in docs)
    except Exception as e:
        log.warning("Quick row count failed: %s", e)
        return 0


# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(
    title="Shrimp Feed Prediction API",
    description="Predicts optimal next feed quantity. Auto-retrains from Firestore.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def startup():
    _load_model()
    # Start background scheduler thread
    t = threading.Thread(target=_scheduler_loop, daemon=True, name="retrain-scheduler")
    t.start()
    log.info("Retrain scheduler started (every %dh or %d new records).",
             RETRAIN_EVERY_HOURS, NEW_RECORDS_THRESHOLD)

# =============================================================================
# SCHEMAS
# =============================================================================
class FeedPredictionRequest(BaseModel):
    doc:                      int
    pond_no:                  int
    slot_time:                str = Field(..., description="07:00 | 11:00 | 15:00 | 19:00")
    current_mbw_g:            float
    current_est_biomass_kg:   float
    standard_guideline_feed_g: float
    feed_quantity_g:          float
    check_tray_avg:           float  = 0.0
    check_tray_total:         float  = 6.0
    prev_feed_quantity:       float
    prev_waste_g:             float  = 0.0
    yesterday_same_slot_feed: float  = 0.0
    tithi_score:              int    = 2
    is_molting_day:           int    = 0
    humidity:                 float  = 0.75
    min_temp:                 float  = 28.0

class FeedPredictionV2Request(FeedPredictionRequest):
    """Extended request with environmental + community signals."""
    is_raining:              int   = 0      # 0 or 1
    precipitation_mm:        float = 0.0   # mm in last 1h
    is_high_tide:            int   = 0      # 0 or 1 for this slot
    area_feed_reduction_pct: float = 0.0   # 0.0–1.0 fraction of farms that reduced
    area:                    str   = ""    # e.g. "Dandi"

class FeedPredictionResponse(BaseModel):
    pond_no:                      int
    doc:                          int
    slot_time:                    str
    predicted_next_feed_g:        float
    rounded_next_feed_g:          float
    prev_waste_g:                 float
    current_tray_waste_g:         float
    waste_flag:                   str
    adjustment_from_guideline_pct: float
    recommendation:               str
    model_version:                int
    # v2 extras (always present; defaults for v1 endpoint)
    is_community_alert:           bool  = False
    community_reduction_pct:      float = 0.0
    rain_reduction_pct:           float = 0.0
    high_tide_increase_pct:       float = 0.0
    tithi_reduction_pct:          float = 0.0

class AreaSignalResponse(BaseModel):
    area:                  str
    slot_time:             str
    date:                  str
    total_ponds:           int
    reduced_ponds:         int
    reduction_fraction:    float   # 0.0 – 1.0
    alert:                 bool
    suggested_reduction_pct: int   # e.g. 15
    message:               str

class RetrainResponse(BaseModel):
    status:  str
    reason:  Optional[str] = None
    rows:    Optional[int] = None
    mae:     Optional[float] = None
    r2:      Optional[float] = None
    version: Optional[int] = None

class ModelStatusResponse(BaseModel):
    model_loaded:     bool
    trained_at:       Optional[str]
    trained_on_rows:  int
    mae:              Optional[float]
    r2:               Optional[float]
    version:          int
    next_retrain_in:  str

# =============================================================================
# HELPERS
# =============================================================================
def _waste_flag(tray_waste_g: float, feed_g: float) -> str:
    if feed_g == 0: return "unknown"
    p = tray_waste_g / feed_g
    if p == 0:      return "no_waste"
    if p < 0.10:    return "low"
    if p < 0.25:    return "moderate"
    return "high"

def _recommendation(flag: str, adj_pct: float, molting: bool,
                    community_alert: bool = False, rain: bool = False,
                    high_tide: bool = False) -> str:
    parts = []
    if community_alert: parts.append("🤝 70%+ area farms reduced feed — community signal applied.")
    if rain:            parts.append("🌧️ Rainfall detected — feed reduced.")
    if high_tide:       parts.append("🌊 High tide — shrimp appetite slightly higher.")
    if molting:         parts.append("🙏 Molting day — shrimp feed less, reduce 20–30%.")
    if flag == "high":  parts.append(f"High waste. Feed reduced by {abs(adj_pct):.1f}%.")
    elif flag == "moderate": parts.append(f"Moderate waste. Slight reduction of {abs(adj_pct):.1f}% applied.")
    elif flag == "no_waste": parts.append("Trays fully consumed — appetite is strong.")
    if not parts:       parts.append("Feed on track with guidelines.")
    return " ".join(parts)

# ---------------------------------------------------------------------------
# Area signal helpers
# ---------------------------------------------------------------------------
def _get_area_feed_logs(area: str, slot_time: str, date_str: str):
    """Returns list of feed_log dicts for the given area+slot+date."""
    if not FIREBASE_AVAILABLE:
        return []
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_CRED)
            firebase_admin.initialize_app(cred)
        db = fs_admin.client()
        docs = (
            db.collection("feed_logs")
              .where("area", "==", area)
              .where("slotTime", "==", slot_time)
              .stream()
        )
        results = []
        for doc in docs:
            d = doc.to_dict()
            # Filter by date client-side (Firestore Timestamp)
            entry_date = d.get("entryDate")
            if entry_date is not None:
                from google.cloud.firestore_v1.base_document import datetime as _dt
                if hasattr(entry_date, 'date'):
                    if entry_date.date().isoformat() != date_str:
                        continue
            results.append(d)
        return results
    except Exception as e:
        log.warning("_get_area_feed_logs failed: %s", e)
        return []

# =============================================================================
# ENDPOINTS
# =============================================================================
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}

# ---------------------------------------------------------------------------
# /area-signal  — community reduction detector
# ---------------------------------------------------------------------------
@app.get("/area-signal", response_model=AreaSignalResponse)
def area_signal(
    area: str,
    slot_time: str = "07:00",
    date: str = "",
):
    """Queries feed logs for 'area' on 'date' at 'slot_time' and returns
    what fraction of farms reduced feed vs their standard guideline."""
    if not date:
        date = datetime.utcnow().date().isoformat()

    logs = _get_area_feed_logs(area, slot_time, date)

    total   = len(logs)
    reduced = 0
    for d in logs:
        guideline = (d.get("standardGuidelineFeedG") or 0)
        actual    = (d.get("feedQuantityG")          or 0)
        if guideline > 0 and actual < guideline * 0.90:
            reduced += 1

    fraction = reduced / total if total > 0 else 0.0
    alert    = fraction >= 0.70

    # Suggested reduction: proportional to how far below 90% the farms went
    suggested = 15 if fraction >= 0.70 else (10 if fraction >= 0.50 else 0)

    msg = (
        f"🤝 {int(fraction*100)}% of {total} farms in {area} reduced feed this slot."
        if total > 0 else
        f"No feed logs found for area '{area}' on {date} at {slot_time}."
    )

    return AreaSignalResponse(
        area=area,
        slot_time=slot_time,
        date=date,
        total_ponds=total,
        reduced_ponds=reduced,
        reduction_fraction=round(fraction, 3),
        alert=alert,
        suggested_reduction_pct=suggested,
        message=msg,
    )

@app.get("/model/status", response_model=ModelStatusResponse)
def model_status():
    meta = _load_meta()
    trained_at = meta.get("trained_at")
    if trained_at:
        elapsed_h = (datetime.utcnow() - datetime.fromisoformat(trained_at)).total_seconds() / 3600
        remaining  = max(0, RETRAIN_EVERY_HOURS - elapsed_h)
        next_str   = f"{remaining:.1f}h" if remaining > 0 else "imminent"
    else:
        next_str = "pending first train"

    return ModelStatusResponse(
        model_loaded    = _model is not None,
        trained_at      = trained_at,
        trained_on_rows = meta.get("trained_on_rows", 0),
        mae             = meta.get("mae"),
        r2              = meta.get("r2"),
        version         = meta.get("version", 0),
        next_retrain_in = next_str,
    )

@app.post("/predict", response_model=FeedPredictionResponse)
def predict(req: FeedPredictionRequest):
    slot_num = SLOT_MAP.get(req.slot_time)
    if slot_num is None:
        raise HTTPException(400, f"slot_time must be one of {list(SLOT_MAP.keys())}")

    prev_waste_pct = req.prev_waste_g / req.prev_feed_quantity if req.prev_feed_quantity > 0 else 0.0
    tray_waste_g   = req.check_tray_avg * req.check_tray_total
    tray_waste_pct = tray_waste_g / req.feed_quantity_g if req.feed_quantity_g > 0 else 0.0

    # Build feature vector — pad the 4 new v2 features with 0
    fv = np.array([[
        req.doc, slot_num, req.current_mbw_g, req.current_est_biomass_kg,
        req.standard_guideline_feed_g, req.feed_quantity_g, req.prev_feed_quantity,
        req.prev_waste_g, prev_waste_pct, req.check_tray_avg, tray_waste_pct,
        req.tithi_score, req.is_molting_day, req.humidity, req.min_temp,
        req.yesterday_same_slot_feed,
        0, 0, 0, 0,   # is_raining, precipitation_mm, is_high_tide, area_feed_reduction_pct
    ]])

    raw     = _predict(fv)
    rounded = round(raw / 100) * 100
    g       = req.standard_guideline_feed_g
    rounded = max(g * 0.5, min(rounded, g * 2.0)) if g > 0 else rounded

    flag    = _waste_flag(tray_waste_g, req.feed_quantity_g)
    adj_pct = ((rounded - g) / g * 100) if g > 0 else 0.0
    rec     = _recommendation(flag, adj_pct, bool(req.is_molting_day))
    ver     = _load_meta().get("version", 0)

    return FeedPredictionResponse(
        pond_no=req.pond_no, doc=req.doc, slot_time=req.slot_time,
        predicted_next_feed_g=raw, rounded_next_feed_g=rounded,
        prev_waste_g=req.prev_waste_g, current_tray_waste_g=tray_waste_g,
        waste_flag=flag, adjustment_from_guideline_pct=round(adj_pct, 2),
        recommendation=rec, model_version=ver,
    )


@app.post("/predict-v2", response_model=FeedPredictionResponse)
def predict_v2(req: FeedPredictionV2Request):
    """Enhanced prediction with rainfall, precipitation, high-tide, and community signals."""
    slot_num = SLOT_MAP.get(req.slot_time)
    if slot_num is None:
        raise HTTPException(400, f"slot_time must be one of {list(SLOT_MAP.keys())}")

    prev_waste_pct = req.prev_waste_g / req.prev_feed_quantity if req.prev_feed_quantity > 0 else 0.0
    tray_waste_g   = req.check_tray_avg * req.check_tray_total
    tray_waste_pct = tray_waste_g / req.feed_quantity_g if req.feed_quantity_g > 0 else 0.0

    fv = np.array([[
        req.doc, slot_num, req.current_mbw_g, req.current_est_biomass_kg,
        req.standard_guideline_feed_g, req.feed_quantity_g, req.prev_feed_quantity,
        req.prev_waste_g, prev_waste_pct, req.check_tray_avg, tray_waste_pct,
        req.tithi_score, req.is_molting_day, req.humidity, req.min_temp,
        req.yesterday_same_slot_feed,
        float(req.is_raining), req.precipitation_mm,
        float(req.is_high_tide), req.area_feed_reduction_pct,
    ]])

    raw = _predict(fv)
    g   = req.standard_guideline_feed_g

    # ── Post-processing multipliers (layered, transparent) ──────────────────
    community_reduction = 0.0
    rain_reduction      = 0.0
    tide_increase       = 0.0
    tithi_reduction     = 0.0

    # 1. Community signal — 70%+ area farms reduced
    community_alert = req.area_feed_reduction_pct >= 0.70
    if community_alert:
        community_reduction = 15.0
        raw *= (1 - community_reduction / 100)

    # 2. Rainfall + precipitation
    if req.is_raining and req.precipitation_mm > 5:
        rain_reduction = 10.0
        raw *= 0.90
    elif req.is_raining and req.precipitation_mm > 0:
        rain_reduction = 5.0
        raw *= 0.95

    # 3. High tide — shrimp are more active and eat more
    if req.is_high_tide:
        tide_increase = 5.0
        raw *= 1.05

    # 4. Tithi score > 3 → further reduction (molting already in model)
    if req.tithi_score >= 5:   # Purnima / Amavasya
        tithi_reduction = 20.0
        raw *= 0.80
    elif req.tithi_score == 4:  # Chaturdashi / Ekadashi
        tithi_reduction = 10.0
        raw *= 0.90

    rounded = round(raw / 100) * 100
    if g > 0:
        rounded = max(g * 0.40, min(rounded, g * 2.0))

    flag    = _waste_flag(tray_waste_g, req.feed_quantity_g)
    adj_pct = ((rounded - g) / g * 100) if g > 0 else 0.0
    rec     = _recommendation(
        flag, adj_pct, bool(req.is_molting_day),
        community_alert=community_alert,
        rain=bool(req.is_raining),
        high_tide=bool(req.is_high_tide),
    )
    ver = _load_meta().get("version", 0)

    return FeedPredictionResponse(
        pond_no=req.pond_no, doc=req.doc, slot_time=req.slot_time,
        predicted_next_feed_g=raw, rounded_next_feed_g=rounded,
        prev_waste_g=req.prev_waste_g, current_tray_waste_g=tray_waste_g,
        waste_flag=flag, adjustment_from_guideline_pct=round(adj_pct, 2),
        recommendation=rec, model_version=ver,
        is_community_alert=community_alert,
        community_reduction_pct=community_reduction,
        rain_reduction_pct=rain_reduction,
        high_tide_increase_pct=tide_increase,
        tithi_reduction_pct=tithi_reduction,
    )


@app.post("/admin/retrain", response_model=RetrainResponse)
def trigger_retrain(background_tasks: BackgroundTasks):
    """
    Manually trigger a retraining from Firestore data.
    Runs in the background so the endpoint returns immediately.
    """
    background_tasks.add_task(retrain)
    return RetrainResponse(status="triggered", reason="manual request")


@app.get("/admin/retrain/status")
def retrain_status():
    meta = _load_meta()
    running = not _retrain_lock.acquire(blocking=False)
    if not running:
        _retrain_lock.release()
    return {**meta, "currently_running": running}