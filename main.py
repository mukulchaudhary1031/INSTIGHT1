# ── CRITICAL: matplotlib backend MUST be set before any other import ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
import os, uuid, traceback, hashlib, secrets
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import razorpay

from database import engine, get_db, Base
from models import User, SavedDataset
from data_processor import load_file, clean_data, get_dataset_info
from ml_engine import train_classification, train_regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from visualizer import generate_eda_visualizations, generate_comparison_viz, compute_kpis
from chatbot import get_chat_response, build_dataset_context

Base.metadata.create_all(bind=engine)


# ── Startup: pre-build matplotlib font cache so first request never crashes ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[InsightIQ] Pre-building matplotlib font cache...")
    try:
        matplotlib.font_manager._load_fontmanager(try_read_cache=False)
        fig, ax = plt.subplots()
        plt.close(fig)
        print("[InsightIQ] Font cache ready ✅")
    except Exception as e:
        print(f"[InsightIQ] Font cache warning (non-fatal): {e}")
    yield


app = FastAPI(title="InsightIQ Platform", version="3.1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# ── FIX: Use /tmp on Render (ephemeral filesystem, only /tmp is writable) ──
UPLOAD_DIR  = Path("/tmp/insightiq_uploads");  UPLOAD_DIR.mkdir(exist_ok=True)
CLEANED_DIR = Path("/tmp/insightiq_cleaned");  CLEANED_DIR.mkdir(exist_ok=True)

# ── Session store ──
store: dict = {}
MAX_SESSIONS = 100

def _evict_old_sessions():
    if len(store) > MAX_SESSIONS:
        keys = list(store.keys())
        for k in keys[:len(store) - MAX_SESSIONS]:
            try: del store[k]
            except: pass

RAZORPAY_KEY_ID     = os.environ.get("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET")

if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET: raise ValueError("RAZORPAY keys not set in environment variables")

PLAN_AMOUNT   = 249900
FREE_UPLOADS  = 2
PAID_UPLOADS  = 10


# ════════════════════════════════════════
# AUTH HELPERS
# ════════════════════════════════════════

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def get_user_by_token(token: str, db: Session):
    if not token:
        return None
    return db.query(User).filter(User.token == token).first()

def upload_status(user: User) -> dict:
    if user is None:
        return {"allowed": False, "reason": "login_required", "uploads_left": 0}
    now = datetime.utcnow()
    if user.is_subscribed and user.subscription_end and user.subscription_end > now:
        left = PAID_UPLOADS - user.upload_count
        if left > 0:
            return {"allowed": True, "reason": "subscribed", "uploads_left": left}
        return {"allowed": False, "reason": "monthly_limit_reached", "uploads_left": 0}
    if user.upload_count < FREE_UPLOADS:
        return {"allowed": True, "reason": "free_trial",
                "uploads_left": FREE_UPLOADS - user.upload_count}
    return {"allowed": False, "reason": "trial_expired", "uploads_left": 0}


# ════════════════════════════════════════
# HOME
# ════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    with open("templates/index.html", "r") as f:
        content = f.read()
    return HTMLResponse(content=content)


# ════════════════════════════════════════
# AUTH
# ════════════════════════════════════════

@app.post("/register")
async def register(request: Request, db: Session = Depends(get_db)):
    body     = await request.json()
    email    = body.get("email", "").strip().lower()
    password = body.get("password", "").strip()
    if not email or not password:
        raise HTTPException(400, "Email and password required.")
    if len(password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(400, "Email already registered.")
    token = secrets.token_urlsafe(32)
    user  = User(email=email, password_hash=hash_password(password),
                 token=token, upload_count=0, is_subscribed=False)
    db.add(user); db.commit(); db.refresh(user)
    return JSONResponse({
        "status": "ok", "token": token, "email": email,
        "upload_count": 0, "is_subscribed": False, "uploads_left": FREE_UPLOADS,
        "message": f"Welcome! You have {FREE_UPLOADS} free uploads."
    })


@app.post("/login")
async def login(request: Request, db: Session = Depends(get_db)):
    body     = await request.json()
    email    = body.get("email", "").strip().lower()
    password = body.get("password", "").strip()
    user = db.query(User).filter(User.email == email).first()
    if not user or user.password_hash != hash_password(password):
        raise HTTPException(401, "Invalid email or password.")
    token = secrets.token_urlsafe(32)
    user.token = token; db.commit()
    info = upload_status(user)
    return JSONResponse({
        "status": "ok", "token": token, "email": email,
        "upload_count": user.upload_count,
        "is_subscribed": user.is_subscribed,
        "uploads_left": info["uploads_left"],
        "subscription_end": str(user.subscription_end) if user.subscription_end else None,
    })


@app.get("/me")
async def get_me(request: Request, db: Session = Depends(get_db)):
    token = request.headers.get("X-Auth-Token", "")
    user  = get_user_by_token(token, db)
    if not user:
        raise HTTPException(401, "Not authenticated.")
    info = upload_status(user)
    return JSONResponse({
        "status": "ok", "email": user.email,
        "upload_count": user.upload_count,
        "is_subscribed": user.is_subscribed,
        "uploads_left": info["uploads_left"],
        "subscription_end": str(user.subscription_end) if user.subscription_end else None,
    })


# ════════════════════════════════════════
# PAYMENT
# ════════════════════════════════════════

@app.post("/create-order")
async def create_order(request: Request, db: Session = Depends(get_db)):
    token = request.headers.get("X-Auth-Token", "")
    user  = get_user_by_token(token, db)
    if not user:
        raise HTTPException(401, "Please login first.")
    try:
        client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
        order  = client.order.create({
            "amount": PLAN_AMOUNT, "currency": "INR",
            "receipt": f"rcpt_{user.id}_{uuid.uuid4().hex[:8]}",
            "notes": {"user_id": str(user.id), "email": user.email}
        })
        user.razorpay_order_id = order["id"]; db.commit()
        return JSONResponse({"status": "ok", "order_id": order["id"],
                             "amount": PLAN_AMOUNT, "currency": "INR", "key_id": RAZORPAY_KEY_ID})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Order creation failed: {e}")


@app.post("/verify-payment")
async def verify_payment(request: Request, db: Session = Depends(get_db)):
    token = request.headers.get("X-Auth-Token", "")
    user  = get_user_by_token(token, db)
    if not user:
        raise HTTPException(401, "Not authenticated.")
    body       = await request.json()
    order_id   = body.get("razorpay_order_id", "")
    payment_id = body.get("razorpay_payment_id", "")
    signature  = body.get("razorpay_signature", "")
    try:
        client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
        client.utility.verify_payment_signature({
            "razorpay_order_id": order_id,
            "razorpay_payment_id": payment_id,
            "razorpay_signature": signature,
        })
    except Exception:
        raise HTTPException(400, "Payment verification failed.")
    user.is_subscribed       = True
    user.razorpay_payment_id = payment_id
    user.upload_count        = 0
    user.subscription_end    = datetime.utcnow() + timedelta(days=30)
    db.commit()
    return JSONResponse({
        "status": "ok",
        "message": "Payment verified! 10 uploads activated for 30 days.",
        "uploads_left": PAID_UPLOADS,
        "subscription_end": str(user.subscription_end),
    })


# ════════════════════════════════════════
# UPLOAD
# ════════════════════════════════════════

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...),
                      db: Session = Depends(get_db)):
    try:
        token = request.headers.get("X-Auth-Token", "")
        user  = get_user_by_token(token, db)
        info  = upload_status(user)
        if not info["allowed"]:
            reason = info["reason"]
            if reason == "login_required":
                raise HTTPException(401, "Please login to upload files.")
            elif reason == "trial_expired":
                raise HTTPException(402, "Free trial used (2/2). Subscribe to continue.")
            elif reason == "monthly_limit_reached":
                raise HTTPException(402, "Monthly limit reached (10/10). Renew subscription.")
            else:
                raise HTTPException(402, "Subscription required.")

        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in ("csv", "xlsx", "xls"):
            raise HTTPException(400, "Only CSV and Excel (.xlsx/.xls) files are supported.")

        sid       = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{sid}.{ext}"
        file_path.write_bytes(await file.read())

        df = load_file(str(file_path), ext)
        if df is None or df.empty:
            raise HTTPException(400, "Could not parse file — check format.")

        df.columns = [str(c).strip() for c in df.columns]

        _evict_old_sessions()
        store[sid] = {"df": df, "filename": file.filename, "ext": ext,
                      "file_path": str(file_path)}

        if user:
            user.upload_count += 1; db.commit()

        new_info = upload_status(user)
        return JSONResponse({
            "status":       "ok",
            "session_id":   sid,
            "filename":     file.filename,
            "uploads_left": new_info["uploads_left"],
            "shape":        {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
            "columns":      df.columns.tolist(),
            "dtypes":       {c: str(df[c].dtype) for c in df.columns},
            "missing":      {c: int(df[c].isnull().sum()) for c in df.columns},
            "sample":       df.head(5).fillna("N/A").astype(str).to_dict(orient="records"),
            "info":         get_dataset_info(df),
        })
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Upload error: {e}")


# ════════════════════════════════════════
# TRAIN
# ════════════════════════════════════════

@app.post("/train")
async def train(session_id: str = Form(...), task_type: str = Form(...),
                target_col: str = Form(...)):
    try:
        if session_id not in store:
            raise HTTPException(404, "Session not found — please re-upload.")
        if target_col not in store[session_id]["df"].columns:
            raise HTTPException(400, f"Column '{target_col}' not in dataset.")

        df = store[session_id]["df"].copy()
        df_clean, cleaning_report = clean_data(df, target_col)
        cleaned_path = CLEANED_DIR / f"{session_id}_cleaned.csv"
        df_clean.to_csv(cleaned_path, index=False)
        store[session_id].update({"df_clean": df_clean, "cleaned_path": str(cleaned_path)})

        eda_charts = generate_eda_visualizations(df_clean, target_col, task_type)
        result = (train_classification if task_type == "classification"
                  else train_regression)(df_clean, target_col)

        store[session_id].update({
            "model":          result["model"],
            "feature_cols":   result["feature_cols"],
            "feature_dtypes": result["feature_dtypes"],
            "label_encoder":  result.get("label_encoder"),
            "task_type":      task_type,
            "target_col":     target_col,
            "chat_context":   build_dataset_context(df_clean, target_col, task_type, result),
        })

        metrics = result["metrics"]

        return JSONResponse({
            "status":             "ok",
            "cleaning_report":    cleaning_report,
            "eda_charts":         eda_charts,
            "metrics":            metrics,
            "feature_importance": result.get("feature_importance", {}),
            "feature_cols":       result["feature_cols"],
            "feature_dtypes":     result["feature_dtypes"],
            "best_model":         result["best_model_name"],
            "task_type":          task_type,
            "target_col":         target_col,
        })
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Training error: {e}")


# ════════════════════════════════════════
# DOWNLOAD CLEANED
# ════════════════════════════════════════

@app.get("/download-cleaned/{session_id}")
async def download_cleaned(session_id: str):
    s = store.get(session_id, {})
    p = s.get("cleaned_path", "")
    if not p or not Path(p).exists():
        raise HTTPException(404, "Cleaned file not ready — train first.")
    return FileResponse(p, media_type="text/csv", filename="insightiq_cleaned.csv")


# ════════════════════════════════════════
# KPI DASHBOARD
# ════════════════════════════════════════

@app.get("/kpi/{session_id}")
async def get_kpi(session_id: str, request: Request):
    try:
        s = store.get(session_id)
        if not s:
            raise HTTPException(404, "Session not found.")
        df = s.get("df_clean")
        if df is None:
            df = s.get("df")
        if df is None:
            raise HTTPException(400, "No data in session.")
        kpis = compute_kpis(df)
        return JSONResponse({"status": "ok", "kpis": kpis})
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"KPI error: {e}")


# ════════════════════════════════════════
# LLM INSIGHTS — Claude API
# ════════════════════════════════════════

@app.post("/llm-insight")
async def llm_insight(request: Request):
    try:
        import anthropic
        body        = await request.json()
        chart_title = body.get("chart_title", "")
        chart_data  = body.get("chart_data", "")
        session_id  = body.get("session_id", "")

        s  = store.get(session_id, {})
        df = s.get("df_clean")
        if df is None:
            df = s.get("df")

        ctx = ""
        if df is not None:
            ctx = f"Dataset: {df.shape[0]} rows x {df.shape[1]} cols. Columns: {df.columns.tolist()}. "
            try: ctx += f"Stats: {df.describe().round(2).to_string()}"
            except: pass

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return JSONResponse({"insight": chart_data})

        import httpx
        client = anthropic.Anthropic(
            api_key=api_key,
            http_client=httpx.Client(timeout=10.0)
        )

        prompt = f"""You are a business data analyst. Chart: "{chart_title}".
Dataset: {ctx}
Rule-based insight: {chart_data}

Write 2-3 lines in simple English:
1. What this means for the business
2. One specific actionable recommendation
No jargon. Be direct."""

        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        return JSONResponse({"insight": msg.content[0].text})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"insight": body.get("chart_data", "")})


# ════════════════════════════════════════
# PREDICT FUTURE TRENDS
# ════════════════════════════════════════

@app.post("/predict-trends")
async def predict_trends(request: Request):
    try:
        body       = await request.json()
        session_id = body.get("session_id", "")
        n_future   = min(int(body.get("n_future", 10)), 50)

        if session_id not in store:
            raise HTTPException(404, "Session not found — please re-upload.")

        s  = store[session_id]
        df = s.get("df_clean")
        if df is None:
            df = s.get("df")
        if df is None:
            raise HTTPException(400, "No data in session.")

        from visualizer import _detect_date_col, _smart_num
        import base64
        from io import BytesIO

        num_cols = _smart_num(df)
        if not num_cols:
            raise HTTPException(400, "No numeric columns found for trend prediction.")

        date_col, date_series = _detect_date_col(df)
        target_cols = num_cols[:3]
        results = []

        BG='#07090F'; BG2='#0D1120'; TEXT='#E8EDF7'; MUTED='#6B7A99'
        PALETTE=['#00E5C8','#00BFFF','#FF8C42']

        n_cols = len(target_cols)
        fig, axes = plt.subplots(n_cols, 1, figsize=(13, 4.5*n_cols), facecolor=BG)
        if n_cols == 1:
            axes = [axes]

        for i, col in enumerate(target_cols):
            ax = axes[i]
            ax.set_facecolor(BG2)
            for sp in ax.spines.values():
                sp.set_edgecolor('#1E2D45')
            ax.tick_params(colors=MUTED)
            ax.xaxis.label.set_color(MUTED)
            ax.yaxis.label.set_color(MUTED)

            color  = PALETTE[i % len(PALETTE)]
            series = df[col].dropna().reset_index(drop=True)
            n      = len(series)

            if n < 5:
                ax.text(0.5, 0.5, f'{col}: Not enough data (need 5+ rows)',
                        ha='center', va='center', color=MUTED,
                        transform=ax.transAxes, fontsize=10)
                continue

            y = series.values.astype(float)

            window    = min(5, n // 3)
            roll_mean = pd.Series(y).rolling(window, min_periods=1).mean().values
            roll_std  = pd.Series(y).rolling(window, min_periods=1).std().fillna(0).values

            X = np.column_stack([np.arange(n), roll_mean, roll_std])
            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            lr = LinearRegression()
            lr.fit(X_scaled, y)
            lr_score = lr.score(X_scaled, y)

            future_idx      = np.arange(n, n + n_future)
            X_future        = np.column_stack([future_idx, np.full(n_future, roll_mean[-1]), np.full(n_future, roll_std[-1])])
            X_future_scaled = scaler.transform(X_future)
            y_future        = lr.predict(X_future_scaled)

            residuals = y - lr.predict(X_scaled)
            std_err   = np.std(residuals)
            ci_upper  = y_future + 1.0 * std_err
            ci_lower  = y_future - 1.0 * std_err

            if date_series is not None:
                try:
                    x_actual = pd.to_datetime(date_series.reset_index(drop=True)).values[:n]
                    freq = pd.infer_freq(pd.to_datetime(date_series.dropna()))
                    if freq:
                        last_date       = pd.to_datetime(date_series.dropna()).iloc[-1]
                        x_future_dates  = pd.date_range(last_date, periods=n_future+1, freq=freq)[1:]
                        ax.plot(x_actual, y, color=color, linewidth=1.5, alpha=0.9, label='Actual')
                        ax.plot(x_future_dates, y_future, color='white', linewidth=2, linestyle='--', alpha=0.9, label='Forecast')
                        ax.fill_between(x_future_dates, ci_lower, ci_upper, color='white', alpha=0.08, label='±1 Std Error')
                        ax.axvline(x=x_actual[-1], color=MUTED, linewidth=1, linestyle=':')
                        fig.autofmt_xdate()
                    else:
                        raise ValueError("Cannot infer frequency")
                except:
                    ax.plot(range(n), y, color=color, linewidth=1.5, alpha=0.9, label='Actual')
                    ax.plot(range(n-1, n+n_future), [y[-1]]+list(y_future), color='white', linewidth=2, linestyle='--', label='Forecast')
                    ax.fill_between(range(n, n+n_future), ci_lower, ci_upper, color='white', alpha=0.08, label='±1 Std Error')
                    ax.axvline(x=n-1, color=MUTED, linewidth=1, linestyle=':')
            else:
                ax.plot(range(n), y, color=color, linewidth=1.5, alpha=0.9, label='Actual')
                ax.plot(range(n-1, n+n_future), [y[-1]]+list(y_future), color='white', linewidth=2, linestyle='--', label='Forecast')
                ax.fill_between(range(n, n+n_future), ci_lower, ci_upper, color='white', alpha=0.08, label='±1 Std Error')
                ax.axvline(x=n-1, color=MUTED, linewidth=1, linestyle=':')
                ax.text(n-1, ax.get_ylim()[0], '  Now', fontsize=8, color=MUTED)

            pct_chg = ((y_future[-1] - y[-1]) / (abs(y[-1]) + 1e-9)) * 100
            trend   = "Upward ↑" if y_future[-1] > y_future[0] else "Downward ↓"
            chg_col = '#3DDB8A' if pct_chg >= 0 else '#FF4D6D'
            r2_disp = f"R²={lr_score:.2f}" if lr_score >= 0 else f"R²={lr_score:.2f} (weak fit)"

            ax.set_title(f'{col}   |   {trend}   |   Forecast change: {pct_chg:+.1f}%   |   {r2_disp}',
                         fontsize=10, color=chg_col if abs(pct_chg) > 2 else TEXT, pad=8)
            ax.set_ylabel(col, fontsize=9, color=color)
            ax.legend(fontsize=8, loc='upper left', facecolor=BG2, edgecolor='#1E2D45', labelcolor=TEXT)
            ax.grid(True, alpha=0.2, color='#1E2D45')

            results.append({
                "column":         col,
                "model_used":     "LinearRegression (trend-based)",
                "r2_fit":         round(float(lr_score), 4),
                "last_actual":    round(float(y[-1]), 4),
                "predicted_next": [round(float(v), 4) for v in y_future],
                "trend":          "up" if y_future[-1] > y_future[0] else "down",
                "pct_change":     round(float(pct_chg), 2),
                "std_error":      round(float(std_err), 4),
            })

        fig.suptitle(f'Trend Forecast — Next {n_future} Steps',
                     fontsize=14, color=TEXT, y=1.01, fontweight='bold')
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)

        insights = []
        for r in results:
            fit_warn  = "" if r["r2_fit"] >= 0.3 else " ⚠️ Low R² — trend may not be reliable."
            direction = "increase" if r["trend"] == "up" else "decrease"
            insights.append(
                f"<b>{r['column']}</b>: Expected to <b>{direction}</b> by "
                f"<b>{r['pct_change']:+.1f}%</b> over next {n_future} steps. "
                f"Forecast accuracy (R²): {r['r2_fit']}.{fit_warn}"
            )

        return JSONResponse({
            "status":   "ok",
            "image":    img_b64,
            "results":  results,
            "insights": insights,
            "n_future": n_future,
        })

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Trend prediction error: {e}")


# ════════════════════════════════════════
# PREDICT
# ════════════════════════════════════════

@app.post("/predict")
async def predict(request: Request):
    try:
        body     = await request.json()
        sid      = body.get("session_id", "")
        features = body.get("features", {})

        if sid not in store or "model" not in store[sid]:
            raise HTTPException(404, "No trained model — train first.")

        s   = store[sid]
        row = {}
        for col in s["feature_cols"]:
            val   = features.get(col, "")
            dtype = s["feature_dtypes"].get(col, "object")
            if "int" in dtype or "float" in dtype:
                try:    row[col] = float(val)
                except: row[col] = np.nan
            else:
                row[col] = str(val) if val != "" else np.nan

        inp  = pd.DataFrame([row])[s["feature_cols"]]
        pred = s["model"].predict(inp)[0]

        le = s.get("label_encoder")
        if s["task_type"] == "classification" and le is not None:
            try:    label = str(le.inverse_transform([int(pred)])[0])
            except: label = str(pred)
        else:
            label = round(float(pred), 4)

        proba = None
        if s["task_type"] == "classification" and hasattr(s["model"], "predict_proba"):
            try:
                pa  = s["model"].predict_proba(inp)[0]
                cls = le.classes_.tolist() if le else list(range(len(pa)))
                proba = {str(c): round(float(p)*100, 2) for c, p in zip(cls, pa)}
            except: pass

        return JSONResponse({"status": "ok", "prediction": str(label), "probabilities": proba})
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Prediction error: {e}")


# ════════════════════════════════════════
# CHAT
# ════════════════════════════════════════

@app.post("/chat")
async def chat(request: Request):
    try:
        body     = await request.json()
        sid      = body.get("session_id", "")
        question = body.get("question", "").strip()
        if not question:
            return JSONResponse({"answer": "Please ask something!"})
        ctx = store.get(sid, {}).get("chat_context", "No dataset loaded yet.")
        return JSONResponse({"status": "ok", "answer": get_chat_response(question, ctx)})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"answer": f"Error: {e}"})


# ════════════════════════════════════════
# SAVE / LIST / COMPARE
# ════════════════════════════════════════

@app.post("/save-dataset")
async def save_dataset(request: Request, db: Session = Depends(get_db)):
    try:
        body       = await request.json()
        session_id = body.get("session_id", "").strip()
        label      = body.get("label", "").strip()

        if not session_id or session_id in ("null", "undefined"):
            raise HTTPException(400, "No active session — please upload a file first.")
        if not label:
            raise HTTPException(400, "Label is required.")

        token   = request.headers.get("X-Auth-Token", "")
        user    = get_user_by_token(token, db)
        user_id = user.id if user else None

        if session_id not in store:
            raise HTTPException(404, "Session not found — please re-upload.")

        df = store[session_id].get("df_clean")
        if df is None:
            df = store[session_id].get("df")
        if df is None:
            raise HTTPException(400, "No dataframe in session.")

        lbl = label.replace(" ", "_")
        existing = db.query(SavedDataset).filter(
            SavedDataset.label == lbl, SavedDataset.user_id == user_id
        ).first()

        if existing:
            existing.data_json = df.to_json()
            existing.filename  = store[session_id].get("filename", "unknown")
        else:
            db.add(SavedDataset(
                user_id=user_id, label=lbl,
                data_json=df.to_json(),
                filename=store[session_id].get("filename", "unknown"),
            ))
        db.commit()
        return JSONResponse({"status": "ok", "saved": lbl, "rows": int(len(df))})
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Save error: {e}")


@app.get("/saved-datasets")
async def list_saved(request: Request, db: Session = Depends(get_db)):
    token    = request.headers.get("X-Auth-Token", "")
    user     = get_user_by_token(token, db)
    user_id  = user.id if user else None
    datasets = db.query(SavedDataset).filter(SavedDataset.user_id == user_id).all()
    return JSONResponse({"status": "ok", "datasets": [d.label for d in datasets]})


@app.post("/compare")
async def compare(request: Request, db: Session = Depends(get_db)):
    try:
        body          = await request.json()
        session_id    = body.get("session_id", "").strip()
        compare_label = body.get("compare_label", "").strip()

        token   = request.headers.get("X-Auth-Token", "")
        user    = get_user_by_token(token, db)
        user_id = user.id if user else None

        if not session_id or session_id == "undefined":
            raise HTTPException(400, "No active session — upload a file first.")
        if session_id not in store:
            raise HTTPException(404, "Session not found.")

        df_new = store[session_id].get("df_clean")
        if df_new is None:
            df_new = store[session_id].get("df")
        if df_new is None:
            raise HTTPException(400, "No dataframe in session.")

        saved = db.query(SavedDataset).filter(
            SavedDataset.label == compare_label,
            SavedDataset.user_id == user_id
        ).first()
        if not saved:
            raise HTTPException(404, f"'{compare_label}' not found.")

        df_old = pd.read_json(saved.data_json)
        charts, stats = generate_comparison_viz(df_old, df_new, compare_label, "Current")
        return JSONResponse({"status": "ok", "charts": charts, "stats": stats})
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


# ════════════════════════════════════════
# HEALTH
# ════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "platform": "InsightIQ",
        "version": "3.1",
        "active_sessions": len(store)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))