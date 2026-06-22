#!/usr/bin/env python3
"""
Swing Scanner Web App — Backend (Flask)
========================================
• Runs the high-probability swing scanner on demand or via daily schedule
• Saves results as JSON + Excel in ./results/
• Serves a REST API consumed by index.html
• Auto-scheduler runs at 4:15 PM IST every weekday (after NSE close)

Setup:
    pip install flask flask-cors yfinance pandas openpyxl apscheduler numpy

Run:
    python swing_scanner_app.py --excel NSEV2.xlsx

Then open:  http://localhost:5000
"""

import argparse
import json
import os
import threading
import time
import math
from datetime import datetime
from itertools import islice
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler

# ─────────────────────── CONFIG ───────────────────────
RESULTS_DIR      = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CHUNK_SIZE       = 40
MIN_VALID_DAYS   = 150
PERIOD           = "250d"
INTERVAL         = "1d"
RETRY_SLEEP      = 2
MAX_RETRIES      = 3

MIN_ATR_PCT        = 0.015
MIN_RR             = 3.0
RSI_MIN            = 45
RSI_MAX            = 68
PULLBACK_TOLERANCE = 0.03
VOL_CONSOLIDATION  = 0.85
VOL_EXPAND         = 1.2
HH_LOOKBACK        = 20
MIN_SCORE_DEFAULT  = 8
NIFTY_SYMBOL       = "^NSEI"
IST                = ZoneInfo("Asia/Kolkata")
# ──────────────────────────────────────────────────────

app = Flask(__name__, static_folder=".")
CORS(app)

scan_status = {
    "running": False,
    "last_run": None,
    "last_run_ts": None,
    "progress": 0,
    "total": 0,
    "error": None,
}

EXCEL_FILE = "NSEV2.xlsx"


# ══════════════════════ SCANNER LOGIC ══════════════════════

def chunked(iterable, size):
    it = iter(iterable)
    for first in it:
        yield [first] + list(islice(it, size - 1))


def download_with_retries(tickers):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return yf.download(
                tickers, period=PERIOD, interval=INTERVAL,
                auto_adjust=True, threads=True, progress=False
            )
        except Exception as e:
            time.sleep(RETRY_SLEEP * attempt)
    return None


def rsi(close, window=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_atr(high, low, close, window=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def is_making_higher_highs(close, lookback=HH_LOOKBACK):
    segment = close.tail(lookback)
    if len(segment) < lookback:
        return False
    w  = lookback // 3
    w1 = segment.iloc[:w].max()
    w2 = segment.iloc[w:2*w].max()
    w3 = segment.iloc[2*w:].max()
    return bool(w3 > w2 > w1)


def get_nifty_trend():
    try:
        data  = yf.download(NIFTY_SYMBOL, period="300d", interval="1d",
                            auto_adjust=True, progress=False)
        close = data["Close"].squeeze().dropna()
        price = close.iloc[-1]
        ema50 = close.ewm(span=50).mean().iloc[-1]
        ema200= close.ewm(span=200).mean().iloc[-1]
        return bool(price > ema50 > ema200)
    except Exception:
        return True


def score_stock(close, high, low, vol, price, atr, atr_pct):
    checks = {}
    ema20  = close.ewm(span=20).mean()
    ema50  = close.ewm(span=50).mean()
    ema200 = close.ewm(span=200).mean()
    c20, c50, c200 = ema20.iloc[-1], ema50.iloc[-1], ema200.iloc[-1]

    checks["Bull Trend"]       = bool(price > c20 > c50 > c200)
    checks["Higher Highs"]     = is_making_higher_highs(close)

    near_ema20 = abs(price - c20) / c20 <= PULLBACK_TOLERANCE
    near_ema50 = abs(price - c50) / c50 <= PULLBACK_TOLERANCE
    checks["Pullback Entry"]   = bool(near_ema20 or near_ema50)

    avg_vol_20 = vol.tail(20).mean()
    avg_vol_3  = vol.iloc[-4:-1].mean()
    today_vol  = vol.iloc[-1]
    checks["Volume Pattern"]   = bool(
        (avg_vol_3 / avg_vol_20 < VOL_CONSOLIDATION if avg_vol_20 > 0 else False) and
        (today_vol / avg_vol_20 > VOL_EXPAND        if avg_vol_20 > 0 else False)
    )

    rsi_val = rsi(close).iloc[-1]
    checks["RSI Sweet Spot"]   = bool(RSI_MIN <= rsi_val <= RSI_MAX)
    checks["ATR Ok"]           = bool(atr_pct >= MIN_ATR_PCT)

    high_52w = close.tail(252).max()
    low_52w  = close.tail(252).min()
    rng      = high_52w - low_52w
    pct_rng  = (price - low_52w) / rng if rng > 0 else 0
    checks["Not At Resistance"]= bool(pct_rng < 0.92)

    stop   = price - 1.5 * atr
    risk   = price - stop
    target = price + MIN_RR * risk
    checks["R:R ≥ 1:3"]        = bool(risk > 0 and (target / price - 1) > 0.05)
    checks["EMA200 Buffer"]    = bool(price > c200 * 1.02)
    checks["Green Candle"]     = bool(close.iloc[-1] > close.iloc[-2])

    score = sum(checks.values())
    return score, checks, {
        "ema20": round(c20, 2), "ema50": round(c50, 2), "ema200": round(c200, 2),
        "rsi": round(float(rsi_val), 2), "pct_rng": round(pct_rng * 100, 1),
        "stop": round(stop, 2), "target": round(target, 2),
        "avg_vol": int(avg_vol_20),
        "rel_vol": round(float(today_vol / avg_vol_20), 2) if avg_vol_20 > 0 else 0,
    }


def grade(score):
    if score == 10: return "A+"
    if score == 9:  return "A"
    if score == 8:  return "B+"
    if score == 7:  return "B"
    return "C"


def run_scan(excel_file=None, min_score=MIN_SCORE_DEFAULT):
    global scan_status
    scan_status["running"]  = True
    scan_status["error"]    = None
    scan_status["progress"] = 0

    try:
        ef = excel_file or EXCEL_FILE
        df = pd.read_excel(ef)
        if "Symbol" not in df.columns:
            raise ValueError("Excel must have a 'Symbol' column")

        tickers    = df["Symbol"].dropna().astype(str).str.strip().str.replace(".NS", "").tolist()
        yf_tickers = [t + ".NS" for t in tickers]
        scan_status["total"] = len(yf_tickers)

        market_ok = get_nifty_trend()
        results   = []
        processed = 0

        for chunk in chunked(yf_tickers, CHUNK_SIZE):
            data = download_with_retries(chunk)
            if data is None or not isinstance(data.columns, pd.MultiIndex):
                processed += len(chunk)
                scan_status["progress"] = processed
                continue

            close_df = data["Close"].dropna(axis=1, thresh=MIN_VALID_DAYS)
            high_df  = data["High"]
            low_df   = data["Low"]
            vol_df   = data["Volume"]

            for ticker in close_df.columns:
                try:
                    close = close_df[ticker].dropna()
                    high  = high_df[ticker].reindex(close.index).dropna()
                    low   = low_df[ticker].reindex(close.index).dropna()
                    vol   = vol_df[ticker].reindex(close.index).dropna()

                    if len(close) < MIN_VALID_DAYS:
                        continue

                    price   = float(close.iloc[-1])
                    atr_s   = calculate_atr(high, low, close)
                    atr     = float(atr_s.iloc[-1])
                    atr_pct = atr / price if price > 0 else 0

                    score, checks, extras = score_stock(close, high, low, vol, price, atr, atr_pct)
                    eff_score = score - (0 if market_ok else 1)

                    if eff_score >= min_score:
                        failed = [k for k, v in checks.items() if not v]
                        results.append({
                            "stock":      ticker.replace(".NS", ""),
                            "price":      round(price, 2),
                            "score":      score,
                            "grade":      grade(score),
                            "atr_pct":    round(atr_pct * 100, 2),
                            "rsi":        extras["rsi"],
                            "pct_rng":    extras["pct_rng"],
                            "ema20":      extras["ema20"],
                            "ema50":      extras["ema50"],
                            "ema200":     extras["ema200"],
                            "stop":       extras["stop"],
                            "target":     extras["target"],
                            "rr":         f"1:{MIN_RR:.0f}",
                            "avg_vol":    extras["avg_vol"],
                            "rel_vol":    extras["rel_vol"],
                            "market_ok":  market_ok,
                            "failed":     failed,
                            "checks":     checks,
                        })
                except Exception:
                    continue

            processed += len(chunk)
            scan_status["progress"] = processed
            time.sleep(0.3)

        # Sort: score desc, rel_vol desc
        results.sort(key=lambda x: (-x["score"], -x["rel_vol"]))

        ts       = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
        date_str = datetime.now(IST).strftime("%Y-%m-%d")

        # Save JSON
        json_path = RESULTS_DIR / f"scan_{ts}.json"
        meta = {
            "timestamp":  datetime.now(IST).isoformat(),
            "date":       date_str,
            "total_scanned": len(yf_tickers),
            "candidates": len(results),
            "market_ok":  market_ok,
            "min_score":  min_score,
            "results":    results,
        }
        json_path.write_text(json.dumps(meta, indent=2))

        # Save Excel
        if results:
            xl_path = RESULTS_DIR / f"scan_{ts}.xlsx"
            rows = []
            for r in results:
                rows.append({
                    "Stock": r["stock"], "Price": r["price"],
                    "Score": r["score"], "Grade": r["grade"],
                    "RSI": r["rsi"], "ATR %": r["atr_pct"],
                    "52w Range %": r["pct_rng"],
                    "EMA20": r["ema20"], "EMA50": r["ema50"], "EMA200": r["ema200"],
                    "Stop Loss": r["stop"], "Target": r["target"], "R:R": r["rr"],
                    "Avg Vol": r["avg_vol"], "Rel Vol": r["rel_vol"],
                    "Failed Checks": ", ".join(r["failed"]) if r["failed"] else "None ✅",
                })
            pd.DataFrame(rows).to_excel(xl_path, index=False)

        scan_status["last_run"]    = date_str
        scan_status["last_run_ts"] = datetime.now(IST).isoformat()
        scan_status["running"]     = False
        scan_status["progress"]    = len(yf_tickers)

        print(f"✅ Scan complete — {len(results)} candidates found")
        return meta

    except Exception as e:
        scan_status["running"] = False
        scan_status["error"]   = str(e)
        print(f"❌ Scan error: {e}")
        return None


# ══════════════════════ API ROUTES ══════════════════════

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/status")
def api_status():
    return jsonify(scan_status)


@app.route("/api/run", methods=["POST"])
def api_run():
    if scan_status["running"]:
        return jsonify({"error": "Scan already running"}), 409
    min_score = int(request.json.get("min_score", MIN_SCORE_DEFAULT)) if request.json else MIN_SCORE_DEFAULT
    thread    = threading.Thread(target=run_scan, kwargs={"min_score": min_score})
    thread.daemon = True
    thread.start()
    return jsonify({"message": "Scan started", "min_score": min_score})


@app.route("/api/results")
def api_results():
    """Return the latest scan results."""
    files = sorted(RESULTS_DIR.glob("scan_*.json"), reverse=True)
    if not files:
        return jsonify({"results": [], "message": "No scans yet"})
    data = json.loads(files[0].read_text())
    return jsonify(data)


@app.route("/api/history")
def api_history():
    """Return list of all past scan summaries."""
    files  = sorted(RESULTS_DIR.glob("scan_*.json"), reverse=True)[:30]
    history = []
    for f in files:
        try:
            d = json.loads(f.read_text())
            history.append({
                "date":       d.get("date"),
                "timestamp":  d.get("timestamp"),
                "candidates": d.get("candidates", 0),
                "total":      d.get("total_scanned", 0),
                "market_ok":  d.get("market_ok", True),
                "file":       f.name,
            })
        except Exception:
            continue
    return jsonify(history)


@app.route("/api/history/<filename>")
def api_history_detail(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    return jsonify(json.loads(path.read_text()))


@app.route("/api/download/<filename>")
def api_download(filename):
    return send_from_directory(RESULTS_DIR, filename, as_attachment=True)


# ══════════════════════ SCHEDULER ══════════════════════

def scheduled_scan():
    now = datetime.now(IST)
    if now.weekday() < 5:   # Mon–Fri only
        print(f"⏰ Scheduled scan starting at {now.strftime('%Y-%m-%d %H:%M IST')}")
        run_scan()
    else:
        print("⏰ Weekend — skipping scheduled scan")


def start_scheduler():
    scheduler = BackgroundScheduler(timezone=IST)
    scheduler.add_job(scheduled_scan, "cron",
                      day_of_week="mon-fri", hour=16, minute=15)
    scheduler.start()
    print("⏰ Scheduler active — auto-scan runs Mon–Fri at 4:15 PM IST")
    return scheduler


# ══════════════════════ MAIN ══════════════════════

def main():
    global EXCEL_FILE
    parser = argparse.ArgumentParser(description="Swing Scanner Web App")
    parser.add_argument("--excel",   default="NSEV2.xlsx", help="Path to NSE symbols Excel")
    parser.add_argument("--port",    type=int, default=5000)
    parser.add_argument("--no-sched", action="store_true", help="Disable auto-scheduler")
    args = parser.parse_args()

    EXCEL_FILE = args.excel
    if not Path(EXCEL_FILE).exists():
        print(f"⚠️  Warning: {EXCEL_FILE} not found. Upload it or use --excel flag.")

    if not args.no_sched:
        start_scheduler()

    print(f"\n🚀 Swing Scanner App running at http://localhost:{args.port}")
    print(f"   Excel file : {EXCEL_FILE}")
    print(f"   Results dir: {RESULTS_DIR.absolute()}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
