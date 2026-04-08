#!/usr/bin/env python3
"""
Stock Watchlist Report Generator
=================================
Screens ~46 high-quality stocks across 7 sectors and selects the top performers
using a composite score: momentum (25pts) + fundamentals (40pts) +
technical setup (25pts) + valuation (10pts) = 100pts total.

Usage:
    python stock_watchlist_report.py [--top-n 7] [--output-dir .]

Outputs:
    watchlist_report_YYYY-MM-DD.html   — interactive report with TradingView charts
    tradingview_watchlist.txt          — importable into TradingView Watchlist panel
    screener.pine                      — Pine Script v5 for TradingView alerts

Requirements:
    pip install yfinance pandas numpy jinja2 requests
"""

import argparse
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from jinja2 import Environment

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
TOP_N = 7
BENCHMARK = "SPY"

UNIVERSE: Dict[str, List[str]] = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL",
        "CRM", "ADBE", "NOW", "ANET", "KLAC",
    ],
    "Healthcare": [
        "LLY", "UNH", "ABBV", "TMO", "ABT",
        "ISRG", "VRTX", "BSX", "EW",
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "MCD",
        "NKE", "LULU", "SBUX", "LOW",
    ],
    "Financials": [
        "BRK-B", "JPM", "V", "MA",
        "GS", "AXP", "SPGI", "ICE",
    ],
    "Energy": [
        "XOM", "CVX", "SLB", "PSX", "VLO", "OXY",
    ],
    "Industrials": [
        "CAT", "DE", "RTX", "HON",
        "ETN", "GE", "CTAS", "PCAR",
    ],
    "Communication Services": [
        "GOOGL", "META", "NFLX", "DIS",
        "TMUS", "VZ", "T",
    ],
}

# Yahoo Finance exchange code → TradingView exchange prefix
EXCHANGE_MAP: Dict[str, str] = {
    "NMS":      "NASDAQ",   # NASDAQ National Market System
    "NGM":      "NASDAQ",   # NASDAQ Global Market
    "NCM":      "NASDAQ",   # NASDAQ Capital Market
    "NYQ":      "NYSE",
    "NYSEArca": "NYSE",
    "PCX":      "NYSE",     # NYSE Arca
    "ASE":      "AMEX",
    "BATS":     "BATS",
}


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class RawStock:
    ticker: str
    sector: str
    info: Dict[str, Any]
    hist: Optional[pd.DataFrame]   # 1y OHLCV, auto-adjusted


@dataclass
class ScoredStock:
    ticker: str
    sector: str
    name: str
    price: float
    exchange_raw: str

    # Sub-scores (filled after batch normalization)
    momentum_score: float = 0.0       # 0–25
    fundamental_score: float = 0.0    # 0–40
    technical_score: float = 0.0      # 0–25
    valuation_score: float = 0.0      # 0–10

    # Raw metrics for display (None = not available)
    rev_growth: Optional[float] = None
    eps_growth: Optional[float] = None
    profit_margin: Optional[float] = None
    roe: Optional[float] = None
    pe_trailing: Optional[float] = None
    pe_forward: Optional[float] = None
    peg: Optional[float] = None
    rsi: Optional[float] = None
    ma50: Optional[float] = None
    ma200: Optional[float] = None
    ret_1mo: Optional[float] = None
    ret_3mo: Optional[float] = None
    ret_6mo: Optional[float] = None

    # Raw sub-values before normalization (used internally)
    _raw: Dict[str, Optional[float]] = field(default_factory=dict)

    @property
    def total_score(self) -> float:
        return self.momentum_score + self.fundamental_score + self.technical_score + self.valuation_score

    @property
    def tv_symbol(self) -> str:
        """Return EXCHANGE:TICKER in TradingView format."""
        exch = EXCHANGE_MAP.get(self.exchange_raw, self.exchange_raw.upper() if self.exchange_raw else "NASDAQ")
        sym = self.ticker.replace("-", ".")
        return f"{exch}:{sym}"

    @property
    def tv_chart_url(self) -> str:
        return f"https://www.tradingview.com/chart/?symbol={self.tv_symbol}"

    def fmt(self, val: Optional[float], pct: bool = False, decimals: int = 1) -> str:
        """Format a value for display, handling None gracefully."""
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "N/A"
        if pct:
            return f"{val * 100:+.{decimals}f}%"
        return f"{val:.{decimals}f}"


# ─── Utility Functions ────────────────────────────────────────────────────────

def safe_float(val: Any) -> Optional[float]:
    """Convert to float, returning None on failure or NaN."""
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def compute_rsi(closes: pd.Series, period: int = 14) -> Optional[float]:
    """Wilder's RSI using EWM smoothing."""
    closes = closes.dropna()
    if len(closes) < period + 1:
        return None
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def period_return(closes: pd.Series, lookback_days: int) -> Optional[float]:
    """Return (close[-1] / close[-lookback] - 1). None if not enough data."""
    closes = closes.dropna()
    if len(closes) < lookback_days + 1:
        return None
    end = float(closes.iloc[-1])
    start = float(closes.iloc[-lookback_days])
    if start == 0:
        return None
    return (end / start) - 1.0


def normalize_component(
    values: List[Optional[float]], max_pts: float
) -> List[float]:
    """Min-max normalize a list to [0, max_pts]. None/NaN → 0."""
    clean = [v for v in values if v is not None]
    if not clean:
        return [0.0] * len(values)
    lo, hi = min(clean), max(clean)
    if math.isclose(hi, lo):
        return [max_pts * 0.5 if v is not None else 0.0 for v in values]
    return [
        0.0 if v is None else max_pts * (v - lo) / (hi - lo)
        for v in values
    ]


def eps_growth_from_financials(tk: yf.Ticker) -> Optional[float]:
    """Fallback: compute EPS growth from annual income statement."""
    try:
        fin = tk.financials  # rows=line items, cols=dates (most recent first)
        if fin is None or fin.empty:
            return None
        net_row = None
        for label in ["Net Income", "NetIncome", "Net Income From Continuing Operations",
                       "Net Income Common Stockholders"]:
            if label in fin.index:
                net_row = fin.loc[label]
                break
        if net_row is None or len(net_row) < 2:
            return None
        curr = safe_float(net_row.iloc[0])
        prev = safe_float(net_row.iloc[1])
        if curr is None or prev is None or prev == 0:
            return None
        return (curr - prev) / abs(prev)
    except Exception:
        return None


# ─── Data Fetching ────────────────────────────────────────────────────────────

def _get_hist_for_ticker(price_data: pd.DataFrame, ticker: str, all_tickers: List[str]) -> Optional[pd.DataFrame]:
    """Extract single-ticker history from a bulk-download DataFrame."""
    if price_data is None or price_data.empty:
        return None
    try:
        if len(all_tickers) == 1:
            # Single-ticker download returns flat columns (Open, High, Low, Close, Volume)
            return price_data.copy()
        # Multi-ticker: MultiIndex columns (Price, Ticker)
        if isinstance(price_data.columns, pd.MultiIndex):
            if ticker in price_data.columns.get_level_values(1):
                df = price_data.xs(ticker, axis=1, level=1).copy()
                return df
        return None
    except Exception:
        return None


def fetch_all_data(universe: Dict[str, List[str]]) -> Dict[str, RawStock]:
    """Download price history (bulk) and fundamentals (per-ticker)."""
    tickers_flat = [t for sector_tickers in universe.values() for t in sector_tickers]
    sector_map = {t: s for s, ts in universe.items() for t in ts}

    # ── Step 1: Bulk price download (one HTTP call) ────────────────────────
    log.info("Downloading 1-year price history for %d tickers + SPY …", len(tickers_flat))
    all_syms = tickers_flat + [BENCHMARK]
    try:
        price_data = yf.download(
            tickers=all_syms,
            period="1y",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )
        # Strip timezone from index (pandas 2.0+ returns tz-aware)
        if hasattr(price_data.index, "tz") and price_data.index.tz is not None:
            price_data.index = price_data.index.tz_localize(None)
    except Exception as e:
        log.error("Bulk download failed: %s", e)
        price_data = pd.DataFrame()

    # Extract benchmark history separately
    spy_hist = _get_hist_for_ticker(price_data, BENCHMARK, all_syms)

    # ── Step 2: Per-ticker fundamentals ───────────────────────────────────
    results: Dict[str, RawStock] = {}
    log.info("Fetching fundamentals for %d tickers (this takes ~60-90 seconds) …", len(tickers_flat))

    for i, ticker in enumerate(tickers_flat, 1):
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}

            # Validate: skip if Yahoo returned an empty or stub response
            price = safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))
            if price is None:
                log.warning("  [%2d/%d] %-8s  skipped (no price data)", i, len(tickers_flat), ticker)
                continue

            hist = _get_hist_for_ticker(price_data, ticker, all_syms)
            if hist is None or hist.empty:
                log.warning("  [%2d/%d] %-8s  skipped (no price history)", i, len(tickers_flat), ticker)
                continue

            results[ticker] = RawStock(
                ticker=ticker,
                sector=sector_map.get(ticker, "Unknown"),
                info=info,
                hist=hist,
            )
            log.info("  [%2d/%d] %-8s  OK  ($%.2f)", i, len(tickers_flat), ticker, price)

        except Exception as e:
            log.warning("  [%2d/%d] %-8s  error: %s", i, len(tickers_flat), ticker, e)

        # Brief pause every 10 tickers to be polite to Yahoo's rate limiter
        if i % 10 == 0:
            time.sleep(1.0)

    log.info("Fetched data for %d/%d tickers.", len(results), len(tickers_flat))
    return results, spy_hist


# ─── Scoring ──────────────────────────────────────────────────────────────────

def extract_raw_scores(raw: RawStock, spy_hist: Optional[pd.DataFrame]) -> ScoredStock:
    """Extract raw (un-normalized) metric values from a RawStock."""
    info = raw.info
    hist = raw.hist
    closes = hist["Close"].dropna() if hist is not None else pd.Series([], dtype=float)

    # Current price
    price = safe_float(info.get("regularMarketPrice") or info.get("currentPrice")) or 0.0

    # ── Fundamentals ──────────────────────────────────────────────────────
    rev_growth = safe_float(info.get("revenueGrowth"))
    eps_growth = safe_float(info.get("earningsGrowth"))
    if eps_growth is None:
        eps_growth = eps_growth_from_financials(yf.Ticker(raw.ticker))
    profit_margin = safe_float(info.get("profitMargins"))
    roe = safe_float(info.get("returnOnEquity"))

    # ── Valuation ─────────────────────────────────────────────────────────
    pe_trailing = safe_float(info.get("trailingPE"))
    pe_forward = safe_float(info.get("forwardPE"))
    peg = safe_float(info.get("pegRatio"))
    # Exclude negative PEG (losing money — not a value signal)
    if peg is not None and peg <= 0:
        peg = None

    # PEG raw score: 1/peg (lower PEG → higher score)
    peg_raw = (1.0 / peg) if peg and peg > 0 else None

    # Forward vs trailing P/E: positive means analysts expect earnings to grow into valuation
    pe_direction_raw = None
    if pe_trailing and pe_forward and pe_forward > 0:
        pe_direction_raw = (pe_trailing / pe_forward) - 1.0

    # ── Technical ─────────────────────────────────────────────────────────
    ma50 = safe_float(info.get("fiftyDayAverage"))
    ma200 = safe_float(info.get("twoHundredDayAverage"))
    rsi = compute_rsi(closes)

    price_vs_ma50 = ((price / ma50) - 1.0) if ma50 and ma50 > 0 else None
    price_vs_ma200 = ((price / ma200) - 1.0) if ma200 and ma200 > 0 else None
    golden_cross = 4.0 if (ma50 and ma200 and ma50 > ma200) else 0.0

    # RSI score (0–4): ideal band 40–70
    rsi_score = 0.0
    if rsi is not None:
        if 40 <= rsi <= 70:
            rsi_score = 4.0
        elif rsi < 40:
            rsi_score = max(0.0, 4.0 * (rsi - 20) / 20)
        else:  # > 70 (overbought)
            rsi_score = max(0.0, 4.0 * (90 - rsi) / 20)

    # Volume trend: 5-day avg vs 90-day avg
    vol_trend_raw = None
    if hist is not None and "Volume" in hist.columns:
        vol = hist["Volume"].dropna()
        if len(vol) >= 90:
            avg5 = float(vol.iloc[-5:].mean())
            avg90 = float(vol.iloc[-90:].mean())
            vol_trend_raw = (avg5 / avg90 - 1.0) if avg90 > 0 else None

    # ── Momentum ──────────────────────────────────────────────────────────
    spy_closes = spy_hist["Close"].dropna() if spy_hist is not None and not spy_hist.empty else pd.Series([], dtype=float)

    def excess_return(lookback: int) -> Optional[float]:
        stock_ret = period_return(closes, lookback)
        spy_ret = period_return(spy_closes, lookback)
        if stock_ret is None:
            return None
        if spy_ret is None:
            return stock_ret  # no benchmark → use absolute return
        return stock_ret - spy_ret

    ret_1mo = period_return(closes, 22)
    ret_3mo = period_return(closes, 66)
    ret_6mo = period_return(closes, 126)
    exc_1mo = excess_return(22)
    exc_3mo = excess_return(66)
    exc_6mo = excess_return(126)

    return ScoredStock(
        ticker=raw.ticker,
        sector=raw.sector,
        name=info.get("longName") or info.get("shortName") or raw.ticker,
        price=price,
        exchange_raw=info.get("exchange", ""),
        # display metrics
        rev_growth=rev_growth,
        eps_growth=eps_growth,
        profit_margin=profit_margin,
        roe=roe,
        pe_trailing=pe_trailing,
        pe_forward=pe_forward,
        peg=peg,
        rsi=rsi,
        ma50=ma50,
        ma200=ma200,
        ret_1mo=ret_1mo,
        ret_3mo=ret_3mo,
        ret_6mo=ret_6mo,
        # raw sub-values for batch normalization
        _raw={
            # Momentum components (3 × ~8.3 pts each)
            "exc_1mo": exc_1mo,
            "exc_3mo": exc_3mo,
            "exc_6mo": exc_6mo,
            # Fundamentals (4 × 10 pts)
            "rev_growth": rev_growth,
            "eps_growth": eps_growth,
            "profit_margin": profit_margin,
            "roe": roe,
            # Technical — batch-normalized components
            "price_vs_ma50": price_vs_ma50,
            "price_vs_ma200": price_vs_ma200,
            "vol_trend": vol_trend_raw,
            # Technical — already-scored (binary/custom)
            "_golden_cross": golden_cross,
            "_rsi_score": rsi_score,
            # Valuation
            "peg_raw": peg_raw,
            "pe_direction": pe_direction_raw,
        },
    )


def apply_batch_normalization(stocks: List[ScoredStock]) -> None:
    """Normalize raw sub-scores across the batch and populate final sub-scores."""
    n = len(stocks)
    if n == 0:
        return

    def get_raw(key: str) -> List[Optional[float]]:
        return [s._raw.get(key) for s in stocks]

    # ── Momentum (25 pts) ─────────────────────────────────────────────────
    norm_1mo  = normalize_component(get_raw("exc_1mo"),  8.33)
    norm_3mo  = normalize_component(get_raw("exc_3mo"),  8.33)
    norm_6mo  = normalize_component(get_raw("exc_6mo"),  8.34)
    for i, s in enumerate(stocks):
        s.momentum_score = round(norm_1mo[i] + norm_3mo[i] + norm_6mo[i], 2)

    # ── Fundamentals (40 pts) ─────────────────────────────────────────────
    norm_rev  = normalize_component(get_raw("rev_growth"),    10.0)
    norm_eps  = normalize_component(get_raw("eps_growth"),    10.0)
    norm_marg = normalize_component(get_raw("profit_margin"), 10.0)
    norm_roe  = normalize_component(get_raw("roe"),           10.0)
    for i, s in enumerate(stocks):
        s.fundamental_score = round(norm_rev[i] + norm_eps[i] + norm_marg[i] + norm_roe[i], 2)

    # ── Technical (25 pts) ────────────────────────────────────────────────
    norm_ma50  = normalize_component(get_raw("price_vs_ma50"),  7.0)
    norm_ma200 = normalize_component(get_raw("price_vs_ma200"), 6.0)
    norm_vol   = normalize_component(get_raw("vol_trend"),      4.0)
    for i, s in enumerate(stocks):
        golden = s._raw.get("_golden_cross", 0.0) or 0.0
        rsi_sc = s._raw.get("_rsi_score",    0.0) or 0.0
        s.technical_score = round(norm_ma50[i] + norm_ma200[i] + golden + rsi_sc + norm_vol[i], 2)

    # ── Valuation (10 pts) ────────────────────────────────────────────────
    norm_peg   = normalize_component(get_raw("peg_raw"),    5.0)
    norm_pe_d  = normalize_component(get_raw("pe_direction"), 5.0)
    for i, s in enumerate(stocks):
        s.valuation_score = round(norm_peg[i] + norm_pe_d[i], 2)


def score_all(
    raw_data: Dict[str, RawStock], spy_hist: Optional[pd.DataFrame]
) -> List[ScoredStock]:
    """Extract raw scores, then batch-normalize."""
    log.info("Scoring %d stocks …", len(raw_data))
    stocks: List[ScoredStock] = []
    for ticker, raw in raw_data.items():
        try:
            s = extract_raw_scores(raw, spy_hist)
            stocks.append(s)
        except Exception as e:
            log.warning("Scoring failed for %s: %s", ticker, e)

    apply_batch_normalization(stocks)
    stocks.sort(key=lambda s: s.total_score, reverse=True)
    log.info("Scoring complete. Top ticker: %s (%.1f pts)", stocks[0].ticker if stocks else "N/A",
             stocks[0].total_score if stocks else 0)
    return stocks


# ─── TradingView Watchlist File ───────────────────────────────────────────────

def generate_tv_watchlist(
    top_n: List[ScoredStock],
    all_scored: List[ScoredStock],
    output_dir: str,
) -> str:
    today = date.today().isoformat()
    lines = [f"### STOCK WATCHLIST TOP PICKS — {today} ###"]
    for s in top_n:
        lines.append(s.tv_symbol)

    lines.append("")
    lines.append("### FULL SCREENED UNIVERSE (by score) ###")
    top_tickers = {s.ticker for s in top_n}
    for s in all_scored:
        if s.ticker not in top_tickers:
            lines.append(s.tv_symbol)

    path = f"{output_dir}/tradingview_watchlist.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info("TradingView watchlist written to %s", path)
    return path


# ─── Pine Script ──────────────────────────────────────────────────────────────

def generate_pine_script(top_n: List[ScoredStock], output_dir: str) -> str:
    today = date.today().isoformat()
    top_symbols = ", ".join(s.tv_symbol for s in top_n)

    pine = f"""\
//@version=5
// ─────────────────────────────────────────────────────────────────────────────
// Stock Watchlist Screener — generated {today}
// Top picks: {top_symbols}
//
// Add this script to any chart in TradingView Pine Editor.
// Set alerts using the alertcondition() calls at the bottom.
// Compatible with TradingView FREE accounts.
// ─────────────────────────────────────────────────────────────────────────────
indicator("Stock Watchlist Screener", overlay=true, max_bars_back=500)

// ── Inputs ────────────────────────────────────────────────────────────────────
rsi_low   = input.int(40,  "RSI Lower Bound",  minval=0,   maxval=100)
rsi_high  = input.int(70,  "RSI Upper Bound",  minval=0,   maxval=100)
rsi_len   = input.int(14,  "RSI Length",       minval=2,   maxval=50)
ma_fast   = input.int(50,  "Fast MA Period",   minval=5,   maxval=200)
ma_slow   = input.int(200, "Slow MA Period",   minval=20,  maxval=500)
vol_mult  = input.float(1.2, "Volume Surge Multiplier", minval=1.0, maxval=5.0, step=0.1)

// ── Calculations ──────────────────────────────────────────────────────────────
rsi_val  = ta.rsi(close, rsi_len)
ma50     = ta.sma(close, ma_fast)
ma200    = ta.sma(close, ma_slow)
ret_6mo  = (close - close[126]) / close[126] * 100
vol_avg  = ta.sma(volume, 20)
vol_surge = volume > vol_avg * vol_mult

is_above_ma50  = close > ma50
is_above_ma200 = close > ma200
golden_cross   = ma50 > ma200
golden_event   = ta.crossover(ma50, ma200)
death_event    = ta.crossunder(ma50, ma200)

// ── Composite Bull Setup ──────────────────────────────────────────────────────
rsi_ok     = rsi_val >= rsi_low and rsi_val <= rsi_high
bull_setup = rsi_ok and golden_cross and is_above_ma50 and ret_6mo > 0

// ── Visuals ───────────────────────────────────────────────────────────────────
plot(ma50,  "MA 50",  color=color.new(color.blue,   0), linewidth=2)
plot(ma200, "MA 200", color=color.new(color.orange, 0), linewidth=2)

bgcolor(
  bull_setup ? color.new(color.green, 92) : na,
  title="Bull Setup Background"
)

plotshape(
  golden_event,
  title="Golden Cross",
  style=shape.labelup,
  location=location.belowbar,
  color=color.green,
  textcolor=color.white,
  text="GX",
  size=size.small
)

plotshape(
  death_event,
  title="Death Cross",
  style=shape.labeldown,
  location=location.abovebar,
  color=color.red,
  textcolor=color.white,
  text="DX",
  size=size.small
)

plotshape(
  bull_setup and vol_surge and barstate.isconfirmed,
  title="High-Conviction Signal",
  style=shape.triangleup,
  location=location.belowbar,
  color=color.lime,
  size=size.normal
)

// ── Info Table ────────────────────────────────────────────────────────────────
var table info_tbl = table.new(
  position.top_right, 2, 6,
  bgcolor=color.new(color.black, 60),
  border_color=color.new(color.gray, 50),
  border_width=1,
  frame_color=color.new(color.gray, 30),
  frame_width=1
)

color_yn(b) =>
  b ? color.lime : color.red

if barstate.islast
    table.cell(info_tbl, 0, 0, "RSI",         text_color=color.silver, text_size=size.small)
    table.cell(info_tbl, 1, 0, str.tostring(math.round(rsi_val, 1)),
               text_color=rsi_ok ? color.lime : color.orange, text_size=size.small)

    table.cell(info_tbl, 0, 1, "MA50>MA200",  text_color=color.silver, text_size=size.small)
    table.cell(info_tbl, 1, 1, golden_cross ? "YES" : "NO",
               text_color=color_yn(golden_cross), text_size=size.small)

    table.cell(info_tbl, 0, 2, ">MA50",        text_color=color.silver, text_size=size.small)
    table.cell(info_tbl, 1, 2, is_above_ma50 ? "YES" : "NO",
               text_color=color_yn(is_above_ma50), text_size=size.small)

    table.cell(info_tbl, 0, 3, "6mo Ret %",   text_color=color.silver, text_size=size.small)
    table.cell(info_tbl, 1, 3, str.tostring(math.round(ret_6mo, 1)) + "%",
               text_color=ret_6mo > 0 ? color.lime : color.red, text_size=size.small)

    table.cell(info_tbl, 0, 4, "Vol Surge",   text_color=color.silver, text_size=size.small)
    table.cell(info_tbl, 1, 4, vol_surge ? "YES" : "NO",
               text_color=color_yn(vol_surge), text_size=size.small)

    table.cell(info_tbl, 0, 5, "SETUP",       text_color=color.silver, text_size=size.small)
    table.cell(info_tbl, 1, 5, bull_setup ? "PASS" : "FAIL",
               text_color=color_yn(bull_setup), text_size=size.small, text_size=size.normal)

// ── Alert Conditions ──────────────────────────────────────────────────────────
alertcondition(
  bull_setup,
  title="Bull Setup",
  message="{{ticker}} meets bull setup criteria (RSI={{plot_0}}, MA50>MA200, price>MA50, 6mo>0)"
)

alertcondition(
  golden_event,
  title="Golden Cross",
  message="{{ticker}} — Golden Cross: MA50 crossed above MA200"
)

alertcondition(
  bull_setup and vol_surge,
  title="High-Conviction Setup",
  message="{{ticker}} — High-conviction: bull setup + volume surge"
)

alertcondition(
  death_event,
  title="Death Cross Warning",
  message="{{ticker}} — Warning: MA50 crossed below MA200 (Death Cross)"
)
"""
    path = f"{output_dir}/screener.pine"
    with open(path, "w") as f:
        f.write(pine)
    log.info("Pine Script written to %s", path)
    return path


# ─── HTML Report ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock Watchlist Report — {{ report_date }}</title>
<style>
:root {
  --bg:       #131722;
  --bg2:      #1e222d;
  --bg3:      #2a2e39;
  --accent:   #2962ff;
  --accent2:  #00bcd4;
  --green:    #26a69a;
  --red:      #ef5350;
  --text:     #d1d4dc;
  --text2:    #787b86;
  --border:   #363c4e;
  --gold:     #f5c518;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 14px; line-height: 1.5; }

/* ─── Header ─────────────────────────────────────────────────────────── */
header { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 24px 32px; display: flex; align-items: flex-end; gap: 24px; flex-wrap: wrap; }
header h1 { font-size: 22px; font-weight: 700; color: #fff; letter-spacing: -0.3px; }
header h1 span { color: var(--accent); }
.header-meta { font-size: 12px; color: var(--text2); line-height: 1.8; }
.pill { display: inline-block; background: var(--accent); color: #fff; border-radius: 12px; padding: 2px 10px; font-size: 11px; font-weight: 600; margin-left: 8px; }

/* ─── Summary Bar ────────────────────────────────────────────────────── */
.summary-bar { display: flex; gap: 24px; padding: 16px 32px; background: var(--bg2); border-bottom: 1px solid var(--border); flex-wrap: wrap; }
.summary-stat { text-align: center; }
.summary-stat .val { font-size: 22px; font-weight: 700; color: #fff; }
.summary-stat .lbl { font-size: 11px; color: var(--text2); text-transform: uppercase; letter-spacing: 0.5px; }

/* ─── Section ────────────────────────────────────────────────────────── */
section { padding: 28px 32px; }
section h2 { font-size: 16px; font-weight: 600; color: #fff; margin-bottom: 16px; border-left: 3px solid var(--accent); padding-left: 10px; }

/* ─── Top Cards ──────────────────────────────────────────────────────── */
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 20px; }

.card { background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; transition: border-color 0.15s; }
.card:hover { border-color: var(--accent); }

.card-header { padding: 16px 18px 12px; display: flex; justify-content: space-between; align-items: flex-start; }
.card-ticker { font-size: 22px; font-weight: 800; color: #fff; letter-spacing: -0.5px; }
.card-name { font-size: 12px; color: var(--text2); margin-top: 2px; max-width: 200px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.card-score-total { text-align: right; }
.card-score-total .score-num { font-size: 28px; font-weight: 800; color: var(--gold); }
.card-score-total .score-lbl { font-size: 10px; color: var(--text2); text-transform: uppercase; }
.card-sector-badge { display: inline-block; font-size: 10px; font-weight: 600; border-radius: 4px; padding: 2px 7px; margin-top: 4px; background: var(--bg3); color: var(--text2); }

.card-price-row { padding: 0 18px 10px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
.card-price { font-size: 18px; font-weight: 700; color: #fff; }
.ret-chip { font-size: 11px; font-weight: 600; border-radius: 4px; padding: 2px 7px; }
.ret-chip.pos { background: rgba(38,166,154,0.15); color: var(--green); }
.ret-chip.neg { background: rgba(239,83,80,0.15); color: var(--red); }
.ret-chip.neu { background: var(--bg3); color: var(--text2); }

/* Score bars */
.score-bars { padding: 10px 18px; display: flex; flex-direction: column; gap: 6px; }
.score-row { display: flex; align-items: center; gap: 8px; }
.score-row .lbl { font-size: 11px; color: var(--text2); width: 90px; flex-shrink: 0; }
.bar-track { flex: 1; height: 6px; background: var(--bg3); border-radius: 3px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 3px; }
.bar-fill.momentum    { background: linear-gradient(90deg, #2979ff, #00bcd4); }
.bar-fill.fundamental { background: linear-gradient(90deg, #26a69a, #66bb6a); }
.bar-fill.technical   { background: linear-gradient(90deg, #ab47bc, #7e57c2); }
.bar-fill.valuation   { background: linear-gradient(90deg, #f5c518, #ff9800); }
.score-row .pts { font-size: 11px; color: var(--text); width: 34px; text-align: right; font-weight: 600; }

/* Metrics table */
.metrics-table { width: 100%; border-collapse: collapse; font-size: 12px; margin: 4px 0; }
.metrics-table td { padding: 4px 18px; border-top: 1px solid var(--border); }
.metrics-table td:first-child { color: var(--text2); }
.metrics-table td:last-child { text-align: right; font-weight: 500; }
.val-pos { color: var(--green); }
.val-neg { color: var(--red); }

/* TV Widget area */
.tv-widget-wrap { padding: 0 0 4px; }
.tv-btn { display: block; margin: 0 18px 14px; background: var(--accent); color: #fff; text-align: center; padding: 8px 0; border-radius: 6px; text-decoration: none; font-size: 12px; font-weight: 600; letter-spacing: 0.3px; transition: opacity 0.15s; }
.tv-btn:hover { opacity: 0.85; }

/* ─── Full Table ─────────────────────────────────────────────────────── */
.table-wrap { overflow-x: auto; border-radius: 8px; border: 1px solid var(--border); }
table.scores { width: 100%; border-collapse: collapse; font-size: 12px; }
table.scores thead th { background: var(--bg3); padding: 8px 12px; text-align: right; font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px; color: var(--text2); cursor: pointer; white-space: nowrap; user-select: none; }
table.scores thead th:first-child, table.scores thead th:nth-child(2), table.scores thead th:nth-child(3) { text-align: left; }
table.scores thead th:hover { color: var(--text); }
table.scores thead th.sort-asc::after  { content: " ▲"; color: var(--accent); }
table.scores thead th.sort-desc::after { content: " ▼"; color: var(--accent); }
table.scores tbody tr { border-top: 1px solid var(--border); transition: background 0.1s; }
table.scores tbody tr:hover { background: var(--bg3); }
table.scores tbody tr.top-pick { border-left: 3px solid var(--gold); }
table.scores tbody td { padding: 7px 12px; text-align: right; white-space: nowrap; }
table.scores tbody td:first-child, table.scores tbody td:nth-child(2), table.scores tbody td:nth-child(3) { text-align: left; }
.rank-badge { display: inline-flex; align-items: center; justify-content: center; width: 22px; height: 22px; border-radius: 50%; background: var(--gold); color: #131722; font-size: 11px; font-weight: 800; }
.score-cell { font-weight: 700; color: var(--gold); }

/* ─── Methodology ────────────────────────────────────────────────────── */
details { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; }
details summary { padding: 14px 18px; cursor: pointer; font-weight: 600; color: var(--text); list-style: none; }
details summary::-webkit-details-marker { display: none; }
details summary::before { content: "▶ "; font-size: 11px; color: var(--accent); }
details[open] summary::before { content: "▼ "; }
.method-body { padding: 0 18px 16px; }
.method-body p { color: var(--text2); margin: 8px 0; font-size: 13px; }
.method-body table { border-collapse: collapse; font-size: 12px; margin: 12px 0; }
.method-body th, .method-body td { padding: 5px 12px; border: 1px solid var(--border); }
.method-body th { background: var(--bg3); color: var(--text2); }

/* ─── Footer ─────────────────────────────────────────────────────────── */
footer { background: var(--bg2); border-top: 1px solid var(--border); padding: 16px 32px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
footer p { font-size: 11px; color: var(--text2); }
footer a { color: var(--accent2); text-decoration: none; }

@media (max-width: 640px) {
  header, section, .summary-bar, footer { padding-left: 16px; padding-right: 16px; }
  .cards { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<header>
  <div>
    <h1>Stock Watchlist Report <span>{{ report_date }}</span></h1>
    <p class="header-meta">
      Universe: {{ total_screened }} stocks across {{ num_sectors }} sectors
      &nbsp;|&nbsp; Style: Growth + Quality
      &nbsp;|&nbsp; Data: Yahoo Finance via yfinance
      <span class="pill">Top {{ top_n }}</span>
    </p>
  </div>
</header>

<div class="summary-bar">
  <div class="summary-stat"><div class="val">{{ total_screened }}</div><div class="lbl">Stocks Screened</div></div>
  <div class="summary-stat"><div class="val">{{ num_sectors }}</div><div class="lbl">Sectors</div></div>
  <div class="summary-stat"><div class="val">{{ top_n }}</div><div class="lbl">Top Picks</div></div>
  <div class="summary-stat"><div class="val">{{ avg_score }}</div><div class="lbl">Avg Top Score</div></div>
  <div class="summary-stat"><div class="val">{{ top_sector }}</div><div class="lbl">Leading Sector</div></div>
</div>

<!-- TOP PICK CARDS -->
<section>
  <h2>Top {{ top_n }} Picks — Detailed View</h2>
  <div class="cards">
  {% for s in top_stocks %}
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-ticker">{{ s.ticker }}</div>
          <div class="card-name">{{ s.name }}</div>
          <div class="card-sector-badge">{{ s.sector }}</div>
        </div>
        <div class="card-score-total">
          <div class="score-num">{{ "%.1f"|format(s.total_score) }}</div>
          <div class="score-lbl">/ 100 pts</div>
        </div>
      </div>

      <div class="card-price-row">
        <span class="card-price">${{ "%.2f"|format(s.price) }}</span>
        {% if s.ret_1mo is not none %}
          <span class="ret-chip {{ 'pos' if s.ret_1mo >= 0 else 'neg' }}">1mo {{ s.fmt(s.ret_1mo, pct=True) }}</span>
        {% endif %}
        {% if s.ret_3mo is not none %}
          <span class="ret-chip {{ 'pos' if s.ret_3mo >= 0 else 'neg' }}">3mo {{ s.fmt(s.ret_3mo, pct=True) }}</span>
        {% endif %}
        {% if s.ret_6mo is not none %}
          <span class="ret-chip {{ 'pos' if s.ret_6mo >= 0 else 'neg' }}">6mo {{ s.fmt(s.ret_6mo, pct=True) }}</span>
        {% endif %}
      </div>

      <div class="score-bars">
        <div class="score-row">
          <span class="lbl">Momentum</span>
          <div class="bar-track"><div class="bar-fill momentum" style="width:{{ (s.momentum_score/25*100)|int }}%"></div></div>
          <span class="pts">{{ "%.1f"|format(s.momentum_score) }}/25</span>
        </div>
        <div class="score-row">
          <span class="lbl">Fundamentals</span>
          <div class="bar-track"><div class="bar-fill fundamental" style="width:{{ (s.fundamental_score/40*100)|int }}%"></div></div>
          <span class="pts">{{ "%.1f"|format(s.fundamental_score) }}/40</span>
        </div>
        <div class="score-row">
          <span class="lbl">Technical</span>
          <div class="bar-track"><div class="bar-fill technical" style="width:{{ (s.technical_score/25*100)|int }}%"></div></div>
          <span class="pts">{{ "%.1f"|format(s.technical_score) }}/25</span>
        </div>
        <div class="score-row">
          <span class="lbl">Valuation</span>
          <div class="bar-track"><div class="bar-fill valuation" style="width:{{ (s.valuation_score/10*100)|int }}%"></div></div>
          <span class="pts">{{ "%.1f"|format(s.valuation_score) }}/10</span>
        </div>
      </div>

      <table class="metrics-table">
        <tr><td>Rev Growth YoY</td><td class="{{ 'val-pos' if s.rev_growth and s.rev_growth > 0 else 'val-neg' if s.rev_growth and s.rev_growth < 0 else '' }}">{{ s.fmt(s.rev_growth, pct=True) }}</td></tr>
        <tr><td>EPS Growth YoY</td><td class="{{ 'val-pos' if s.eps_growth and s.eps_growth > 0 else 'val-neg' if s.eps_growth and s.eps_growth < 0 else '' }}">{{ s.fmt(s.eps_growth, pct=True) }}</td></tr>
        <tr><td>Net Margin</td><td class="{{ 'val-pos' if s.profit_margin and s.profit_margin > 0 else 'val-neg' if s.profit_margin and s.profit_margin < 0 else '' }}">{{ s.fmt(s.profit_margin, pct=True) }}</td></tr>
        <tr><td>ROE</td><td class="{{ 'val-pos' if s.roe and s.roe > 0 else 'val-neg' if s.roe and s.roe < 0 else '' }}">{{ s.fmt(s.roe, pct=True) }}</td></tr>
        <tr><td>Forward P/E</td><td>{{ s.fmt(s.pe_forward) }}</td></tr>
        <tr><td>PEG Ratio</td><td>{{ s.fmt(s.peg) }}</td></tr>
        <tr><td>RSI (14)</td><td class="{{ 'val-pos' if s.rsi and 40 <= s.rsi <= 70 else 'val-neg' if s.rsi and s.rsi > 70 else '' }}">{{ s.fmt(s.rsi) }}</td></tr>
        <tr><td>MA50 / MA200</td><td>${{ s.fmt(s.ma50) }} / ${{ s.fmt(s.ma200) }}</td></tr>
      </table>

      <div class="tv-widget-wrap">
        <div class="tradingview-widget-container" style="height:220px;">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript"
            src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js"
            async>
          {
            "symbol": "{{ s.tv_symbol }}",
            "width": "100%",
            "height": 220,
            "locale": "en",
            "dateRange": "6M",
            "colorTheme": "dark",
            "isTransparent": true,
            "autosize": true,
            "largeChartUrl": "{{ s.tv_chart_url }}"
          }
          </script>
        </div>
        <a class="tv-btn" href="{{ s.tv_chart_url }}" target="_blank" rel="noopener">
          Open Full Chart in TradingView ↗
        </a>
      </div>
    </div>
  {% endfor %}
  </div>
</section>

<!-- FULL SCORES TABLE -->
<section>
  <h2>All Screened Stocks — Score Breakdown</h2>
  <p style="font-size:12px;color:var(--text2);margin-bottom:12px;">Click any column header to sort. Gold rows are the top picks.</p>
  <div class="table-wrap">
    <table class="scores" id="scores-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Ticker</th>
          <th>Sector</th>
          <th>Price</th>
          <th>1mo</th>
          <th>3mo</th>
          <th>6mo</th>
          <th>RevGrow</th>
          <th>EPSGrow</th>
          <th>Margin</th>
          <th>ROE</th>
          <th>Fwd P/E</th>
          <th>PEG</th>
          <th>RSI</th>
          <th>Mmt</th>
          <th>Fund</th>
          <th>Tech</th>
          <th>Val</th>
          <th>TOTAL</th>
        </tr>
      </thead>
      <tbody>
      {% for s in all_stocks %}
        <tr class="{{ 'top-pick' if loop.index0 < top_n else '' }}">
          <td>{% if loop.index0 < top_n %}<span class="rank-badge">{{ loop.index }}</span>{% else %}{{ loop.index }}{% endif %}</td>
          <td><strong>{{ s.ticker }}</strong></td>
          <td>{{ s.sector }}</td>
          <td>${{ "%.2f"|format(s.price) }}</td>
          <td class="{{ 'val-pos' if s.ret_1mo and s.ret_1mo >= 0 else 'val-neg' if s.ret_1mo and s.ret_1mo < 0 else '' }}">{{ s.fmt(s.ret_1mo, pct=True) }}</td>
          <td class="{{ 'val-pos' if s.ret_3mo and s.ret_3mo >= 0 else 'val-neg' if s.ret_3mo and s.ret_3mo < 0 else '' }}">{{ s.fmt(s.ret_3mo, pct=True) }}</td>
          <td class="{{ 'val-pos' if s.ret_6mo and s.ret_6mo >= 0 else 'val-neg' if s.ret_6mo and s.ret_6mo < 0 else '' }}">{{ s.fmt(s.ret_6mo, pct=True) }}</td>
          <td class="{{ 'val-pos' if s.rev_growth and s.rev_growth >= 0 else 'val-neg' if s.rev_growth and s.rev_growth < 0 else '' }}">{{ s.fmt(s.rev_growth, pct=True) }}</td>
          <td class="{{ 'val-pos' if s.eps_growth and s.eps_growth >= 0 else 'val-neg' if s.eps_growth and s.eps_growth < 0 else '' }}">{{ s.fmt(s.eps_growth, pct=True) }}</td>
          <td class="{{ 'val-pos' if s.profit_margin and s.profit_margin >= 0 else 'val-neg' if s.profit_margin and s.profit_margin < 0 else '' }}">{{ s.fmt(s.profit_margin, pct=True) }}</td>
          <td class="{{ 'val-pos' if s.roe and s.roe >= 0 else 'val-neg' if s.roe and s.roe < 0 else '' }}">{{ s.fmt(s.roe, pct=True) }}</td>
          <td>{{ s.fmt(s.pe_forward) }}</td>
          <td>{{ s.fmt(s.peg) }}</td>
          <td class="{{ 'val-pos' if s.rsi and 40 <= s.rsi <= 70 else 'val-neg' if s.rsi and s.rsi > 70 else '' }}">{{ s.fmt(s.rsi) }}</td>
          <td>{{ "%.1f"|format(s.momentum_score) }}</td>
          <td>{{ "%.1f"|format(s.fundamental_score) }}</td>
          <td>{{ "%.1f"|format(s.technical_score) }}</td>
          <td>{{ "%.1f"|format(s.valuation_score) }}</td>
          <td class="score-cell">{{ "%.1f"|format(s.total_score) }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
</section>

<!-- METHODOLOGY -->
<section>
  <details>
    <summary>Scoring Methodology</summary>
    <div class="method-body">
      <p>Stocks are scored on four dimensions. All components are <strong>min-max normalized within the current batch</strong> — the best stock in each dimension earns full marks. This makes the system self-calibrating: thresholds adjust automatically to current market conditions.</p>
      <table>
        <tr><th>Dimension</th><th>Weight</th><th>Components</th></tr>
        <tr><td>Momentum</td><td>25 pts</td><td>Excess return vs SPY over 1mo, 3mo, 6mo (equally weighted)</td></tr>
        <tr><td>Fundamentals</td><td>40 pts</td><td>EPS growth YoY, Revenue growth YoY, Net profit margin, ROE (10 pts each)</td></tr>
        <tr><td>Technical Setup</td><td>25 pts</td><td>Price vs MA50 (7), Price vs MA200 (6), Golden cross / MA50>MA200 (4), RSI 40-70 band (4), Volume trend (4)</td></tr>
        <tr><td>Valuation</td><td>10 pts</td><td>PEG ratio (lower = better, 5 pts), Forward vs trailing P/E direction (5 pts)</td></tr>
      </table>
      <p>Data source: Yahoo Finance via <a href="https://github.com/ranaroussi/yfinance" target="_blank">yfinance</a>. Scores are computed fresh each run. N/A values score 0 for that component.</p>
      <p>This report is for informational purposes only and does not constitute financial advice.</p>
    </div>
  </details>
</section>

<footer>
  <p>Generated {{ generated_at }} &nbsp;|&nbsp; Data: Yahoo Finance via yfinance &nbsp;|&nbsp;
     <a href="https://github.com/ranaroussi/yfinance" target="_blank">yfinance</a></p>
  <p>For educational/informational use only. Not financial advice.</p>
</footer>

<script>
// Sortable table
(function() {
  const table = document.getElementById('scores-table');
  const headers = table.querySelectorAll('thead th');
  let sortCol = 18, sortAsc = false;  // default: TOTAL desc

  function cellVal(row, col) {
    const txt = row.cells[col].innerText.trim();
    if (txt === 'N/A' || txt === '') return sortAsc ? Infinity : -Infinity;
    // strip rank badge text, $, %, +
    const n = parseFloat(txt.replace(/[^0-9.\\-]/g, ''));
    return isNaN(n) ? (sortAsc ? 'zzz' : '') : n;
  }

  function sort(col) {
    const asc = (col === sortCol) ? !sortAsc : false;
    sortAsc = asc; sortCol = col;
    headers.forEach((h, i) => { h.classList.remove('sort-asc','sort-desc'); if(i===col) h.classList.add(asc?'sort-asc':'sort-desc'); });
    const rows = Array.from(table.tBodies[0].rows);
    rows.sort((a, b) => {
      const av = cellVal(a, col), bv = cellVal(b, col);
      if (av === bv) return 0;
      return (av < bv ? -1 : 1) * (asc ? 1 : -1);
    });
    rows.forEach(r => table.tBodies[0].appendChild(r));
  }

  headers.forEach((h, i) => h.addEventListener('click', () => sort(i)));
  // Apply initial sort indicator
  headers[18].classList.add('sort-desc');
})();
</script>

</body>
</html>
"""


def generate_html_report(
    top_n: List[ScoredStock],
    all_scored: List[ScoredStock],
    output_dir: str,
) -> str:
    today = date.today()
    env = Environment()
    tmpl = env.from_string(HTML_TEMPLATE)

    top_sector_counts: Dict[str, int] = {}
    for s in top_n:
        top_sector_counts[s.sector] = top_sector_counts.get(s.sector, 0) + 1
    top_sector = max(top_sector_counts, key=top_sector_counts.get) if top_sector_counts else "—"
    avg_score = f"{sum(s.total_score for s in top_n) / len(top_n):.1f}" if top_n else "—"

    html = tmpl.render(
        report_date=today.strftime("%B %d, %Y"),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_screened=len(all_scored),
        num_sectors=len(set(s.sector for s in all_scored)),
        top_n=len(top_n),
        avg_score=avg_score,
        top_sector=top_sector,
        top_stocks=top_n,
        all_stocks=all_scored,
    )

    path = f"{output_dir}/watchlist_report_{today.isoformat()}.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("HTML report written to %s", path)
    return path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a stock watchlist report with TradingView integration."
    )
    parser.add_argument("--top-n",     type=int, default=TOP_N,
                        help=f"Number of top picks to feature (default: {TOP_N})")
    parser.add_argument("--output-dir", default=".",
                        help="Directory for output files (default: current directory)")
    args = parser.parse_args()
    top_n_count = args.top_n
    out_dir = args.output_dir.rstrip("/")

    log.info("=" * 60)
    log.info("Stock Watchlist Report Generator")
    log.info("Universe: %d tickers | Top picks: %d | Output: %s",
             sum(len(v) for v in UNIVERSE.values()), top_n_count, out_dir)
    log.info("=" * 60)

    # 1. Fetch data
    raw_data, spy_hist = fetch_all_data(UNIVERSE)
    if not raw_data:
        log.error("No data fetched. Check internet connection and yfinance installation.")
        return

    # 2. Score
    all_scored = score_all(raw_data, spy_hist)
    top_picks = all_scored[:top_n_count]

    log.info("")
    log.info("TOP %d PICKS:", top_n_count)
    log.info("%-5s %-8s %-28s %6s %7s %7s %7s %7s", "Rank", "Ticker", "Name", "Price", "Mmt", "Fund", "Tech", "TOTAL")
    log.info("-" * 75)
    for i, s in enumerate(top_picks, 1):
        log.info("%-5d %-8s %-28s %6.2f %7.1f %7.1f %7.1f %7.1f",
                 i, s.ticker, s.name[:27], s.price,
                 s.momentum_score, s.fundamental_score, s.technical_score, s.total_score)
    log.info("")

    # 3. Generate outputs
    html_path = generate_html_report(top_picks, all_scored, out_dir)
    tv_path   = generate_tv_watchlist(top_picks, all_scored, out_dir)
    pine_path = generate_pine_script(top_picks, out_dir)

    log.info("")
    log.info("=" * 60)
    log.info("All outputs written:")
    log.info("  HTML Report : %s", html_path)
    log.info("  TV Watchlist: %s  (import via TradingView → Watchlist ⋮ → Import List)", tv_path)
    log.info("  Pine Script : %s  (add via Pine Editor → Add to Chart, then set alerts)", pine_path)
    log.info("=" * 60)
    log.info("")
    log.info("TradingView (free account) quick-start:")
    log.info("  1. Watchlist: open TradingView, right-click Watchlist → Import List → select tradingview_watchlist.txt")
    log.info("  2. Charts:    click any ticker in your watchlist to open its chart")
    log.info("  3. Alerts:    open screener.pine in Pine Editor → Add to Chart → Create Alert → choose condition")


if __name__ == "__main__":
    main()
