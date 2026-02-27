import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta

ISS_BASE = "https://iss.moex.com/iss"

def fetch_candles(secid: str,
                  from_date: str,
                  till_date: str,
                  interval: int = 24,
                  engine: str = "stock",
                  market: str = "shares") -> pd.DataFrame:
    """
    Fetch OHLCV candles from MOEX ISS.
    Endpoint: /iss/engines/{engine}/markets/{market}/securities/{secid}/candles.json
    Params: from, till, interval, start (pagination)
    """
    url = f"{ISS_BASE}/engines/{engine}/markets/{market}/securities/{secid}/candles.json"
    params = {
        "from": from_date,
        "till": till_date,
        "interval": interval,
        "start": 0
    }

    rows = []
    while True:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()

        candles = js.get("candles", {})
        data = candles.get("data", [])
        cols = candles.get("columns", [])

        if not data:
            break

        chunk = pd.DataFrame(data, columns=cols)
        rows.append(chunk)

        # ISS often paginates; advance start by returned rows
        params["start"] += len(chunk)

        # Safety stop: if you ask for small range, you likely don't need endless paging
        if params["start"] > 5000:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)

    # normalize dtypes
    for c in ["open", "high", "low", "close", "value"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    if "begin" in df.columns:
        df["begin"] = pd.to_datetime(df["begin"], errors="coerce")

    df = df.dropna(subset=["begin", "close"]).sort_values("begin").reset_index(drop=True)
    return df

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = pd.Series(gain, index=series.index).rolling(period).mean()
    loss = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["rsi14"] = rsi(df["close"], 14)
    df["ret5"] = df["close"].pct_change(5)
    df["vol20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / (df["vol20"] + 1e-12)
    df["prev_high"] = df["high"].shift(1)
    return df

def screen_momentum(latest: pd.Series) -> bool:
    return (
        latest["close"] > latest["sma20"] > latest["sma50"]
        and 55 <= latest["rsi14"] <= 70
        and latest["ret5"] > 0
        and latest["vol_ratio"] >= 1.2
    )

def screen_reversal(latest: pd.Series) -> bool:
    return (
        latest["rsi14"] < 30
        and latest["close"] > latest["prev_high"]  # simple "break" over prev day high
        and latest["vol_ratio"] >= 1.2
    )

def score_momentum(latest: pd.Series) -> float:
    # Higher = stronger trend + volume confirmation
    return float((latest["ret5"] * 100) + (latest["vol_ratio"] - 1) * 10 + (latest["rsi14"] - 55))

def score_reversal(latest: pd.Series) -> float:
    # Higher = deeper oversold + stronger reversal + volume
    return float((30 - latest["rsi14"]) + (latest["vol_ratio"] - 1) * 10)

def run_screener(tickers, days_back=140, interval=24):
    till = date.today()
    from_ = till - timedelta(days=days_back)
    from_s, till_s = from_.isoformat(), till.isoformat()

    momentum_hits = []
    reversal_hits = []

    for t in tickers:
        try:
            df = fetch_candles(t, from_s, till_s, interval=interval)
            if df.empty or len(df) < 60:
                continue

            df = add_indicators(df)
            latest = df.iloc[-1]

            # require indicators available
            if pd.isna(latest["sma50"]) or pd.isna(latest["rsi14"]) or pd.isna(latest["vol_ratio"]):
                continue

            if screen_momentum(latest):
                momentum_hits.append({
                    "ticker": t,
                    "date": latest["begin"].date().isoformat(),
                    "close": latest["close"],
                    "rsi14": round(latest["rsi14"], 2),
                    "ret5_%": round(latest["ret5"] * 100, 2),
                    "vol_ratio": round(latest["vol_ratio"], 2),
                    "score": round(score_momentum(latest), 2)
                })

            if screen_reversal(latest):
                reversal_hits.append({
                    "ticker": t,
                    "date": latest["begin"].date().isoformat(),
                    "close": latest["close"],
                    "rsi14": round(latest["rsi14"], 2),
                    "vol_ratio": round(latest["vol_ratio"], 2),
                    "score": round(score_reversal(latest), 2)
                })

        except requests.HTTPError as e:
            print(f"[{t}] HTTP error: {e}")
        except Exception as e:
            print(f"[{t}] error: {e}")

    mom = pd.DataFrame(momentum_hits)
    if not mom.empty:
        mom = mom.sort_values("score", ascending=False).reset_index(drop=True)
    else:
        mom = pd.DataFrame(
            columns=["ticker", "date", "close", "rsi14", "ret5_%", "vol_ratio", "score"]
        )

    rev = pd.DataFrame(reversal_hits)
    if not rev.empty:
        rev = rev.sort_values("score", ascending=False).reset_index(drop=True)
    else:
        rev = pd.DataFrame(
            columns=["ticker", "date", "close", "rsi14", "vol_ratio", "score"]
        )

    return mom, rev

if __name__ == "__main__":
    # Пример: подставь свой список ликвидных тикеров
    TICKERS = [
        "SBER", "GAZP", "LKOH", "ROSN", "NVTK", "GMKN", "PLZL", "TATN",
        "SIBN", "MGNT", "CHMF", "VTBR", "ALRS", "YDEX"
    ]

    momentum, reversal = run_screener(TICKERS, days_back=200, interval=24)

    print("\n=== MOMENTUM (trend continuation) ===")
    print(momentum.head(15).to_string(index=False))

    print("\n=== REVERSAL (oversold bounce) ===")
    print(reversal.head(15).to_string(index=False))
