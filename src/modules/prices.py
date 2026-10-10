"""Price loader: yfinance if available/online, else reproducible simulated prices."""
import numpy as np
import pandas as pd


def get_prices(tickers, start, end, simulate=False):
    if not simulate:
        try:
            import yfinance as yf
            df = yf.download(tickers, start=start, end=end + pd.Timedelta(days=1),
                             auto_adjust=True, progress=False)["Close"]
            if isinstance(df, pd.Series):
                df = df.to_frame(tickers[0])
            df = df.dropna(how="all").ffill()
            if len(df) > 3 and df.shape[1] == len(tickers):
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                return df
        except Exception:
            pass
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0004, 0.015, (len(idx), len(tickers)))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=tickers)
