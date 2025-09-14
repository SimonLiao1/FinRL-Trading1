import argparse
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # for non-interactive environments
import matplotlib.pyplot as plt
import os



def sector_map(
    fundamental_path: str = "./data_processor/sp500_tickers_fundamental_quarterly_20250712.csv",
    out_map_path: str = "./data_processor/gvkey_gsector_map.csv",
    gvkey_col: str = "gvkey",
    gsector_col: str = "gsector",
) -> pd.DataFrame:
    """
    Build a mapping from gvkey -> gsector (GICS sector).
    Strategy:
      - Read the fundamentals file (chunked if CSV is large)
      - Pick the last non-null gsector per gvkey using available date columns if present
      - Save the mapping to out_map_path and return it
    """
    ext = os.path.splitext(fundamental_path)[1].lower()
    usecols_candidates = [gvkey_col, gsector_col, "datadate", "rdq", "datafqtr"]

    if ext in [".csv", ".txt"]:
        chunks = []
        for chunk in pd.read_csv(fundamental_path, chunksize=200000, low_memory=False):
            cols = [c for c in usecols_candidates if c in chunk.columns]
            chunk = chunk[cols]
            # convert date-ish columns
            for dc in ["datadate", "rdq"]:
                if dc in chunk.columns:
                    chunk[dc] = pd.to_datetime(chunk[dc], errors='coerce')
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_excel(fundamental_path)
        cols = [c for c in usecols_candidates if c in df.columns]
        df = df[cols]
        for dc in ["datadate", "rdq"]:
            if dc in df.columns:
                df[dc] = pd.to_datetime(df[dc], errors='coerce')

    if gsector_col not in df.columns or gvkey_col not in df.columns:
        raise ValueError("Fundamental file must include gvkey and gsector columns")

    # Prefer latest available record per gvkey
    sort_keys = []
    for dc in ["datadate", "rdq", "datafqtr"]:
        if dc in df.columns:
            sort_keys.append(dc)
    if sort_keys:
        df = df.sort_values(sort_keys)

    # Drop rows with missing gsector
    df = df.dropna(subset=[gsector_col])
    # Keep the last occurrence per gvkey
    map_df = df.groupby(gvkey_col, as_index=False).tail(1)[[gvkey_col, gsector_col]].copy()

    # Normalize dtypes
    try:
        map_df[gsector_col] = pd.to_numeric(map_df[gsector_col], errors='coerce').astype('Int64')
    except Exception:
        # keep as string if numeric cast fails
        map_df[gsector_col] = map_df[gsector_col].astype(str)

    os.makedirs(os.path.dirname(out_map_path), exist_ok=True)
    map_df.to_csv(out_map_path, index=False)
    return map_df


def load_equity_prices(
    path: str,
    gvkey_col: str = "gvkey",
    date_col: str = "datadate",
    prccd_col: str = "prccd",
    ajexdi_col: str = "ajexdi",
    fundamental_path: str = "./data_processor/sp500_tickers_fundamental_quarterly_20250712.csv",
    out_with_sector_path: str = "sp500_daily_price_with_sectors.csv",
) -> pd.DataFrame:
    """
    Build and save a long-form dataset with columns:
      - date (datetime64[ns] normalized to YYYY-MM-DD)
      - tic_name (str) if available; otherwise NaN
      - gvkey
      - adj_close_q (float)
      - gsector (from fundamentals mapping)

    Also writes the merged output to `out_with_sector_path` and returns the DataFrame.
    """
    # Ensure we have a gvkey->gsector mapping
    map_path = "./data_processor/gvkey_gsector_map.csv"
    if not os.path.exists(map_path):
        sector_map(fundamental_path=fundamental_path, out_map_path=map_path, gvkey_col=gvkey_col)
    gv_map = pd.read_csv(map_path)

    ext = os.path.splitext(path)[1].lower()
    chunk_size = 200000
    chunks: list[pd.DataFrame] = []

    tic_candidates = ["tic", "tic_name", "ticName", "TIC"]

    if ext in [".csv", ".txt"]:
        for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
            required_cols = [c for c in [gvkey_col, date_col, prccd_col, ajexdi_col] if c in chunk.columns]
            missing = set([gvkey_col, date_col, prccd_col, ajexdi_col]) - set(required_cols)
            if missing:
                raise ValueError(f"Price file missing required columns: {missing}")

            keep = chunk[required_cols].copy()
            # Find ticker column if present
            tic_col_found = None
            for tc in tic_candidates:
                if tc in chunk.columns:
                    tic_col_found = tc
                    break
            if tic_col_found:
                keep["tic_name"] = chunk[tic_col_found].astype(str)
            else:
                keep["tic_name"] = pd.NA

            keep[date_col] = pd.to_datetime(keep[date_col], errors='coerce')
            keep[prccd_col] = pd.to_numeric(keep[prccd_col], errors='coerce')
            keep[ajexdi_col] = pd.to_numeric(keep[ajexdi_col], errors='coerce')
            keep = keep.dropna(subset=[date_col, prccd_col, ajexdi_col])
            keep["adj_close_q"] = keep[prccd_col] / keep[ajexdi_col]

            # Normalize date to date (no time)
            keep["date"] = keep[date_col].dt.normalize()
            # Select and rename columns
            keep = keep[["date", "tic_name", gvkey_col, "adj_close_q"]]

            chunks.append(keep)
        df_long = pd.concat(chunks, ignore_index=True)
    else:
        raw = pd.read_excel(path)
        missing = [c for c in [gvkey_col, date_col, prccd_col, ajexdi_col] if c not in raw.columns]
        if missing:
            raise ValueError(f"Price file missing required columns: {missing}")
        df_long = raw[[gvkey_col, date_col, prccd_col, ajexdi_col]].copy()
        # ticker
        tic_col_found = None
        for tc in tic_candidates:
            if tc in raw.columns:
                tic_col_found = tc
                break
        if tic_col_found:
            df_long["tic_name"] = raw[tic_col_found].astype(str)
        else:
            df_long["tic_name"] = pd.NA

        df_long[date_col] = pd.to_datetime(df_long[date_col], errors='coerce')
        df_long[prccd_col] = pd.to_numeric(df_long[prccd_col], errors='coerce')
        df_long[ajexdi_col] = pd.to_numeric(df_long[ajexdi_col], errors='coerce')
        df_long = df_long.dropna(subset=[date_col, prccd_col, ajexdi_col])
        df_long["adj_close_q"] = df_long[prccd_col] / df_long[ajexdi_col]
        df_long["date"] = pd.to_datetime(df_long[date_col]).dt.normalize()
        df_long = df_long[["date", "tic_name", gvkey_col, "adj_close_q"]]

    # Merge sector mapping
    df_long = df_long.merge(gv_map, how='left', left_on=gvkey_col, right_on="gvkey")
    if "gvkey_y" in df_long.columns:
        # Remove duplicate column from merge
        df_long = df_long.drop(columns=["gvkey_y"]).rename(columns={"gvkey_x": "gvkey"})

    # Ensure dtypes and ordering
    df_long["date"] = pd.to_datetime(df_long["date"]).dt.date
    # Save output
    df_long.to_csv(out_with_sector_path, index=False)

    return df_long

def plot_turbulence_contrib_stack(
    contrib_csv: str = "turbulence_sector_contrib.csv",
    out_png: str = "turbulence_contrib_stack.png",
    top_k: int = 8,
    normalize: bool = False,
    turbulence_csv: str | None = None,
) -> None:
    """
    Make a stacked area chart to explain daily Turbulence by sector contributions.

    Parameters
    ----------
    contrib_csv : str
        CSV produced by sector_turbulence_from_long.py (`turbulence_sector_contrib.csv`).
        Index should be date; columns are sector names/codes; values are per-sector quadratic contribution
        defined as diff_i * (Precision @ diff)_i.
    out_png : str
        Output PNG path.
    top_k : int
        Keep the top_k sectors by mean absolute contribution; merge the rest into 'Others'.
    normalize : bool
        If True, row-normalize positive and negative stacks by total turbulence so each day's area sums to 1.
        (Uses absolute contributions for the denominator to be numerically stable.)
    turbulence_csv : Optional[str]
        Optional CSV path with a single column 'turbulence' to overlay on a secondary axis.

    Notes
    -----
    - Contributions can be positive or negative. We draw two stacks in a single chart:
        (1) Positive contributions stacked above zero
        (2) Negative contributions stacked below zero (stacked by absolute value, plotted as negative)
    - We avoid specifying colors (matplotlib chooses defaults).
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # 1) Load contribution table
    C = pd.read_csv(contrib_csv, index_col=0)
    # Parse date-like index robustly
    try:
        C.index = pd.to_datetime(C.index)
    except Exception:
        # try common formats
        C.index = pd.to_datetime(C.index, errors='coerce', infer_datetime_format=True)
    C = C.sort_index()
    C = C.fillna(0.0)

    if C.empty or C.shape[1] == 0:
        raise ValueError("No data found in contributions CSV.")

    # Ensure columns are str (sector labels)
    C.columns = [str(c) for c in C.columns]

    # 2) Select top_k sectors by mean absolute contribution
    mean_abs = C.abs().mean().sort_values(ascending=False)
    keep = list(mean_abs.head(max(1, top_k)).index)
    drop = [c for c in C.columns if c not in keep]

    C_top = C[keep].copy()
    if drop:
        C_top['Others'] = C[drop].sum(axis=1)

    # 3) Split into positive and negative stacks
    POS = C_top.clip(lower=0.0)
    NEG = (-C_top.clip(upper=0.0))  # take absolute for stacking, will plot as negative

    # Optional normalization by daily total absolute contribution
    if normalize:
        denom = POS.sum(axis=1) + NEG.sum(axis=1)
        denom = denom.replace(0, np.nan)
        POS = POS.div(denom, axis=0)
        NEG = NEG.div(denom, axis=0)

    # 4) Plot
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    # Positive stack
    POS.plot.area(ax=ax, linewidth=0.8, alpha=0.9)
    # Negative stack (plot as negative values)
    (-NEG).plot.area(ax=ax, linewidth=0.8, alpha=0.9)

    ax.set_title("Turbulence Sector Contribution (stacked; pos above / neg below)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Contribution" + (" (normalized)" if normalize else ""))

    # Optional overlay of total turbulence
    if turbulence_csv is not None:
        try:
            T = pd.read_csv(turbulence_csv, index_col=0)
            if 'turbulence' in T.columns:
                T.index = pd.to_datetime(T.index)
                T = T.sort_index()
                # Align to contributions
                aligned = T.reindex(C_top.index).dropna()
                if not aligned.empty:
                    ax2 = ax.twinx()
                    aligned['turbulence'].plot(ax=ax2, linewidth=1.2)
                    ax2.set_ylabel("Turbulence")
        except Exception:
            # Silent fail if file not available or malformed
            pass

    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

# === Backtest Function for timing strategy===
def run_backtest(df, params):
    """
    according to "./outputs/vix_z.csv" and the best parameter set in "./outputs/Timing_strategy_top5.csv" 
    to backtest the timing strategy, and return the following metrics:
    ann_ret, ann_vol, sharpe, max_dd, calmar, timing_equity, date
     params={
    "zscore_thr": 2.5,
    "dzscore_thr": 2.0,
    "dzscore_neg_thr": -2.0,
    "ma_buy": 5,
    "ma_sell": 20,
    "cooldown_buy": 3,
    "cooldown_sell": 5
    }

    return_df=  {
    "params": params,
    "ann_ret": ann_ret,
    "ann_vol": ann_vol,
    "sharpe": sharpe,
    "max_dd": max_dd,
    "calmar": calmar,
    "timing_equity": df['timing_equity'],
    "date": df['date'],
    "position": df['position']   
     
}
also write to "./outputs/daily_probabilities.csv"

    """
    zscore_thr = params.get("zscore_thr", 3.0)
    dzscore_thr = params.get("dzscore_thr", 2.0)
    dzscore_neg_thr = params.get("dzscore_neg_thr", -3.0)
    ma_buy = params.get("ma_buy", 5)
    ma_sell = params.get("ma_sell", 10)
    cooldown_buy = params.get("cooldown_buy", 3)
    cooldown_sell = params.get("cooldown_sell", 5)

    # Compute moving averages
    df['spx_ma_buy'] = df['spx_close'].rolling(ma_buy).mean()
    df['spx_ma_sell'] = df['spx_close'].rolling(ma_sell).mean()

    position = []
    last_buy_day = -999
    last_sell_day = -999
    current_pos = 1.0

    for i in range(len(df)):
        row = df.iloc[i]
        day = i
        pos = current_pos

        # Buy condition
        if day - last_buy_day >= cooldown_buy:
            if (row['zscore'] < -1 or row['delta_zscore'] < dzscore_neg_thr) and row['spx_close'] > row['spx_ma_buy']:
                pos = 1.0
                last_buy_day = day

        # Sell condition
        if day - last_sell_day >= cooldown_sell:
            if (row['zscore'] > zscore_thr or row['delta_zscore'] > dzscore_thr) and row['spx_close'] < row['spx_ma_sell']:
                pos = 0.5
                last_sell_day = day

        position.append(pos)
        current_pos = pos

    df['position'] = position
    df['timing_ret'] = df['position'].shift(1).fillna(1.0) * df['port_ret']
    df['timing_equity'] = (1 + df['timing_ret']).cumprod()

    ann_ret, ann_vol, sharpe, max_dd, calmar = performance_stats(df['timing_ret'], df['timing_equity'])

    res = {
        "params": params,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "timing_equity": df['timing_equity'],   
        "date": df['date']   ,     
        "position": df['position']    #--->0.5 /1              
    }
    out_df = pd.DataFrame({
    "date": res["date"],
    "prob": res["position"]   # use position as Prob
    })
    out_df.to_csv("./outputs/daily_probabilities_spx_vix_z.csv", index=False)
    return res
# === Performance Metrics ===
def performance_stats(returns, equity):
    ann_ret = (equity.iloc[-1]) ** (252/len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
    max_dd = ((equity / equity.cummax()) - 1).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    return ann_ret, ann_vol, sharpe, max_dd, calmar

if __name__ == "__main__":
    # Minimal CLI for quick testing from terminal:
    import argparse as _argparse
    _ap = _argparse.ArgumentParser(description="Plot stacked sector contributions to Turbulence")
    _ap.add_argument("--contrib", default="turbulence_sector_contrib.csv")
    _ap.add_argument("--out", default="turbulence_contrib_stack.png")
    _ap.add_argument("--topk", type=int, default=8)
    _ap.add_argument("--normalize", action="store_true")
    _ap.add_argument("--turb", default=None, help="Optional path to turbulence_sector.csv")
    _args = _ap.parse_args()
    plot_turbulence_contrib_stack(
        contrib_csv=_args.contrib,
        out_png=_args.out,
        top_k=_args.topk,
        normalize=_args.normalize,
        turbulence_csv=_args.turb,
    )
