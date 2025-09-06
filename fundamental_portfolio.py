# file: fundamental_portfolio.py
"""
Enhanced Fundamental Portfolio (robust CLI + duplicate-safe):
- Adds SPX benchmark ingestion + SPX-relative filters (1Y outperformance OR last-quarter acceleration).
- Fixes `SystemExit: 2`: help printed when args missing; unknown args are ignored with a warning.
- **Fixes pivot error**: handle duplicate (gvkey, datadate) gracefully and silence pct_change FutureWarning.
- Daily returns stored as decimals (0.01 == 1%). Printed stats use %.
- **FIXED**: Data type mismatch between mu.index (int64) and R.columns (str) in portfolio optimization.

CLI:
  --stocks_price PATH     CSV with [gvkey, datadate, prccd, ajexdi]
  --stock_selected PATH   CSV with [gvkey, predicted_return, trade_date]
  --output_dir PATH       Output directory (default: ./output)
  --spx PATH              SPX CSV with [date, close] (default: SPX.csv)
  --lookback INT          1Y window trading days (default: 252)
  --accel_win INT         Quarter window trading days (default: 63)
  --accel_metric STR      ['slope','cumret'] (default: slope)
  --max_weight FLOAT      Per-asset cap (default: 0.05)

Duplicate policy (please confirm):
- If multiple prices per (gvkey, datadate), use **last** price of the day when building series.
- When pivoting returns, if duplicates still exist, **average** duplicates per day.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Optional: portfolio optimization; degrade gracefully if missing
try:
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt import risk_models
    from pypfopt import objective_functions
    HAS_PYPORTFOLIOPT = True
except Exception:
    HAS_PYPORTFOLIOPT = False


# -----------------------------
# Argument parsing (robust)
# -----------------------------

def parse_arguments(argv: List[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(
        description='Fundamental Portfolio Optimization (Enhanced)'
    )
    parser.add_argument('--stocks_price', type=str, required=True)
    parser.add_argument('--stock_selected', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--spx', type=str, default='SPX.csv', help='SPX CSV with columns: date, close')
    parser.add_argument('--lookback', type=int, default=252)
    parser.add_argument('--accel_win', type=int, default=63)
    parser.add_argument('--accel_metric', type=str, default='slope', choices=['slope', 'cumret'])
    parser.add_argument('--max_weight', type=float, default=0.1)

    if argv is None:
        argv = sys.argv[1:]

    if len(argv) == 0:
        parser.print_help()
        print('\nExamples:')
        print('  python fundamental_portfolio.py \\\n  --stocks_price data_processor/sp500_tickers_daily_price_20250712.csv \\\n  --stock_selected result/stock_selected.csv \\\n  --spx output/SPX.csv \\\n  --output_dir ./outputs --lookback 252 --accel_win 63 --accel_metric slope')
        print('  python backtest_quickcheck.py --stocks_price ... --stock_selected ... --spx output/SPX.csv')
        return None

    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"! Warning: ignoring unknown args: {unknown}")
    return args


# -----------------------------
# IO helpers
# -----------------------------

def _to_datetime_maybe(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series.astype(str), errors='coerce')


def load_and_preprocess_data(stocks_price_path: str, stock_selected_path: str, spx_path: str
                             ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print('='*72)
    print('Loading and preprocessing data...')
    print('='*72)

    usecols = ['gvkey', 'datadate', 'prccd', 'ajexdi']
    df_price = pd.read_csv(stocks_price_path, usecols=usecols)
    df_price['datadate'] = _to_datetime_maybe(df_price['datadate'])
    df_price = df_price.dropna(subset=['datadate']).copy()
    df_price['adj_price'] = df_price['prccd'] / df_price['ajexdi']
    df_price = df_price[['gvkey', 'datadate', 'adj_price']]
    # --- de-duplicate at source level: keep last price per (gvkey, datadate)
    df_price = (
        df_price.sort_values(['gvkey', 'datadate'])
                .dropna(subset=['adj_price'])
                .groupby(['gvkey', 'datadate'], as_index=False)
                .agg(adj_price=('adj_price', 'last'))
    )

    selected_stock = pd.read_csv(stock_selected_path)
    if 'trade_date' not in selected_stock.columns:
        raise ValueError('stock_selected.csv must include column: trade_date')
    selected_stock['trade_date'] = _to_datetime_maybe(selected_stock['trade_date'])
    selected_stock = selected_stock[selected_stock['trade_date'] >= pd.Timestamp('2018-03-01')].reset_index(drop=True)

    spx_df = pd.read_csv(spx_path)
    required = {'date', 'close'}
    if not required.issubset(set(spx_df.columns)):
        raise ValueError('SPX.csv must have columns: date, close')
    spx_df['date'] = _to_datetime_maybe(spx_df['date'])
    spx_df = spx_df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    spx_df['spx_return'] = spx_df['close'].pct_change(fill_method=None)

    print('✓ Loaded. Price:', df_price.shape, '| Selected:', selected_stock.shape, '| SPX:', spx_df.shape)
    return df_price, selected_stock, spx_df


# -----------------------------
# Intermediate caching
# -----------------------------

def _pickle_path(output_dir: str, name: str) -> str:
    return os.path.join(output_dir, name)


def save_intermediate_results(all_return_table: Dict[pd.Timestamp, pd.DataFrame],
                              all_stocks_info: Dict[pd.Timestamp, pd.DataFrame],
                              output_dir: str) -> None:
    import pickle
    with open(_pickle_path(output_dir, 'all_return_table.pickle'), 'wb') as f:
        pickle.dump(all_return_table, f)
    with open(_pickle_path(output_dir, 'all_stocks_info.pickle'), 'wb') as f:
        pickle.dump(all_stocks_info, f)


def load_intermediate_results(output_dir: str):
    import pickle
    try:
        with open(_pickle_path(output_dir, 'all_return_table.pickle'), 'rb') as f:
            all_return_table = pickle.load(f)
        with open(_pickle_path(output_dir, 'all_stocks_info.pickle'), 'rb') as f:
            all_stocks_info = pickle.load(f)
        print('✓ Loaded intermediate results from pickle.')
        return all_return_table, all_stocks_info
    except FileNotFoundError:
        return None, None


# -----------------------------
# Math helpers
# -----------------------------

def cumret(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return np.nan
    return float(np.prod(1.0 + r.values) - 1.0)


def logcum_series(returns: pd.Series) -> pd.Series:
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return r
    return np.log1p(r).cumsum()


def lin_slope(y: pd.Series) -> float:
    y = pd.Series(y).dropna().astype(float)
    n = len(y)
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    slope, _intercept = np.polyfit(x, y.values, 1)
    return float(slope)


# -----------------------------
# Core computation: historical returns + SPX-based filtering
# -----------------------------

def calculate_historical_returns(
    df_price: pd.DataFrame,
    selected_stock: pd.DataFrame,
    trade_dates: List[pd.Timestamp],
    spx_df: pd.DataFrame,
    lookback: int = 252,
    accel_win: int = 63,
    accel_metric: str = 'slope',  # 'slope' or 'cumret'
) -> Tuple[Dict[pd.Timestamp, pd.DataFrame], Dict[pd.Timestamp, pd.DataFrame]]:
    print('\n' + '='*72)
    print('Calculating historical returns + SPX filters...')
    print('='*72)

    df_price = df_price.sort_values(['gvkey', 'datadate']).copy()
    by_gv = {str(k): g[['datadate', 'adj_price']].reset_index(drop=True)
             for k, g in df_price.groupby('gvkey', sort=False)}

    spx_df = spx_df.sort_values('date').reset_index(drop=True)

    all_return_table: Dict[pd.Timestamp, pd.DataFrame] = {}
    all_stocks_info: Dict[pd.Timestamp, pd.DataFrame] = {}

    t0 = time.time()

    for current_trade_date in sorted(pd.to_datetime(trade_dates)):
        cur_sel = selected_stock[selected_stock['trade_date'] == current_trade_date].copy()
        cur_names = cur_sel['gvkey'].astype(str).unique().tolist()

        if len(cur_names) == 0:
            all_return_table[current_trade_date] = pd.DataFrame(columns=['datadate', 'gvkey', 'adj_price', 'daily_return'])
            all_stocks_info[current_trade_date] = cur_sel.copy()
            continue

        spx_sub = spx_df[spx_df['date'] < current_trade_date]
        if spx_sub.empty:
            all_return_table[current_trade_date] = pd.DataFrame(columns=['datadate', 'gvkey', 'adj_price', 'daily_return'])
            all_stocks_info[current_trade_date] = cur_sel.copy()
            continue

        window_end = spx_sub['date'].iloc[-1]
        start_for_1y = spx_sub['date'].iloc[max(0, len(spx_sub) - lookback)]
        start_for_2q = spx_sub['date'].iloc[max(0, len(spx_sub) - accel_win * 2)]

        spx_1y = spx_df[(spx_df['date'] > start_for_1y) & (spx_df['date'] <= window_end)]['spx_return'].dropna()
        spx_2q = spx_df[(spx_df['date'] > start_for_2q) & (spx_df['date'] <= window_end)]['spx_return'].dropna()
        spx_last_q = spx_2q.tail(accel_win)
        spx_prev_q = spx_2q.head(len(spx_2q) - len(spx_last_q))

        spx_1y_cr = cumret(spx_1y)
        if accel_metric == 'slope':
            spx_q_metric = lin_slope(logcum_series(spx_last_q))
            spx_pq_metric = lin_slope(logcum_series(spx_prev_q))
        else:
            spx_q_metric = cumret(spx_last_q)
            spx_pq_metric = cumret(spx_prev_q)

        kept_rows: List[pd.DataFrame] = []
        kept_info_rows: List[pd.Series] = []

        condA_cnt = 0
        condB_cnt = 0

        for gv in cur_names:
            g = by_gv.get(gv)
            if g is None:
                continue

            g1y = g[(g['datadate'] > start_for_1y) & (g['datadate'] <= window_end)].copy()
            g2q = g[(g['datadate'] > start_for_2q) & (g['datadate'] <= window_end)].copy()
            if len(g1y) < max(30, int(0.5 * lookback)):
                continue

            # ensure unique per day, sorted
            g1y = g1y.drop_duplicates(subset=['datadate'], keep='last').sort_values('datadate')
            g2q = g2q.drop_duplicates(subset=['datadate'], keep='last').sort_values('datadate')

            g1y['daily_return'] = g1y['adj_price'].pct_change(fill_method=None)
            g2q['daily_return'] = g2q['adj_price'].pct_change(fill_method=None)

            gv_1y_cr = cumret(g1y['daily_return'])
            last_q = g2q['daily_return'].dropna().tail(accel_win)
            prev_q = g2q['daily_return'].dropna().head(len(g2q['daily_return'].dropna()) - len(last_q))

            if accel_metric == 'slope':
                gv_q_metric = lin_slope(logcum_series(last_q))
                gv_pq_metric = lin_slope(logcum_series(prev_q))
            else:
                gv_q_metric = cumret(last_q)
                gv_pq_metric = cumret(prev_q)

            condA = np.isfinite(gv_1y_cr) and np.isfinite(spx_1y_cr) and (gv_1y_cr > spx_1y_cr)
            condB = (
                np.isfinite(gv_q_metric)
                and np.isfinite(spx_q_metric)
                and (gv_q_metric > spx_q_metric)
                and np.isfinite(gv_pq_metric)
                and (gv_q_metric > gv_pq_metric)
            )

            if condA:
                condA_cnt += 1
            if condB:
                condB_cnt += 1

            if condA or condB:
                g1y = g1y.dropna(subset=['daily_return'])
                g1y['gvkey'] = gv
                kept_rows.append(g1y[['datadate', 'gvkey', 'adj_price', 'daily_return']])

                row = cur_sel[cur_sel['gvkey'].astype(str) == gv]
                if not row.empty:
                    kept_info_rows.append(row.iloc[0])

        kept_table = (
            pd.concat(kept_rows, ignore_index=True)
            if kept_rows
            else pd.DataFrame(columns=['datadate', 'gvkey', 'adj_price', 'daily_return'])
        )
        kept_info_df = (
            pd.DataFrame(kept_info_rows).reset_index(drop=True)
            if kept_info_rows
            else cur_sel.head(0).copy()
        )

        all_return_table[current_trade_date] = kept_table
        all_stocks_info[current_trade_date] = kept_info_df

        total = len(cur_names)
        keptN = kept_info_df['gvkey'].nunique() if not kept_info_df.empty else 0

        def pct(x: int, d: int) -> str:
            return f"{(100.0 * x / d):.1f}%" if d else '0.0%'

        print(
            f"{current_trade_date.date()} | candidates={total} | 1Y>SPX: {condA_cnt} ({pct(condA_cnt, total)}) "
            f"| accel>SPX & >prevQ: {condB_cnt} ({pct(condB_cnt, total)}) | kept={keptN} ({pct(keptN, total)})"
        )

    print(f"✓ Done. Time: {(time.time() - t0) / 60:.2f} min")
    return all_return_table, all_stocks_info


# -----------------------------
# Matrix builder (duplicate-safe) used by optimizer
# -----------------------------

def hist_to_matrix(hist: pd.DataFrame) -> pd.DataFrame:
    """Pivot long daily returns to wide matrix with duplicate safety.
    - Drops exact duplicate rows.
    - Averages duplicate (datadate, gvkey) observations.
    """
    if hist is None or hist.empty:
        return pd.DataFrame()
    h = hist.copy()
    h['datadate'] = _to_datetime_maybe(h['datadate'])
    h = h.drop_duplicates(subset=['datadate', 'gvkey']).copy()
    # If upstream still supplies duplicates, pivot_table(mean) will aggregate
    R = pd.pivot_table(
        h, index='datadate', columns='gvkey', values='daily_return', aggfunc='mean'
    )
    R = R.sort_index().dropna(how='all', axis=0)
    return R


# -----------------------------
# Portfolio optimization (FIXED: data type mismatch)
# -----------------------------

def perform_portfolio_optimization(
    all_stocks_info: Dict[pd.Timestamp, pd.DataFrame],
    all_return_table: Dict[pd.Timestamp, pd.DataFrame],
    trade_dates: List[pd.Timestamp],
    output_dir: str,
    max_weight: float = 0.05,
) -> pd.DataFrame:
    if not HAS_PYPORTFOLIOPT:
        print('! PyPortfolioOpt not installed. Skipping optimization stage.')
        return pd.DataFrame()

    rows = []

    for td in sorted(pd.to_datetime(trade_dates)):
        info = all_stocks_info.get(td, pd.DataFrame())
        hist = all_return_table.get(td, pd.DataFrame())
        if info.empty or hist.empty:
            continue

        mu = (
            info[['gvkey', 'predicted_return']]
            .dropna()
            .drop_duplicates('gvkey')
            .set_index('gvkey')['predicted_return']
        )

        R = hist_to_matrix(hist)

        # FIXED: Ensure data type consistency between mu.index and R.columns
        mu_index_str = mu.index.astype(str)
        R_columns_str = R.columns.astype(str)
        common = sorted(set(mu_index_str) & set(R_columns_str))
        
        if len(common) < 2:
            continue
            
        # Use string indices for alignment
        mu_aligned = mu.set_axis(mu.index.astype(str)).loc[common]
        R_aligned = R.set_axis(R.columns.astype(str), axis=1)[common]

        Sigma = risk_models.sample_cov(R_aligned, frequency=252)

        ef = EfficientFrontier(mu_aligned, Sigma, weight_bounds=(0.0, float(max_weight)))
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        try:
            ef.max_sharpe()
            w_mean = ef.clean_weights()
        except Exception:
            w_mean = {k: 0.0 for k in common}

        ef_min = EfficientFrontier(None, Sigma, weight_bounds=(0.0, float(max_weight)))
        try:
            ef_min.min_volatility()
            w_min = ef_min.clean_weights()
        except Exception:
            w_min = {k: 0.0 for k in common}

        n = len(common)
        w_eq = {k: 1.0 / n for k in common}

        def stack_weights(tag: str, weights: Dict[str, float], min_weight: float = 0.001):
            """Stack weights with minimum threshold filtering"""
            for k, v in weights.items():
                if v and v > min_weight:  # 只保留大于0.001的权重
                    rows.append({'trade_date': td, 'gvkey': k, 'weight_type': tag, 'weight': float(v)})

        stack_weights('mean_weighted', w_mean)
        stack_weights('minimum_weighted', w_min)
        stack_weights('equally_weighted', w_eq)

    weights_df = pd.DataFrame(rows)
    if not weights_df.empty:
        weights_df['trade_date'] = pd.to_datetime(weights_df['trade_date'])
        weights_df = weights_df.sort_values(['trade_date', 'weight_type', 'gvkey']).reset_index(drop=True)

    save_results(weights_df, output_dir)
    return weights_df


# -----------------------------
# Results writer
# -----------------------------

def save_results(stocks_weight_table: pd.DataFrame, output_dir: str) -> None:
    if stocks_weight_table is None or stocks_weight_table.empty:
        print('! No weights to save.')
        return
    out_mean = os.path.join(output_dir, 'mean_weighted.xlsx')
    out_min = os.path.join(output_dir, 'minimum_weighted.xlsx')
    out_eq = os.path.join(output_dir, 'equally_weighted.xlsx')

    with pd.ExcelWriter(out_mean) as xw:
        stocks_weight_table[stocks_weight_table['weight_type'] == 'mean_weighted'].to_excel(xw, index=False)
    with pd.ExcelWriter(out_min) as xw:
        stocks_weight_table[stocks_weight_table['weight_type'] == 'minimum_weighted'].to_excel(xw, index=False)
    with pd.ExcelWriter(out_eq) as xw:
        stocks_weight_table[stocks_weight_table['weight_type'] == 'equally_weighted'].to_excel(xw, index=False)

    print(f"✓ Saved: {out_mean}, {out_min}, {out_eq}")


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_arguments()
    if args is None:
        return
    os.makedirs(args.output_dir, exist_ok=True)

    df_price, selected_stock, spx_df = load_and_preprocess_data(
        args.stocks_price, args.stock_selected, args.spx
    )

    trade_dates = sorted(pd.to_datetime(selected_stock['trade_date'].unique()))

    all_return_table, all_stocks_info = load_intermediate_results(args.output_dir)

    if all_return_table is None:
        all_return_table, all_stocks_info = calculate_historical_returns(
            df_price=df_price,
            selected_stock=selected_stock,
            trade_dates=trade_dates,
            spx_df=spx_df,
            lookback=args.lookback,
            accel_win=args.accel_win,
            accel_metric=args.accel_metric,
        )
        save_intermediate_results(all_return_table, all_stocks_info, args.output_dir)

    perform_portfolio_optimization(
        all_stocks_info=all_stocks_info,
        all_return_table=all_return_table,
        trade_dates=trade_dates,
        output_dir=args.output_dir,
        max_weight=args.max_weight,
    )


if __name__ == '__main__':
    main()
