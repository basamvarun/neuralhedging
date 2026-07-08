"""
loader.py — Raw Data Loader

Reads the two raw NIFTY option CSV files (expiry day and non-expiry day),
parses symbol names to extract strike price and option type (CE/PE),
computes time-to-expiry T in years, and returns a clean DataFrame
ready for feature engineering.

CSV columns: date, minute_end, symbol, last_trade_price
"""

import pandas as pd
import numpy as np
from pathlib import Path

EXPIRY_DT = pd.Timestamp("2026-02-05 15:30:00")
RISK_FREE_RATE = 0.05  # 5% annualised


def parse_symbol(symbol: str):
    """
    Extract (strike_price, option_type) from a NIFTY symbol string.
    Example: 'NIFTY2621025700CE' -> (2570000.0, 'call')
             'NIFTY2621025700PE' -> (2570000.0, 'put')
             'NIFTY26FEBFUT'     -> (None, None)
    Strike = last 5 digits before CE/PE * 100 (to convert to paise scale).
    """
    if "CE" in symbol:
        option_type = "call"
        digits = "".join(filter(str.isdigit, symbol.split("CE")[0]))
        strike = float(digits[-5:]) * 100.0 if len(digits) >= 5 else None
    elif "PE" in symbol:
        option_type = "put"
        digits = "".join(filter(str.isdigit, symbol.split("PE")[0]))
        strike = float(digits[-5:]) * 100.0 if len(digits) >= 5 else None
    else:
        return None, None
    return strike, option_type


def load_raw_csv(path: str) -> pd.DataFrame:
    """Load one raw CSV and add a parsed datetime column."""
    df = pd.read_csv(path)
    df["observation_datetime"] = pd.to_datetime(
        df["date"].astype(str) + df["minute_end"].astype(str).str.zfill(6),
        format="%Y%m%d%H%M%S",
    )
    return df


def load_and_parse(csv_paths: list, target_symbol: str) -> pd.DataFrame:
    """
    Load CSVs, filter for the target option + futures, merge on timestamp,
    and compute T (time-to-expiry in years).

    Args:
        csv_paths:     List of raw CSV file paths.
        target_symbol: e.g. 'NIFTY2621025700CE'

    Returns:
        DataFrame with columns:
            observation_datetime, S, option_price,
            strike, option_type, T, r
    """
    frames = [load_raw_csv(p) for p in csv_paths]
    df_all = pd.concat(frames, ignore_index=True)

    # Futures (underlying price)
    futures_df = (
        df_all[df_all["symbol"].str.contains("FUT")]
        .rename(columns={"last_trade_price": "S"})
        [["observation_datetime", "S"]]
        .copy()
    )

    # Target option
    option_df = (
        df_all[df_all["symbol"] == target_symbol]
        .rename(columns={"last_trade_price": "option_price"})
        [["observation_datetime", "option_price", "symbol"]]
        .copy()
    )

    merged = pd.merge(futures_df, option_df, on="observation_datetime")
    merged = merged.sort_values("observation_datetime").reset_index(drop=True)

    strike, option_type = parse_symbol(target_symbol)
    merged["strike"] = strike
    merged["option_type"] = option_type

    merged["T"] = (
        (EXPIRY_DT - merged["observation_datetime"]).dt.total_seconds()
        / (365.25 * 24 * 3600)
    ).clip(lower=0.0)

    merged["r"] = RISK_FREE_RATE
    return merged


def load_all_options(csv_paths: list) -> pd.DataFrame:
    """
    Load all CE/PE options from the CSVs (not just one symbol).
    Useful for IV surface analysis.
    """
    frames = [load_raw_csv(p) for p in csv_paths]
    df_all = pd.concat(frames, ignore_index=True)

    futures_df = (
        df_all[df_all["symbol"].str.contains("FUT")]
        .rename(columns={"last_trade_price": "S"})
        [["observation_datetime", "S"]]
    )

    options_df = df_all[~df_all["symbol"].str.contains("FUT")].copy()
    options_df[["strike", "option_type"]] = options_df["symbol"].apply(
        lambda s: pd.Series(parse_symbol(s))
    )
    options_df.dropna(subset=["strike", "option_type"], inplace=True)
    options_df = options_df.rename(columns={"last_trade_price": "option_price"})

    merged = pd.merge(futures_df, options_df, on="observation_datetime")
    merged["T"] = (
        (EXPIRY_DT - merged["observation_datetime"]).dt.total_seconds()
        / (365.25 * 24 * 3600)
    ).clip(lower=0.0)
    merged["r"] = RISK_FREE_RATE

    return merged.sort_values(["observation_datetime", "symbol"]).reset_index(drop=True)
