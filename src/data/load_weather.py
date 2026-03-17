"""
Load UK weather-demand proxy feature packs.

The UK ETS branch uses a compact daily weather panel intended to proxy heating
demand. The bootstrap script writes already-aggregated national features.
"""

from pathlib import Path
from typing import Union
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_weather_feature_pack(
    data_dir: Union[str, Path],
    relative_path: str = "weather-proxy/uk_weather_daily_feature_pack.csv",
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    pack_path = Path(relative_path)
    if not pack_path.is_absolute():
        pack_path = data_dir / pack_path

    if not pack_path.exists():
        raise FileNotFoundError(f"Weather feature pack not found: {pack_path}")

    df = pd.read_csv(pack_path)
    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    if "date" not in df.columns:
        raise ValueError(f"No date column found in weather feature pack: {pack_path}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")

    for col in df.columns:
        if col == "date":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(
        "Loaded weather feature pack: %s records from %s to %s",
        len(df),
        df["date"].min(),
        df["date"].max(),
    )
    return df.reset_index(drop=True)
