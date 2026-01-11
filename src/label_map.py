# src/label_map.py
from __future__ import annotations

import re
import pandas as pd


GROUPS_7 = [
    "Trafik",
    "Våld",
    "Stöld/Inbrott",
    "Brand",
    "Rattfylleri/LOB",
    "Sammanfattning",
    "Övrigt",
]


def normalize_text(x: str) -> str:
    x = (x or "").strip()
    x = re.sub(r"\s+", " ", x)
    return x


def map_type_to_group(event_type: str) -> str:
    """
    Map raw Polisen event 'type' (Swedish) -> 7 grouped classes.
    Robust to new/unknown labels.
    """
    t = normalize_text(event_type).lower()

    if not t:
        return "Övrigt"

    # 2) Traffic
    trafik_keywords = [
        "trafik", "körning", "rattfyll", "fordon", "hastighet",
        "vilt", "personskada", "trafikkontroll", "trafikbrott", "trafikhinder"
    ]
    if any(k in t for k in trafik_keywords):
        # Note: we keep Rattfylleri/LOB separate below, so don't exit yet for those
        # But "rattfyll" appears here as traffic too; we'll override in the next rule.
        traf = True
    else:
        traf = False

    # 3) Drunk / LOB
    if "fylleri" in t or "lob" in t or "rattfyll" in t:
        return "Rattfylleri/LOB"

    if traf:
        return "Trafik"

    # 4) Violence
    vald_keywords = [
        "misshandel", "våld", "våldt", "mord", "dråp", "rån", "bråk",
        "olaga hot", "hot", "hemfridsbrott", "vållande", "människorov",
        "skottlossning", "explosion", "sexualbrott"
    ]
    if any(k in t for k in vald_keywords):
        return "Våld"

    # 5) Theft / burglary
    if "stöld" in t or "inbrott" in t or "rån" in t:
        return "Stöld/Inbrott"

    # 6) Fire
    if "brand" in t:
        return "Brand"

    return "Övrigt"


def add_type_group(df: pd.DataFrame, src_col: str = "type", dst_col: str = "type_group") -> pd.DataFrame:
    """
    Adds a new column type_group based on column `type`.
    """
    if src_col not in df.columns:
        raise ValueError(f"Missing column '{src_col}' in dataframe. Available: {list(df.columns)}")

    df = df.copy()
    df[dst_col] = df[src_col].astype(str).map(map_type_to_group)
    return df
