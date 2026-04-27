import pandas as pd
from pathlib import Path

DATA_DIR = Path("data 2024")
OUT_DIR = Path("data/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRICE_FILES = [
    DATA_DIR / "spotpriser_tyskland_2024_riktig.csv",
]

SOLAR_FILES = [
    DATA_DIR / "solproduksjon_tyskland_2024_riktig.csv",
]

SOLAR_PEAK_MW = 1.0


def read_csv_flexible(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep=";")

    if len(df.columns) == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";")

    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df


def find_time_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)

    for c in ["time", "Time", "Hour", "#Hour", "Datetime", "datetime", "Date", "date"]:
        if c in cols:
            return c

    lowered = {c.lower(): c for c in cols}

    for key in ["hour", "time", "date", "datetime", "#hour"]:
        for low, orig in lowered.items():
            if key in low:
                return orig

    raise ValueError(f"Fant ingen tidkolonne. Kolonner: {cols}")


def parse_time_series(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, errors="coerce", dayfirst=True)

    if t.isna().mean() > 0.5:
        t2 = pd.to_datetime(s, errors="coerce")
        if t2.isna().mean() < t.isna().mean():
            t = t2

    return t


def read_series_from_files(files: list[Path], value_col: str) -> pd.DataFrame:
    parts = []

    for f in files:
        df = read_csv_flexible(f)
        tc = find_time_column(df)

        out = df[[tc, value_col]].copy()
        out["time"] = parse_time_series(out[tc]).dt.floor("h")
        out = out.dropna(subset=["time"])
        out = out[["time", value_col]]
        parts.append(out)

    out = pd.concat(parts, ignore_index=True)
    out = out.groupby("time", as_index=False)[value_col].mean()
    out = out.sort_values("time").reset_index(drop=True)

    return out


def build_year_df() -> pd.DataFrame:
    price = read_series_from_files(PRICE_FILES, "SPOTDE")
    solar = read_series_from_files(SOLAR_FILES, "PRODESOL")

    df = pd.merge(price, solar, on="time", how="inner").sort_values("time").reset_index(drop=True)

    start = pd.Timestamp("2024-01-01 00:00:00")
    end = pd.Timestamp("2025-01-01 00:00:00")

    df = df[(df["time"] >= start) & (df["time"] < end)].copy().reset_index(drop=True)

    peak = float(df["PRODESOL"].max())
    df["PRODESOL_1MW"] = df["PRODESOL"] * (SOLAR_PEAK_MW / peak)

    return df


def descriptive_stats(series: pd.Series) -> dict:
    return {
        "Antall observasjoner": series.count(),
        "Gjennomsnitt": series.mean(),
        "Median": series.median(),
        "Minimum": series.min(),
        "Maksimum": series.max(),
        "Standardavvik": series.std(),
        "25-persentil": series.quantile(0.25),
        "75-persentil": series.quantile(0.75),
    }


def main():
    df = build_year_df()

    stats = pd.DataFrame({
        "Spotpris (EUR/MWh)": descriptive_stats(df["SPOTDE"]),
        "Solproduksjon (MWh/time, 1 MW peak)": descriptive_stats(df["PRODESOL_1MW"]),
    })

    print("\nDescriptive statistics for 2024:\n")
    print(stats.round(2))

    out_path = OUT_DIR / "descriptive_statistics_2024.csv"
    stats.round(2).to_csv(out_path, encoding="utf-8-sig")

    print(f"\nLagret til: {out_path}")


if __name__ == "__main__":
    main()