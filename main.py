from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


TAXON_ID = 125612              # WoRMS AphiaID (Tetraodontidae). Swap for a bigger clade
TAXON_NAME = "Tetraodontidae"  # to get a larger dataset.
MAX_RECORDS = 200_000          # records to request from OBIS
BIN_WIDTH = 5                  # degrees of latitude per band
MIN_RECORDS_PER_BIN = 100      # bands with fewer records are excluded (too little data)
N_RAREFY_REPS = 200            # subsampling repetitions per band
SEED = 42
CACHE = Path(f"{TAXON_NAME}_obis_raw.csv")
STYLE = "rose-pine-matplotlib/themes/rose-pine-moon.mplstyle"

try:
    plt.style.use(STYLE)
except OSError:
    pass  # style file not found: fall back to matplotlib defaults

LOVE, FOAM, GOLD, IRIS = "#eb6f92", "#9ccfd8", "#f6c177", "#c4a7e7"



def fetch_obis(taxon_id: int, max_records: int) -> pd.DataFrame:
    """Download occurrence records once, then reuse the cached CSV."""
    if CACHE.exists():
        print(f"Loading cached records from {CACHE}")
        return pd.read_csv(CACHE, low_memory=False)

    from pyobis import occurrences

    print(f"Requesting up to {max_records:,} records for taxon {taxon_id} from OBIS ...")
    df = occurrences.search(taxonid=taxon_id, size=max_records).execute()
    df.to_csv(CACHE, index=False)
    return df



def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map differently-cased OBIS column names onto the ones used below."""
    wanted = {
        "decimallatitude": "decimalLatitude",
        "decimallongitude": "decimalLongitude",
        "species": "species",
        "datasetid": "datasetID",
        "eventdate": "eventDate",
    }
    df = df.rename(columns={c: wanted[c.lower()] for c in df.columns if c.lower() in wanted})

    if "species" not in df.columns:
        for candidate in ["scientificName", "scientificname", "taxonName", "taxonname"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "species"})
                break
    return df


def clean(df: pd.DataFrame):
    """Return the cleaned frame plus a log of how many records each step removed."""
    log = {"fetched": len(df)}
    df = normalise_columns(df).copy()

    for col in ("decimalLatitude", "decimalLongitude"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Missing coordinates or species name
    df = df.dropna(subset=["decimalLatitude", "decimalLongitude", "species"])
    df["species"] = df["species"].astype(str).str.strip()
    df = df[df["species"] != ""]
    log["after_missing_removed"] = len(df)

    # Impossible coordinates, plus the (0, 0) "null island" default
    ok = df["decimalLatitude"].between(-90, 90) & df["decimalLongitude"].between(-180, 180)
    ok &= ~((df["decimalLatitude"] == 0) & (df["decimalLongitude"] == 0))
    df = df[ok]
    log["after_bad_coords_removed"] = len(df)

    # Absence records and records OBIS itself flagged as dropped
    for flag in ("absence", "dropped"):
        if flag in df.columns:
            df = df[df[flag].astype(str).str.lower() != "true"]
    log["after_absence_removed"] = len(df)

    # Exact duplicates (same species, place, date and dataset)
    key = [c for c in ["species", "decimalLatitude", "decimalLongitude", "eventDate", "datasetID"]
           if c in df.columns]
    df = df.drop_duplicates(subset=key)
    log["after_dedup"] = len(df)

    return df.reset_index(drop=True), log


def add_lat_bins(df: pd.DataFrame) -> pd.DataFrame:
    n_bins = int(180 / BIN_WIDTH)
    idx = np.floor((df["decimalLatitude"] + 90) / BIN_WIDTH).clip(upper=n_bins - 1)
    df = df.copy()
    df["lat_bin"] = -90 + idx * BIN_WIDTH + BIN_WIDTH / 2  # band midpoint
    return df


def rarefy_by_bin(df: pd.DataFrame, rng: np.random.Generator):
    """
    Subsample every retained band to the same number of records, repeat N times,
    and average the species count. Records (not individuals) are the sampling unit.
    """
    counts = df.groupby("lat_bin").size()
    keep = counts[counts >= MIN_RECORDS_PER_BIN].index
    dropped = counts[counts < MIN_RECORDS_PER_BIN]
    if len(keep) < 5:
        raise RuntimeError(
            f"Only {len(keep)} latitude bands have >= {MIN_RECORDS_PER_BIN} records; "
            "lower MIN_RECORDS_PER_BIN or use a broader taxon."
        )
    n_target = int(counts[keep].min())

    rows = []
    for lat in sorted(keep):
        species = df.loc[df["lat_bin"] == lat, "species"]
        codes = pd.factorize(species)[0]
        draws = [
            len(np.unique(rng.choice(codes, size=n_target, replace=False)))
            for _ in range(N_RAREFY_REPS)
        ]
        rows.append({
            "lat": lat,
            "n_records": len(codes),
            "S_raw": int(species.nunique()),
            "S_rarefied": float(np.mean(draws)),
            "S_rarefied_sd": float(np.std(draws, ddof=1)),
        })
    return pd.DataFrame(rows), n_target, dropped


def aicc(rss: float, n: int, k: int) -> float:
    """Least-squares AIC with the small-sample correction."""
    rss = max(rss, 1e-12)
    aic = n * np.log(rss / n) + 2 * k
    return aic + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf)


def fit_models(x: np.ndarray, y: np.ndarray) -> dict:
    n = len(x)
    models = {}

    def register(name, k, predict):
        rss = float(np.sum((y - predict(x)) ** 2))
        models[name] = {"k": k, "predict": predict, "rss": rss, "aicc": aicc(rss, n, k)}

    # Linear
    p1 = np.polyfit(x, y, 1)
    register("Linear", 2, lambda t, p=p1: np.polyval(p, t))

    # Quadratic
    p2 = np.polyfit(x, y, 2)
    register("Quadratic", 3, lambda t, p=p2: np.polyval(p, t))

    # Gaussian (bell curve + baseline)
    def gauss(t, A, mu, sigma, C):
        return A * np.exp(-0.5 * ((t - mu) / sigma) ** 2) + C

    try:
        p0 = [np.ptp(y), x[np.argmax(y)], 30.0, y.min()]
        pg, _ = curve_fit(
            gauss, x, y, p0=p0, maxfev=20_000,
            bounds=([0, -90, 1, -np.inf], [np.inf, 90, 200, np.inf]),
        )
        register("Gaussian", 4, lambda t, p=pg: gauss(t, *p))
    except (RuntimeError, ValueError) as err:
        print(f"Gaussian fit failed ({err}); skipping.")

    # GAM (penalised spline; effective degrees of freedom used as k)
    if n >= 10:
        try:
            from pygam import LinearGAM, s

            gam = LinearGAM(s(0, n_splines=min(8, n - 2))).gridsearch(
                x[:, None], y, progress=False
            )
            edof = float(gam.statistics_["edof"])
            register("GAM", edof, lambda t, g=gam: g.predict(np.asarray(t)[:, None]))
        except Exception as err:  # pygam missing or incompatible with numpy/scipy version
            print(f"GAM skipped ({type(err).__name__}: {err}). Install pygam to include it.")

    # Model comparison
    best_aicc = min(m["aicc"] for m in models.values())
    for m in models.values():
        m["delta"] = m["aicc"] - best_aicc
    total = sum(np.exp(-0.5 * m["delta"]) for m in models.values())
    for m in models.values():
        m["weight"] = float(np.exp(-0.5 * m["delta"]) / total)
    return models


def describe_gradient(x, predict):
    """Peak latitude and average decline per degree on each side of the peak."""
    grid = np.linspace(x.min(), x.max(), 2000)
    yg = predict(grid)
    i = int(np.argmax(yg))
    peak, peak_val = float(grid[i]), float(yg[i])
    at_edge = i in (0, len(grid) - 1)

    south = (peak_val - yg[0]) / (peak - grid[0]) if peak > grid[0] else np.nan
    north = (peak_val - yg[-1]) / (grid[-1] - peak) if peak < grid[-1] else np.nan
    return {"peak_lat": peak, "peak_S": peak_val, "peak_at_edge": at_edge,
            "decline_per_deg_south": south, "decline_per_deg_north": north}


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def make_figure(rich, models, best_name, x, y, residuals, n_target):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{TAXON_NAME} latitudinal diversity gradient (OBIS)", fontsize=13, fontweight="bold")
    width = BIN_WIDTH * 0.85

    # (a) Why rarefy: raw vs effort-corrected richness
    ax = axes[0]
    ax.bar(rich["lat"], rich["S_raw"], width=width, color=FOAM, alpha=0.45, label="Raw richness")
    ax.errorbar(rich["lat"], rich["S_rarefied"], yerr=rich["S_rarefied_sd"],
                fmt="o", color=GOLD, capsize=3, label=f"Rarefied to {n_target} records")
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Species richness")
    ax.set_title("Raw vs rarefied richness")
    ax.legend(fontsize=9)

    # (b) Model fits
    ax = axes[1]
    ax.scatter(x, y, s=45, color=GOLD, zorder=5, label="Rarefied richness")
    grid = np.linspace(x.min(), x.max(), 500)
    colours = {"Linear": FOAM, "Quadratic": IRIS, "Gaussian": LOVE, "GAM": "#9ccfd8"}
    for name, m in models.items():
        is_best = name == best_name
        ax.plot(grid, m["predict"](grid), color=colours.get(name, None),
                linewidth=3 if is_best else 1.5, linestyle="-" if is_best else "--",
                label=f"{name}  (ΔAICc {m['delta']:.1f}){' *best' if is_best else ''}")
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Rarefied species richness")
    ax.set_title("Model comparison (AICc)")
    ax.legend(fontsize=8)

    # (c) Residuals of the best model
    ax = axes[2]
    ax.bar(x, residuals, width=width,
           color=[LOVE if r < 0 else FOAM for r in residuals], alpha=0.85)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Residual (obs − pred)")
    ax.set_title(f"Residuals: {best_name}")

    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)

    raw = fetch_obis(TAXON_ID, MAX_RECORDS)
    df, log = clean(raw)
    df = add_lat_bins(df)

    print("\n--- Cleaning log ---")
    prev = None
    for step, n in log.items():
        print(f"{step:26s}{n:>10,}" + (f"   (-{prev - n:,})" if prev is not None else ""))
        prev = n
    print(f"Species in cleaned data: {df['species'].nunique():,}")

    rich, n_target, dropped = rarefy_by_bin(df, rng)
    print(f"\nBands kept: {len(rich)}  |  rarefied to {n_target} records per band")
    if len(dropped):
        print(f"Bands excluded (< {MIN_RECORDS_PER_BIN} records): {len(dropped)}")

    x = rich["lat"].to_numpy(float)
    y = rich["S_rarefied"].to_numpy(float)

    models = fit_models(x, y)
    best_name = min(models, key=lambda k: models[k]["aicc"])
    best = models[best_name]

    y_pred = best["predict"](x)
    residuals = y - y_pred
    r2 = 1 - np.sum(residuals ** 2) / np.sum((y - y.mean()) ** 2)
    grad = describe_gradient(x, best["predict"])

    table = pd.DataFrame({
        "model": list(models),
        "k": [m["k"] for m in models.values()],
        "RSS": [m["rss"] for m in models.values()],
        "AICc": [m["aicc"] for m in models.values()],
        "delta_AICc": [m["delta"] for m in models.values()],
        "akaike_weight": [m["weight"] for m in models.values()],
    }).sort_values("AICc")

    print("\n--- Model comparison ---")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\n--- Best model: {best_name} ---")
    print(f"R^2                      : {r2:.3f}")
    edge_note = "  (at edge of data: no interior peak)" if grad["peak_at_edge"] else ""
    print(f"Peak latitude            : {grad['peak_lat']:.1f}°{edge_note}")
    print(f"Peak richness            : {grad['peak_S']:.1f} species")
    print(f"Decline per degree, south: {grad['decline_per_deg_south']:.3f} species/°")
    print(f"Decline per degree, north: {grad['decline_per_deg_north']:.3f} species/°")

    fig = make_figure(rich, models, best_name, x, y, residuals, n_target)
    fig.savefig(f"{TAXON_NAME}_ldg_fit.png", dpi=150, bbox_inches="tight")

    rich["S_pred_best"] = y_pred
    rich["residual"] = residuals
    rich.to_csv(f"{TAXON_NAME}_ldg_bins.csv", index=False)
    table.to_csv(f"{TAXON_NAME}_ldg_model_comparison.csv", index=False)
    plt.show()


if __name__ == "__main__":
    main()
