"""Gemeinsame Evaluationsfunktionen (MASE, RMSE, ...) und das
standardisierte Ergebnis-Format für alle Modell-Notebooks.

Konvention: Jedes Modell-Notebook (02-04) erzeugt einen DataFrame im
Schema config.RESULTS_COLUMNS und speichert ihn mit save_predictions().
Notebook 05 lädt alles mit load_all_predictions() und vergleicht.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import config, plotting


# ---------------------------------------------------------------------------
# Metriken
# ---------------------------------------------------------------------------
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


def mase(y_true, y_pred, y_train) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler 2006).

    Skaliert den Prognosefehler mit dem In-Sample-Fehler der naiven
    1-Schritt-Prognose auf den TRAININGSdaten derselben Serie:
        MASE = MAE(Prognose) / MAE(naiv, Training)
    MASE < 1  => besser als die naive Prognose. Skalenfrei ->
    über Länder aggregierbar (unser Primärkriterium).
    """
    y_train = np.asarray(y_train, float)
    y_train = y_train[~np.isnan(y_train)]
    if len(y_train) < 2:
        return float("nan")
    scale = np.mean(np.abs(np.diff(y_train)))
    if scale == 0:
        return float("nan")
    return mae(y_true, y_pred) / scale


def pinball_loss(y_true, y_q, q: float) -> float:
    """Pinball-/Quantil-Loss für ein einzelnes Quantil q (z.B. 0.1, 0.9)."""
    y_true, y_q = np.asarray(y_true, float), np.asarray(y_q, float)
    m = ~(np.isnan(y_true) | np.isnan(y_q))
    if m.sum() == 0:
        return float("nan")
    diff = y_true[m] - y_q[m]
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def interval_coverage(y_true, lo, hi) -> float:
    """Anteil der Beobachtungen im Intervall [lo, hi] - Soll bei
    q10/q90: ~0.80. Misst die KALIBRIERUNG der Unsicherheit."""
    y_true = np.asarray(y_true, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = ~(np.isnan(y_true) | np.isnan(lo) | np.isnan(hi))
    if m.sum() == 0:
        return float("nan")
    return float(np.mean((y_true[m] >= lo[m]) & (y_true[m] <= hi[m])))


# ---------------------------------------------------------------------------
# Ergebnis-Dateien (Schnittstelle zwischen den Notebooks)
# ---------------------------------------------------------------------------
def save_predictions(preds: pd.DataFrame, model: str) -> None:
    """Validiert das Schema und speichert nach data/results/preds_<model>.csv."""
    preds = preds.copy()
    preds["model"] = model
    for col in ("q10", "q90"):
        if col not in preds.columns:
            preds[col] = np.nan

    missing = [c for c in config.RESULTS_COLUMNS if c not in preds.columns]
    if missing:
        raise ValueError(f"Ergebnis-Schema verletzt, es fehlen: {missing}")
    preds = preds[config.RESULTS_COLUMNS]

    if not preds["h"].between(1, config.HORIZON).all():
        raise ValueError(f"Spalte 'h' muss in 1..{config.HORIZON} liegen.")
    if preds[["country", "origin_year", "year"]].isna().any().any():
        raise ValueError("country/origin_year/year dürfen nicht fehlen.")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"preds_{model}.csv"
    preds.to_csv(path, index=False)
    print(f"Gespeichert: {config.rel(path)}  ({len(preds):,} Prognosen)")


def load_all_predictions() -> pd.DataFrame:
    """Lädt alle preds_*.csv aus data/results/ (für Notebook 05)."""
    files = sorted(config.RESULTS_DIR.glob("preds_*.csv"))
    if not files:
        raise FileNotFoundError(
            "Keine Ergebnis-Dateien in data/results/ - zuerst die "
            "Modell-Notebooks 02-04 ausführen.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_predictions(model: str) -> pd.DataFrame:
    """Lädt die Prognosen genau eines Modells aus data/results/preds_<model>.csv.

    Gegenstück zu save_predictions(). Praktisch, um in einem Modell-Notebook
    eine geteilte Referenz wie die kanonische Baseline preds_rw_drift.csv zu
    laden, statt sie erneut zu berechnen.
    """
    path = config.RESULTS_DIR / f"preds_{model}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} fehlt - das Modell '{model}' wurde noch nicht gespeichert.")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Weltaggregat: Prognosen speichern, laden, bottom-up aggregieren
# ---------------------------------------------------------------------------
def save_world_predictions(preds: pd.DataFrame, model: str) -> None:
    """Speichert Prognosen der WELTREIHE nach data/results/world/preds_<model>.csv.

    Bewusst ein eigener Ordner: load_all_predictions() sammelt
    data/results/preds_*.csv nicht rekursiv ein, deshalb kann eine Weltdatei
    den Ländervergleich nicht verfälschen. Schema und Prüfungen sind identisch
    zu save_predictions(), die Spalte 'country' trägt hier config.WORLD_ID.
    """
    preds = preds.copy()
    preds["model"] = model
    for col in ("q10", "q90"):
        if col not in preds.columns:
            preds[col] = np.nan

    missing = [c for c in config.RESULTS_COLUMNS if c not in preds.columns]
    if missing:
        raise ValueError(f"Ergebnis-Schema verletzt, es fehlen: {missing}")
    preds = preds[config.RESULTS_COLUMNS]

    if not preds["h"].between(1, config.HORIZON).all():
        raise ValueError(f"Spalte 'h' muss in 1..{config.HORIZON} liegen.")
    if (preds["country"] != config.WORLD_ID).any():
        raise ValueError(
            f"Weltprognosen brauchen country == '{config.WORLD_ID}' in JEDER Zeile.")
    if preds.duplicated(["origin_year", "year"]).any():
        raise ValueError("Je Origin und Jahr darf es nur eine Weltprognose geben.")

    config.WORLD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.WORLD_RESULTS_DIR / f"preds_{model}.csv"
    preds.to_csv(path, index=False)
    print(f"Gespeichert: {config.rel(path)}  ({len(preds):,} Weltprognosen)")


def load_world_predictions(model: str) -> pd.DataFrame:
    """Lädt die Weltprognosen genau eines Modells."""
    path = config.WORLD_RESULTS_DIR / f"preds_{model}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} fehlt - Weltprognosen für '{model}' wurden noch nicht "
            "gespeichert.")
    return pd.read_csv(path)


def load_all_world_predictions() -> pd.DataFrame:
    """Lädt alle Weltprognosen aus data/results/world/ (für Notebook 05)."""
    files = sorted(config.WORLD_RESULTS_DIR.glob("preds_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"Keine Weltprognosen in {config.rel(config.WORLD_RESULTS_DIR)} - "
            "zuerst ein Modell-Notebook mit Weltabschnitt ausführen (z.B. 04).")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def aggregate_bottom_up(preds: pd.DataFrame, gdp_levels: pd.DataFrame,
                        world_panel: pd.DataFrame,
                        model: str | None = None) -> pd.DataFrame:
    """Aggregiert LÄNDERprognosen BIP-gewichtet zu einer Weltprognose.

    Je Origin dient das BIP-Niveau des Origin-Jahres als Gewicht - zur
    Prognosezeit bekannt, also leakage-frei - und wird mit den prognostizierten
    Wachstumsraten fortgeschrieben::

        BIP(i, origin+h) = BIP(i, origin) * prod_k (1 + g(i, origin+k) / 100)

    Die Weltwachstumsrate ergibt sich aus der Summe dieser Länderpfade. Damit
    lässt sich aus JEDEM Modell mit Länderprognosen (rw_drift, sarimax, tft,
    chronos2) eine Weltprognose ableiten und dem direkten Top-down-Ansatz
    gegenüberstellen - die klassische Frage der Prognose-Reconciliation.

    q10/q90 bleiben bewusst leer: Quantile sind NICHT additiv. Die Summe der
    Länder-q10 wäre kein q10 der Weltprognose, sondern unterstellte perfekt
    korrelierte Fehler und überzeichnete das Intervall massiv.

    Parameters
    ----------
    preds : Länderprognosen im Schema config.RESULTS_COLUMNS. Enthält der
        Rahmen mehrere Modelle, wird jedes einzeln aggregiert.
    gdp_levels, world_panel : aus data_utils.build_world_panel().
    model : Name des Ergebnismodells. None -> "<modellname>_bottom_up" je
        Eingangsmodell.

    Returns
    -------
    DataFrame im Schema config.RESULTS_COLUMNS mit country == config.WORLD_ID.
    """
    truth = (world_panel[["year", config.TARGET]]
             .rename(columns={config.TARGET: "y_true"}))

    frames = []
    for source_model, per_model in preds.groupby("model"):
        name = model if model is not None else f"{source_model}_bottom_up"
        rows = []
        for origin, g in per_model.groupby("origin_year"):
            paths = g.pivot_table(index="country", columns="h", values="y_pred")
            if origin not in gdp_levels.index:
                raise ValueError(
                    f"Kein BIP-Niveau für das Origin-Jahr {origin} vorhanden.")
            base = gdp_levels.loc[origin].reindex(paths.index)
            # Nur Länder mit Gewicht UND lückenlosem Prognosepfad aggregieren.
            usable = base.notna() & paths.notna().all(axis=1)
            paths, base = paths[usable], base[usable]
            if paths.empty:
                continue
            world_path = ((1 + paths / 100).cumprod(axis=1)
                          .mul(base, axis=0).sum(axis=0))
            previous = base.sum()
            for h in sorted(world_path.index):
                rows.append({
                    "model": name, "country": config.WORLD_ID,
                    "origin_year": int(origin), "year": int(origin + h),
                    "h": int(h),
                    "y_pred": (world_path[h] / previous - 1) * 100,
                    "q10": np.nan, "q90": np.nan,
                })
                previous = world_path[h]
        if rows:
            frames.append(pd.DataFrame(rows).merge(truth, on="year", how="left"))

    if not frames:
        raise ValueError("Keine aggregierbaren Länderprognosen gefunden.")
    return pd.concat(frames, ignore_index=True)[config.RESULTS_COLUMNS]


# ---------------------------------------------------------------------------
# Kanonische Baseline (von Notebook 01 erzeugt, von allen Modellen genutzt)
# ---------------------------------------------------------------------------
def rw_drift_forecast(panel: pd.DataFrame,
                      origins: list[int] | None = None,
                      horizon: int = config.HORIZON,
                      model: str = "rw_drift") -> pd.DataFrame:
    """Kanonische Random-Walk-mit-Drift-Baseline des Projekts.

    Je Land und Backtest-Origin gilt
        y_hat(origin + h) = y(origin) + h * drift,
    wobei drift die mittlere Jahresveränderung von config.TARGET bis
    einschließlich origin ist. Reine Punktprognose, q10/q90 bleiben leer.

    Diese Funktion wird von Notebook 01 aufgerufen und über
    save_predictions(preds, "rw_drift") als data/results/preds_rw_drift.csv
    geteilt, sodass alle Modell-Notebooks dieselbe Baseline laden können,
    statt sie erneut zu berechnen.
    """
    origins = list(config.ALL_ORIGINS if origins is None else origins)
    rows = []
    for origin in origins:
        for country, g in panel.groupby("country"):
            hist = g.loc[g["year"] <= origin, config.TARGET].dropna()
            if len(hist) < 2:
                continue
            last_value = hist.iloc[-1]
            drift = hist.diff().mean()
            truth = g.set_index("year")[config.TARGET]
            for h in range(1, horizon + 1):
                year = origin + h
                rows.append({
                    "model": model, "country": country,
                    "origin_year": origin, "year": year, "h": h,
                    "y_true": truth.get(year, np.nan),
                    "y_pred": last_value + h * drift,
                    "q10": np.nan, "q90": np.nan,
                })
    return pd.DataFrame(rows)[config.RESULTS_COLUMNS]


def mean_forecast(panel: pd.DataFrame,
                  origins: list[int] | None = None,
                  horizon: int = config.HORIZON,
                  model: str = "mean") -> pd.DataFrame:
    """Klimatologie-Baseline: der expandierende historische Mittelwert.

    Je Serie und Origin gilt schlicht
        y_hat(origin + h) = Mittelwert von config.TARGET bis einschliesslich origin,
    konstant über alle h. Reine Punktprognose, q10/q90 bleiben leer.

    Diese Baseline ist der ANGEMESSENE Bezugspunkt für mittelwertrückkehrende
    Reihen. Der Random Walk mit Drift aus rw_drift_forecast() schreibt den
    letzten Wert samt Trend fort und scheitert deshalb an Reihen, die um einen
    stabilen Mittelwert schwanken, besonders direkt nach einem Ausreisserjahr.
    Auf dem Weltaggregat ist er dadurch eine zu schwache Messlatte, und ein
    Modell kann ihn deutlich schlagen, ohne mehr zu können, als den
    langfristigen Durchschnitt vorherzusagen. Für einen belastbaren
    Skill-Nachweis muss ein Modell BEIDE Baselines schlagen.
    """
    origins = list(config.ALL_ORIGINS if origins is None else origins)
    rows = []
    for origin in origins:
        for country, g in panel.groupby("country"):
            hist = g.loc[g["year"] <= origin, config.TARGET].dropna()
            if hist.empty:
                continue
            level = float(hist.mean())
            truth = g.set_index("year")[config.TARGET]
            for h in range(1, horizon + 1):
                year = origin + h
                rows.append({
                    "model": model, "country": country,
                    "origin_year": origin, "year": year, "h": h,
                    "y_true": truth.get(year, np.nan),
                    "y_pred": level,
                    "q10": np.nan, "q90": np.nan,
                })
    return pd.DataFrame(rows)[config.RESULTS_COLUMNS]


def clustered_forecast_test(loss_a, loss_b, clusters) -> dict[str, float]:
    """Gepaarter Vergleich zweier Verlustreihen mit Clusterung.

    paired_forecast_test() behandelt jede Einheit als unabhängig. In einem
    Backtest-Panel ist das zu optimistisch: Alle Länder eines Folds teilen
    dieselben Testjahre und dasselbe Makroumfeld, ihre Verlustdifferenzen sind
    daher nicht unabhängig. Diese Funktion mittelt die Differenzen ZUERST je
    Cluster (typischerweise je Origin) und testet dann die Clustermittelwerte
    gegen null. Das ist der konservative Gegentest: Bleibt ein Effekt auch so
    signifikant, hängt er nicht an der Unabhängigkeitsannahme.

    Die Freiheitsgrade sinken dabei drastisch, bei fünf Origins auf vier. Ein
    nicht signifikantes Ergebnis heisst hier also NICHT "kein Effekt", sondern
    "mit dieser Zahl an Clustern nicht nachweisbar".
    """
    a, b = np.asarray(loss_a, float), np.asarray(loss_b, float)
    cl = np.asarray(clusters)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b, cl = a[mask], b[mask], cl[mask]
    diff = a - b

    means = np.array([diff[cl == c].mean() for c in np.unique(cl)])
    out = {"n_clusters": int(len(means)),
           "mean_diff": float(diff.mean()) if len(diff) else float("nan"),
           "cluster_means": means}
    if len(means) < 3 or np.allclose(means, 0):
        out.update({"ttest_stat": float("nan"), "ttest_p": float("nan")})
        return out
    stat, p = stats.ttest_1samp(means, 0.0)
    out.update({"ttest_stat": float(stat), "ttest_p": float(p)})
    return out


def summarize(preds: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Vergleichstabelle je Modell: MASE (primär), RMSE, MAE,
    Pinball-Loss und 80%-Intervall-Coverage (falls Quantile vorhanden).

    MASE wird je (Land, Origin) berechnet und dann gemittelt, damit
    kein Land das Ergebnis dominiert.
    """
    rows = []
    for model, g in preds.groupby("model"):
        mases = []
        for (country, origin), gg in g.groupby(["country", "origin_year"]):
            y_train = panel.loc[(panel["country"] == country)
                                & (panel["year"] <= origin), config.TARGET]
            mases.append(mase(gg["y_true"], gg["y_pred"], y_train))
        valid_mases = np.asarray(mases, dtype=float)
        rows.append({
            "model": model,
            "MASE": (float(np.nanmean(valid_mases))
                     if np.isfinite(valid_mases).any() else float("nan")),
            "RMSE": rmse(g["y_true"], g["y_pred"]),
            "MAE": mae(g["y_true"], g["y_pred"]),
            "pinball_q10": pinball_loss(g["y_true"], g["q10"], 0.1),
            "pinball_q90": pinball_loss(g["y_true"], g["q90"], 0.9),
            "coverage_80": interval_coverage(g["y_true"], g["q10"], g["q90"]),
            "n_forecasts": len(g),
        })
    return (pd.DataFrame(rows)
            .sort_values("MASE")
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Residualdiagnostik (Out-of-sample-Fehler aus den Backtest-Prognosen)
# ---------------------------------------------------------------------------
DISPLAY_QUANTILES = (0.01, 0.99)


def _prepare_residuals(preds: pd.DataFrame,
                       countries: list[str] | None = None) -> pd.DataFrame:
    """Validiert Prognosen und ergänzt ``residual = y_true - y_pred``."""
    required = {"country", "year", "h", "y_true", "y_pred"}
    missing = required.difference(preds.columns)
    if missing:
        raise ValueError(
            f"Residualdiagnostik braucht die Spalten: {sorted(missing)}")

    residuals = preds.dropna(subset=["y_true", "y_pred"]).copy()
    if countries is not None:
        residuals = residuals[residuals["country"].isin(countries)].copy()
    if residuals.empty:
        scope = ", ".join(countries) if countries else "alle Länder"
        raise ValueError(f"Keine gültigen Residuen für {scope} vorhanden.")

    residuals["residual"] = residuals["y_true"] - residuals["y_pred"]
    return residuals.sort_values(["country", "year"])


def _scope_label(countries: list[str] | None) -> str:
    return "alle Länder" if countries is None else ", ".join(countries)


def _robust_limits(values: pd.Series,
                   quantiles: tuple[float, float] = DISPLAY_QUANTILES) -> tuple[float, float]:
    """Robuste Anzeigegrenzen, damit einzelne Ausreißer den Plot nicht stauchen."""
    low, high = values.quantile(quantiles).astype(float)
    if low == high:
        padding = max(abs(low) * 0.1, 1.0)
        return low - padding, high + padding
    return low, high


def _despine(ax: plt.Axes) -> None:
    # Projektstil zeigt inzwischen den vollstaendigen Rahmen (alle vier Spines).
    # Fuer einen einheitlichen Look bleiben top/right hier sichtbar; die Funktion
    # bleibt als gemeinsamer Aufrufpunkt erhalten, falls der Stil erneut wechselt.
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(plotting.BASELINE)


def _soften_grid(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(True, axis=axis, linestyle="--", linewidth=0.8, alpha=0.7)
    if axis != "x":
        ax.grid(False, axis="x")


def _plot_fold_start_markers(ax: plt.Axes, residuals: pd.DataFrame) -> None:
    """Markiert den Start jedes neuen Forecast-/Trainingsblocks im Jahresplot."""
    if "origin_year" not in residuals.columns:
        return

    starts = (residuals.groupby("origin_year")["year"]
              .min()
              .dropna()
              .sort_values()
              .unique())
    if len(starts) == 0:
        return

    y_min, y_max = ax.get_ylim()
    marker_y = y_max - 0.06 * (y_max - y_min)

    for index, start_year in enumerate(starts):
        label = "Neues Trainingsfenster" if index == 0 else None
        ax.axvline(start_year, color="#9aa3af", linewidth=1.0,
                   linestyle=(0, (2, 3)), alpha=0.85, zorder=0,
                   label=label)
        ax.scatter(start_year, marker_y, marker="v", s=42,
                   color="#9aa3af", edgecolors=plotting.SURFACE,
                   linewidths=0.7, zorder=4)


def _prepare_horizon_matrix(preds: pd.DataFrame,
                            countries: list[str] | None = None) -> pd.DataFrame:
    """Residualmatrix je Forecast-Block: Zeilen=(Land, Origin), Spalten=Horizont."""
    residuals = _prepare_residuals(preds, countries)
    matrix = (residuals
              .pivot_table(index=["country", "origin_year"],
                           columns="h", values="residual", aggfunc="first")
              .sort_index(axis=1))
    matrix = matrix.reindex(columns=range(1, config.HORIZON + 1))
    if matrix.empty:
        raise ValueError(
            "Keine vollständigen Forecast-Blöcke für Horizontdiagnostik.")
    return matrix


def _block_lag_correlation(matrix: pd.DataFrame, lag: int) -> float:
    """Korrelation der Residuen zwischen h und h+lag über alle Forecast-Blöcke."""
    if lag == 0:
        return 1.0

    values = matrix.to_numpy(dtype=float)
    left = values[:, :-lag].ravel()
    right = values[:, lag:].ravel()
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return float("nan")
    left = left[mask]
    right = right[mask]
    if np.isclose(left.std(ddof=0), 0) or np.isclose(right.std(ddof=0), 0):
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def plot_residual_histogram(preds: pd.DataFrame,
                            countries: list[str] | None = None,
                            display_quantiles: tuple[float,
                                                     float] = DISPLAY_QUANTILES,
                            ax: plt.Axes | None = None):
    """Histogramm der Residuen im robusten Anzeigeintervall."""
    residuals = _prepare_residuals(preds, countries)
    fig, ax = (plt.subplots(figsize=(5, 3.1)) if ax is None
               else (ax.figure, ax))
    values = residuals["residual"]
    low, high = _robust_limits(values, display_quantiles)
    displayed = values[values.between(low, high)]

    ax.hist(displayed, bins="auto", density=True,
            color=plotting.SEQ_BLUES[1], alpha=0.9,
            edgecolor=plotting.SURFACE, linewidth=1.0)
    if len(displayed) >= 2 and displayed.std(ddof=1) > 0:
        kde = stats.gaussian_kde(displayed)
        x_grid = np.linspace(low, high, 250)
        density = kde(x_grid)
        ax.plot(x_grid, density, color=plotting.SEQ_BLUES[4], linewidth=2.0,
                label="KDE")
    ax.axvline(0, color=plotting.BASELINE, linewidth=1)
    ax.axvline(values.median(), color=plotting.INK, linewidth=1.3,
               linestyle="--", label=f"Median {values.median():+.2f}")
    ax.axvline(values.mean(), color=plotting.SERIES_COLORS[1], linewidth=1.2,
               linestyle=":", label=f"Mittel {values.mean():+.2f}")
    ax.set_xlim(low, high)
    ax.set_xlabel("Fehler")
    ax.set_ylabel("Dichte")
    ax.set_title("Residualverteilung (Anzeige P1-P99)", loc="left", fontsize=9,
                 color=plotting.INK_SECONDARY)
    _soften_grid(ax)
    _despine(ax)
    ax.legend(frameon=False, fontsize=8)
    return fig, ax


def plot_residual_qq(preds: pd.DataFrame,
                     countries: list[str] | None = None,
                     ax: plt.Axes | None = None):
    """Q-Q-Plot der Out-of-sample-Residuals gegen die Normalverteilung."""
    residuals = _prepare_residuals(preds, countries)
    fig, ax = (plt.subplots(figsize=(5, 3.1)) if ax is None
               else (ax.figure, ax))
    theoretical, ordered = stats.probplot(residuals["residual"], dist="norm",
                                          fit=False)
    slope, intercept, _ = stats.probplot(residuals["residual"], dist="norm",
                                         fit=True)[1]
    ax.scatter(theoretical, ordered, color=plotting.SEQ_BLUES[4],
               alpha=0.85, s=20, edgecolors="none")
    line_x = np.array([min(theoretical), max(theoretical)])
    ax.plot(line_x, intercept + slope * line_x, color=plotting.INK,
            linewidth=1.5, linestyle="--")
    ax.set_xlabel("Theoretische Quantile")
    ax.set_ylabel("Residual-Quantile")
    ax.set_title("Q-Q-Plot", loc="left", fontsize=9,
                 color=plotting.INK_SECONDARY)
    _soften_grid(ax, axis="both")
    _despine(ax)
    return fig, ax


def plot_residuals_vs_fitted(preds: pd.DataFrame,
                             countries: list[str] | None = None,
                             ax: plt.Axes | None = None,
                             cax: plt.Axes | None = None):
    """Residuals gegen Punktprognosen zur Prüfung von Bias und Streuung."""
    residuals = _prepare_residuals(preds, countries)
    fig, ax = (plt.subplots(figsize=(7.5, 3.1)) if ax is None
               else (ax.figure, ax))
    values = residuals[["y_pred", "residual"]]
    display_quantiles = (0.10, 0.90)
    pred_low, pred_high = _robust_limits(values["y_pred"], display_quantiles)
    residual_low, residual_high = _robust_limits(
        values["residual"], display_quantiles)
    displayed = values.loc[
        values["y_pred"].between(pred_low, pred_high)
        & values["residual"].between(residual_low, residual_high)
    ]

    hexbin = ax.hexbin(displayed["y_pred"], displayed["residual"],
                       gridsize=34, mincnt=1, cmap="Blues", linewidths=0)
    if cax is None:
        fig.colorbar(hexbin, ax=ax, pad=0.01, fraction=0.03,
                     label="Anzahl Prognosen")
    else:
        colorbar = fig.colorbar(hexbin, cax=cax, label="Anzahl Prognosen")
        colorbar.outline.set_linewidth(0.6)
        cax.yaxis.set_ticks_position("right")
        cax.yaxis.set_label_position("right")
    ax.axhline(0, color=plotting.BASELINE, linewidth=1)

    fitted_bins = pd.qcut(displayed["y_pred"], q=12, duplicates="drop")
    trend = displayed.groupby(fitted_bins, observed=True).agg(
        y_pred=("y_pred", "mean"), residual=("residual", "mean"))
    ax.plot(trend["y_pred"], trend["residual"], color=plotting.INK,
            linewidth=1.8, marker="o", markersize=3.5,
            label="Mittlerer Fehler je Prognosebereich")

    ax.set_xlabel("Punktprognose")
    ax.set_ylabel("Fehler")
    ax.set_title("Residuals gegen Punktprognose (Anzeige P10-P90)", loc="left", fontsize=9,
                 color=plotting.INK_SECONDARY)
    _soften_grid(ax)
    _despine(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return fig, ax


def plot_residual_acf(preds: pd.DataFrame,
                      countries: list[str] | None = None,
                      max_lag: int = config.HORIZON - 1,
                      ax: plt.Axes | None = None):
    """Autokorrelation der Residuen innerhalb der Forecast-Horizonte."""
    horizon_matrix = _prepare_horizon_matrix(preds, countries)
    fig, ax = (plt.subplots(figsize=(6.4, 3.1)) if ax is None
               else (ax.figure, ax))

    max_lag = min(max_lag, config.HORIZON - 1)
    lags = np.arange(max_lag + 1)

    if countries is None:
        corr_rows = {}
        for country in sorted(horizon_matrix.index.get_level_values("country").unique()):
            country_matrix = horizon_matrix.loc[country]
            if isinstance(country_matrix, pd.Series):
                country_matrix = country_matrix.to_frame().T
            corrs = [_block_lag_correlation(
                country_matrix, lag) for lag in lags]
            corr_rows[country] = corrs

        corr_frame = pd.DataFrame.from_dict(
            corr_rows, orient="index", columns=lags)
    else:
        corrs = [_block_lag_correlation(horizon_matrix, lag) for lag in lags]
        corr_frame = pd.DataFrame(
            [corrs], index=[_scope_label(countries)], columns=lags)

    if corr_frame.dropna(how="all").empty:
        raise ValueError("Zu wenige Residuen für eine ACF-Diagnostik.")

    if countries is None:
        median = corr_frame.median(axis=0, skipna=True)
        q25 = corr_frame.quantile(0.25, axis=0)
        q75 = corr_frame.quantile(0.75, axis=0)
        ax.fill_between(lags, q25, q75, color="#d9dde3",
                        alpha=1.0, linewidth=0, label="IQR über Länder")
        ax.vlines(lags, 0, median, color=plotting.SEQ_BLUES[4], linewidth=2.0,
                  alpha=0.9, label="Median")
        ax.scatter(lags, median, color=plotting.SEQ_BLUES[4], s=28, zorder=3)
        ax.legend(frameon=False, fontsize=8)
    else:
        series = corr_frame.iloc[0]
        ax.vlines(lags, 0, series, color=plotting.SEQ_BLUES[4], linewidth=2.0,
                  alpha=0.9)
        ax.scatter(lags, series, color=plotting.SEQ_BLUES[4], s=28, zorder=3,
                   label=_scope_label(countries))
        ax.legend(frameon=False, fontsize=8)

    ax.axhline(0, color=plotting.BASELINE, linewidth=1)
    ax.set_xlim(-0.5, max_lag + 0.5)
    ax.set_ylim(-1, 1)
    ax.set_xticks(lags)
    ax.set_xlabel("Lag innerhalb des Forecast-Horizonts")
    ax.set_ylabel("Korrelation")
    ax.set_title("Fehlerkorrelation nach Horizontabstand", loc="left", fontsize=9,
                 color=plotting.INK_SECONDARY)
    _soften_grid(ax)
    _despine(ax)
    return fig, ax


def plot_residuals_by_horizon(preds: pd.DataFrame,
                              countries: list[str] | None = None,
                              display_quantiles: tuple[float,
                                                       float] = DISPLAY_QUANTILES,
                              ax: plt.Axes | None = None,
                              positions=None,
                              box_color: str | None = None,
                              box_width: float = 0.45,
                              label: str | None = None):
    """Residualverteilung nach Horizon mit robustem Anzeigeintervall."""
    residuals = _prepare_residuals(preds, countries)
    fig, ax = (plt.subplots(figsize=(6.4, 3.1)) if ax is None
               else (ax.figure, ax))
    horizons = sorted(residuals["h"].unique())
    data = [residuals.loc[residuals["h"] == horizon, "residual"]
            for horizon in horizons]
    box_positions = (np.asarray(horizons, dtype=float)
                     if positions is None else np.asarray(positions, dtype=float))
    low, high = _robust_limits(residuals["residual"], display_quantiles)
    boxes = ax.boxplot(data, positions=box_positions,
                       tick_labels=horizons if positions is None else None,
                       patch_artist=True, widths=box_width, showfliers=False,
                       medianprops={"color": plotting.INK, "linewidth": 1.4},
                       whiskerprops={
                           "color": plotting.SEQ_BLUES[4], "linewidth": 1.1},
                       capprops={"color": plotting.SEQ_BLUES[4], "linewidth": 1.1})
    for box in boxes["boxes"]:
        box.set(facecolor=box_color or "#e9eef5", edgecolor=plotting.SEQ_BLUES[4],
                linewidth=1.2)
    if label is not None:
        boxes["boxes"][0].set_label(label)
    ax.axhline(0, color=plotting.BASELINE, linewidth=1)
    ax.set_ylim(low, high)
    ax.set_xlabel("Forecast-Horizont h")
    ax.set_ylabel("Fehler")
    ax.set_title("Residuals nach Horizont (Anzeige P1-P99)", loc="left", fontsize=9,
                 color=plotting.INK_SECONDARY)
    _soften_grid(ax)
    _despine(ax)
    return fig, ax


def plot_residuals_by_year(preds: pd.DataFrame,
                           countries: list[str] | None = None,
                           ax: plt.Axes | None = None):
    """Jährlicher mittlerer Residualfehler mit Interquartilsband."""
    residuals = _prepare_residuals(preds, countries)
    fig, ax = (plt.subplots(figsize=(7.5, 3.1)) if ax is None
               else (ax.figure, ax))
    yearly = residuals.groupby("year")["residual"]
    mean = yearly.mean()
    q25 = yearly.quantile(0.25)
    q75 = yearly.quantile(0.75)
    ax.fill_between(mean.index, q25, q75, color="#d9dde3",
                    alpha=1.0, linewidth=0, label="IQR")
    ax.vlines(mean.index, 0, mean.values, color=plotting.SEQ_BLUES[3],
              linewidth=2.2, alpha=0.9)
    ax.scatter(mean.index, mean.values, color=plotting.SEQ_BLUES[4],
               s=34, zorder=3, label="Mittelwert")
    ax.plot(mean.index, mean.values, color=plotting.SEQ_BLUES[4],
            linewidth=1.2, alpha=0.7)
    ax.axhline(0, color=plotting.BASELINE, linewidth=1)
    _plot_fold_start_markers(ax, residuals)
    ax.set_xlabel("Testjahr")
    ax.set_ylabel("Fehler")
    ax.set_title("Residuals nach Jahr", loc="left", fontsize=9,
                 color=plotting.INK_SECONDARY)
    _soften_grid(ax)
    _despine(ax)
    ax.legend(frameon=False, fontsize=8)
    return fig, ax


# ===========================================================================
# Statistische Signifikanz von Prognosevergleichen
# ===========================================================================
def diebold_mariano(loss_a, loss_b, h: int = 1) -> tuple[float, float]:
    """Führt den Diebold Mariano Test von 1995 durch, ergänzt um die
    Kleinstichprobenkorrektur nach Harvey, Leybourne und Newbold aus dem
    Jahr 1997.

    Der Test prüft für eine einzelne Verlustreihe, etwa eine Serie oder ein
    einzelnes Land, ob der Erwartungswert der Differenz aus loss_a und
    loss_b gleich 0 ist, was gleicher Prognosegüte entspricht, gegen die
    Alternative unterschiedlicher Güte. Der Parameter h steuert die
    Trunkierung nach Newey und West für die Schätzung der Varianz des
    Verlustdifferentials mit Berücksichtigung von Autokorrelation, denn
    Prognosefehler über mehrere Schritte sind in der Regel bis zum Abstand
    h minus 1 autokorreliert.

    Der klassische Diebold Mariano Test ist für eine einzelne, lange,
    näherungsweise stationäre Zeitreihe entworfen. Für einen Vergleich über
    viele Serien, etwa ein Länderpanel, ist er nicht unmittelbar geeignet,
    siehe paired_forecast_test für die dafür passende Alternative. Sinnvoll
    bleibt er, wenn er auf die Verlustreihe einer einzelnen Serie angewendet
    wird, etwa je Land, mit anschließender Aggregation über die Länder.

    Die Funktion gibt ein Tupel aus der Diebold Mariano Statistik und dem
    zugehörigen p Wert zurück. Der p Wert stammt aus der t Verteilung mit
    T minus 1 Freiheitsgraden gemäß der Kleinstichprobenkorrektur und nicht
    aus der Standardnormalverteilung.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[~np.isnan(d)]
    T = len(d)
    if T < max(3, h + 1):
        return float("nan"), float("nan")

    d_bar = d.mean()
    dd = d - d_bar
    gamma = [np.sum(dd[k:] * dd[:T - k]) / T for k in range(h)]
    var_d_bar = (gamma[0] + 2 * sum(gamma[1:])) / T
    if var_d_bar <= 0:
        return float("nan"), float("nan")

    dm_stat = d_bar / np.sqrt(var_d_bar)
    hln = np.sqrt(max((T + 1 - 2 * h + h * (h - 1) / T) / T, 0))
    dm_corrected = dm_stat * hln
    p_value = float(2 * (1 - stats.t.cdf(abs(dm_corrected), df=T - 1)))
    return float(dm_corrected), p_value


def paired_forecast_test(loss_a, loss_b) -> dict[str, float]:
    """Führt einen gepaarten Wilcoxon Vorzeichen Rang Test durch, ergänzt um
    einen t Test als Vergleich, über viele unabhängige Einheiten, etwa
    Kombinationen aus Land und Origin in einem Backtest Panel.

    Anders als der Diebold Mariano Test, der die zeitliche Abhängigkeit
    einer einzelnen Serie modelliert, behandelt dieser Test jede Einheit
    als näherungsweise unabhängige Beobachtung, was für einen Vergleich
    über ein Länderpanel die realistischere Annahme ist. Der Wilcoxon
    Test ist nichtparametrisch und robust gegenüber der typischerweise
    schiefen Verteilung von MASE Differenzen. Der t Test dient nur als
    Vergleich.

    loss_a und loss_b müssen paarweise auf dieselben Einheiten
    ausgerichtet sein, mit gleicher Reihenfolge beziehungsweise gleichem
    Index, zum Beispiel beide indiziert nach Land und Origin Jahr.
    """
    a, b = np.asarray(loss_a, float), np.asarray(loss_b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    diff = a - b

    if len(diff) < 3 or np.allclose(diff, 0):
        return {"n_pairs": int(len(diff)), "mean_diff": float(diff.mean()) if len(diff) else float("nan"),
                "median_diff": float("nan"), "wilcoxon_stat": float("nan"), "wilcoxon_p": float("nan"),
                "ttest_stat": float("nan"), "ttest_p": float("nan")}

    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(diff)
    ttest_stat, ttest_p = stats.ttest_rel(a, b)
    return {
        "n_pairs": int(len(diff)),
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "wilcoxon_stat": float(wilcoxon_stat), "wilcoxon_p": float(wilcoxon_p),
        "ttest_stat": float(ttest_stat), "ttest_p": float(ttest_p),
    }


def plot_residual_diagnostics(preds: pd.DataFrame,
                              countries: list[str] | None = None,
                              max_lag: int = config.HORIZON - 1):
    """Zeigt die kompakte Out-of-sample-Residualdiagnostik im 1+1+2+2-Layout.

    Parameters
    ----------
    preds:
        Prognosen im Standard-Ergebnisschema mit ``y_true`` und ``y_pred``.
    countries:
        Optionale ISO3-Codes für eine gefilterte Diagnoseansicht.
    max_lag:
        Höchster Lag im Horizont-ACF-Plot.
    """
    fig = plt.figure(figsize=(13.8, 11.0))
    grid = fig.add_gridspec(4, 2, height_ratios=[1.05, 1.05, 1, 1], hspace=0.48,
                            wspace=0.24)
    ax_year = fig.add_subplot(grid[0, :])
    ax_fitted = fig.add_subplot(grid[1, :])
    cax_fitted = ax_fitted.inset_axes([0.989, 0.02, 0.010, 0.96])
    ax_histogram = fig.add_subplot(grid[2, 0])
    ax_qq = fig.add_subplot(grid[2, 1])
    ax_acf = fig.add_subplot(grid[3, 0])
    ax_horizon = fig.add_subplot(grid[3, 1])
    axes = np.array([ax_year, ax_fitted, ax_histogram, ax_qq, ax_acf,
                     ax_horizon])

    plot_residuals_by_year(preds, countries=countries, ax=ax_year)
    plot_residuals_vs_fitted(preds, countries=countries, ax=ax_fitted,
                             cax=cax_fitted)
    plot_residual_histogram(preds, countries=countries, ax=ax_histogram)
    plot_residual_qq(preds, countries=countries, ax=ax_qq)
    plot_residual_acf(preds, countries=countries, max_lag=max_lag, ax=ax_acf)
    plot_residuals_by_horizon(preds, countries=countries, ax=ax_horizon)
    for ax in axes:
        ax.xaxis.labelpad = 4
        ax.yaxis.labelpad = 4
    fig.suptitle("Residualdiagnostik",
                 x=0.5, ha="center", color=plotting.INK)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.06, top=0.94,
                        hspace=0.50)
    return fig, axes
