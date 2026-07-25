"""Gemeinsame Plot-Funktionen für alle Notebooks.

Ein konsistenter, farbfehlsichtigkeits-geprüfter Stil für das ganze
Projekt: feste Farbzuordnung je Modell (Farbe folgt der Entität, nie
der Reihenfolge), eine Achse pro Plot, dezente Gitterlinien.
"""
from __future__ import annotations

import zlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# Palette (validierte Default-Palette, Light Mode)
# ---------------------------------------------------------------------------
SERIES_COLORS = [
    "#2a78d6",  # 1 blau
    "#1baf7a",  # 2 aqua
    "#eda100",  # 3 gelb
    "#008300",  # 4 grün
    "#4a3aa7",  # 5 violett
    "#e34948",  # 6 rot
    "#e87ba4",  # 7 magenta
    "#eb6834",  # 8 orange
]
INK = "#0b0b0b"        # Primärtext / Ist-Werte
INK_SECONDARY = "#52514e"
MUTED = "#898781"      # Achsen, Labels
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# Feste Farb-Zuordnung je Modell - in ALLEN Notebooks identisch:
MODEL_COLORS = {
    "actual": INK,
    "rw_drift": MUTED,
    "rw_drift_intern": MUTED,
    "arima": "#2a78d6",
    "tft": "#1baf7a",
    "chronos2": "#eda100",
    "aggregate": "#1baf7a",
    # Baselines und Weltaggregat (Notebook 04 Abschnitt 6, Notebook 05)
    "mean": BASELINE,
    "mean_world": BASELINE,
    "rw_drift_world": MUTED,
    "chronos2_world": "#eda100",
    "chronos2_world_ctx": "#eb6834",
    "chronos2_world_uni": "#4a3aa7",
    "chronos2_bottom_up": "#1baf7a",
}

# Sequentielle Blau-Stufen (hell -> dunkel) für Abdeckungs-Heatmap
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"]


def _fallback_color(name: str) -> str:
    """Stabile Ersatzfarbe für ein Modell, das nicht in MODEL_COLORS steht.

    Bewusst crc32 statt der eingebauten hash(): Pythons String-Hash ist seit
    3.3 je Prozess zufällig verwürfelt (PYTHONHASHSEED), ein Modell bekäme
    damit bei jedem Notebook-Lauf eine andere Farbe. crc32 ist deterministisch
    und hält die Zuordnung über Läufe und Rechner hinweg stabil.
    """
    return SERIES_COLORS[zlib.crc32(name.encode("utf-8")) % len(SERIES_COLORS)]


def apply_style() -> None:
    """Projektweiten Matplotlib-Stil setzen (einmal pro Notebook aufrufen)."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.grid": True,
        "axes.axisbelow": True,          # Gitterlinien immer unter allen Artists
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "font.family": "sans-serif",
        "axes.prop_cycle": plt.cycler(color=SERIES_COLORS),
        "lines.linewidth": 2.0,
        "figure.dpi": 200,
        # Legende standardmäßig mit halbtransparentem Rahmen über dem Gitter
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.facecolor": SURFACE,
        "legend.edgecolor": BASELINE,
    })


def add_legend(ax, **kwargs):
    """Legende mit halbtransparentem Rahmen, der über den Gitterlinien liegt.

    Vereinheitlicht alle Legenden im Projekt: ein dezenter Rahmen mit leicht
    durchscheinendem Hintergrund (framealpha), der die Gitterlinien dahinter
    überdeckt, und ein erhöhter zorder, damit die Legende sicher oben liegt.
    Zusätzliche kwargs (z. B. ncol, fontsize, loc) werden durchgereicht.
    """
    kwargs.setdefault("frameon", True)
    kwargs.setdefault("framealpha", 0.85)
    kwargs.setdefault("fontsize", 9)
    legend = ax.legend(**kwargs)
    if legend is not None:
        frame = legend.get_frame()
        frame.set_facecolor(SURFACE)
        frame.set_edgecolor(BASELINE)
        legend.set_zorder(6)             # ueber den Gitterlinien (zorder ~2)
    return legend


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_country_series(panel: pd.DataFrame, countries: list[str],
                        column: str = config.TARGET,
                        title: str | None = None):
    """Zeitreihen ausgewählter Länder (max. 8 wegen fester Palette)."""
    if len(countries) > len(SERIES_COLORS):
        raise ValueError(f"Maximal {len(SERIES_COLORS)} Länder pro Plot - "
                         "sonst kleine Multiples verwenden.")
    # Bewusst ohne Beschriftung am Linienende. Bei mehreren Laendern
    # ueberlappen die Labels am rechten Rand, die Legende genuegt.
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, c in enumerate(countries):
        d = panel[panel["country"] == c]
        ax.plot(d["year"], d[column], color=SERIES_COLORS[i], label=c)
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_xlabel("Jahr")
    ax.set_ylabel(column)
    ax.set_title(title or f"{column} - ausgewählte Länder",
                 loc="left", color=INK)
    add_legend(ax, ncol=min(len(countries), 4))
    fig.tight_layout()
    return fig, ax


def plot_coverage(panel: pd.DataFrame, column: str = config.TARGET,
                  title: str | None = None):
    """Datenabdeckung: Anzahl Länder mit gültigem Wert je Jahr."""
    counts = (panel.dropna(subset=[column])
              .groupby("year")["country"].nunique())
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(counts.index, counts.values, color=SEQ_BLUES[3], width=0.85)
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Länder mit Daten")
    ax.set_title(title or f"Datenabdeckung: {column}", loc="left", color=INK)
    fig.tight_layout()
    return fig, ax


def plot_missing_matrix(panel: pd.DataFrame, columns: list[str] | None = None):
    """Fehlende Werte je Variable und Jahr (Anteil der Länder), als
    sequentielle Heatmap - dunkler = mehr Länder mit Daten."""
    if columns is None:
        columns = [c for c in panel.columns if c not in ("country", "year")]
    share = panel.groupby("year")[columns].apply(lambda d: d.notna().mean())
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(columns) + 1.2))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUES)
    # Halbe Zelle Rand auf beiden Seiten, siehe plot_growth_heatmap fuer die
    # Begruendung: sonst verschiebt sich die Jahresachse ueber die Zeit.
    im = ax.imshow(share.T.values, aspect="auto", cmap=cmap,
                   vmin=0, vmax=1,
                   extent=[share.index.min() - 0.5, share.index.max() + 0.5,
                           len(columns) - 0.5, -0.5])
    ax.set_yticks(range(len(columns)), labels=columns, fontsize=9)
    ax.set_xlabel("Jahr")
    ax.set_title("Anteil Länder mit Daten (dunkler = vollständiger)",
                 loc="left", color=INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Anteil")
    fig.tight_layout()
    return fig, ax


def plot_distribution(panel: pd.DataFrame, column: str = config.TARGET):
    """Verteilung einer Variable (Histogramm) mit Kennzahlen."""
    x = panel[column].dropna()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(x, bins=60, color=SEQ_BLUES[3], edgecolor=SURFACE)
    ax.axvline(x.median(), color=INK, linewidth=1.5)
    ax.annotate(f"Median {x.median():.1f}", (x.median(), ax.get_ylim()[1]),
                xytext=(6, -12), textcoords="offset points",
                color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel(column)
    ax.set_ylabel("Anzahl")
    ax.set_title(f"Verteilung: {column}", loc="left", color=INK)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Explorative Analyse (Notebook 01, Abschnitt 3.5)
# ---------------------------------------------------------------------------
DIVERGING_NEG = "#e34948"   # rot  = Rezession
DIVERGING_MID = "#f0efec"   # neutrales Grau um 0
DIVERGING_POS = "#2a78d6"   # blau = Wachstum


def plot_region_growth(panel: pd.DataFrame, static: pd.DataFrame,
                       column: str = config.TARGET):
    """Mittleres BIP-Wachstum je World-Bank-Region als kleine Multiples;
    grau hinterlegt: der globale Mittelwert als Referenz."""
    d = panel.merge(static[["country", "region"]], on="country", how="left")
    regions = sorted(d["region"].dropna().unique())
    global_mean = d.groupby("year")[column].mean()

    ncols = 2
    nrows = -(-len(regions) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.1 * nrows),
                             sharex=True, sharey=True)
    for ax, reg in zip(axes.ravel(), regions):
        m = d[d["region"] == reg].groupby("year")[column].mean()
        ax.plot(global_mean.index, global_mean.values, color=BASELINE,
                linewidth=1.2)
        ax.plot(m.index, m.values, color=SERIES_COLORS[0], linewidth=1.6)
        ax.axhline(0, color=BASELINE, linewidth=0.8)
        ax.set_title(reg if len(reg) < 40 else reg[:37] + "...",
                     loc="left", fontsize=9, color=INK_SECONDARY)
    for ax in axes.ravel()[len(regions):]:
        ax.axis("off")
    fig.suptitle("Mittleres BIP-Wachstum je Region (grau: globaler Mittelwert)",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axes


def plot_histogram_grid(panel: pd.DataFrame, columns: list[str] | None = None):
    """Verteilung aller numerischen Variablen im Überblick."""
    if columns is None:
        columns = [c for c in panel.columns
                   if c not in ("country", "year", "is_global_crisis")]
    ncols = 3
    nrows = -(-len(columns) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.4 * nrows))
    for ax, col in zip(axes.ravel(), columns):
        x = panel[col].dropna()
        ax.hist(x, bins=50, color=SEQ_BLUES[3], edgecolor=SURFACE)
        ax.set_title(col, loc="left", fontsize=9, color=INK_SECONDARY)
        ax.axvline(x.median(), color=INK, linewidth=1)
    for ax in axes.ravel()[len(columns):]:
        ax.axis("off")
    fig.suptitle("Verteilungen (Strich: Median)", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig, axes


def plot_target_correlations(panel: pd.DataFrame,
                             method: str = "spearman"):
    """Rangkorrelation jedes Features mit dem Ziel (gepoolt über alle
    Länder/Jahre). Blau = positiv, Rot = negativ - nur ein grober
    erster Blick, KEINE Kausalität und keine Länder-Heterogenität."""
    cols = [c for c in panel.columns
            if c not in ("country", "year", config.TARGET)]
    corr = (panel[cols + [config.TARGET]]
            .corr(method=method)[config.TARGET]
            .drop(config.TARGET)
            .sort_values())
    colors = [DIVERGING_POS if v >= 0 else DIVERGING_NEG for v in corr.values]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(corr) + 1))
    ax.barh(corr.index, corr.values, color=colors, height=0.62)
    ax.axvline(0, color=BASELINE, linewidth=1)
    for y, v in enumerate(corr.values):
        ax.annotate(f"{v:+.2f}", (v, y), xytext=(4 if v >= 0 else -4, 0),
                    textcoords="offset points", va="center",
                    ha="left" if v >= 0 else "right",
                    fontsize=8.5, color=INK_SECONDARY)
    ax.set_xlim(-0.5, 0.5)
    ax.set_title(f"{method.capitalize()}-Korrelation mit {config.TARGET} "
                 "(gepoolt)", loc="left", color=INK)
    fig.tight_layout()
    return fig, ax


def plot_correlation_matrix(panel: pd.DataFrame,
                            columns: list[str] | None = None,
                            method: str = "spearman",
                            figsize: tuple[float, float] = (7.6, 6.4)):
    """Korrelationsmatrix aller Variablen als untere Dreiecksmatrix.

    Zeigt, welche Variablen dieselbe Information tragen und welche Blöcke
    zusammenhängen. Standard ist die Spearman-Rangkorrelation, weil sie
    robust gegenüber den stark schiefen Verteilungen von Inflation und
    Pro-Kopf-BIP ist und auch monotone, aber nichtlineare Zusammenhänge
    erfasst. Ein Betrag nahe 1 bedeutet, dass sich eine Variable praktisch
    vollständig aus der anderen berechnen lässt, sie also keine zusätzliche
    Information beiträgt.

    Gibt zusätzlich die Korrelationsmatrix zurück, damit sich Auffälligkeiten
    im Notebook direkt nachrechnen lassen.
    """
    from matplotlib.colors import LinearSegmentedColormap

    if columns is None:
        columns = [c for c in panel.columns if c not in ("country", "year")]
    corr = panel[columns].corr(method=method)

    # Obere Dreiecksmatrix ausblenden, sie wiederholt nur die untere Haelfte.
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    cmap = LinearSegmentedColormap.from_list(
        "div_rb", [DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(np.where(mask, np.nan, corr.values), cmap=cmap,
                   vmin=-1, vmax=1)

    ax.set_xticks(range(len(columns)), labels=columns, rotation=45,
                  ha="right", fontsize=9)
    ax.set_yticks(range(len(columns)), labels=columns, fontsize=9)
    ax.grid(False)

    # Werte direkt in die Zellen schreiben, bei dieser Variablenzahl gut
    # lesbar. Auf kraeftigen Farben heller Text, sonst dunkler.
    for i in range(len(columns)):
        for j in range(i + 1):
            v = corr.iloc[i, j]
            ax.annotate(f"{v:.2f}", (j, i), ha="center", va="center",
                        fontsize=8,
                        color=SURFACE if abs(v) > 0.55 else INK_SECONDARY)

    ax.set_title(f"{method.capitalize()}-Korrelation der Variablen",
                 loc="left", color=INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Korrelation")
    fig.tight_layout()
    return fig, ax, corr


def plot_growth_heatmap(panel: pd.DataFrame, static: pd.DataFrame,
                        clip: float = 10.0, since: int = 1970,
                        figsize: tuple[float, float] = (11, 4.4)):
    """Alle Länder x Jahre auf einen Blick: BIP-Wachstum als divergierende
    Heatmap (blau = Wachstum, rot = Rezession), Länder nach Region sortiert.
    Werte werden für die FARBE bei +-clip gekappt (nur Anzeige!).

    Feste, kompakte Größe mit vergrößerter Beschriftung, damit sich die
    Heatmap unabhängig von der Länderzahl direkt in eine Präsentation
    einbetten lässt, ohne unleserlich zu werden. figsize ist in Zoll (wie bei
    matplotlib üblich); cm lassen sich mit cm / 2.54 umrechnen, z. B.
    entspricht ein 13x14-cm-Platzhalter figsize=(13 / 2.54, 14 / 2.54)."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MultipleLocator

    d = (panel[panel["year"] >= since]
         .merge(static[["country", "region"]], on="country", how="left")
         .sort_values(["region", "country"]))
    mat = d.pivot_table(index=["region", "country"], columns="year",
                        values=config.TARGET)
    cmap = LinearSegmentedColormap.from_list(
        "div_rb", [DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS])

    # Feste Größe statt Höhe je Länderzahl (vorher bei 116 Ländern über
    # 10 Zoll hoch, fast quadratisch und damit ungeeignet für eine Folie).
    # Einzelne Länderzeilen bleiben dadurch dünn, das ist bewusst in Kauf
    # genommen: Die Aussage der Heatmap ist ohnehin das Bandmuster über
    # Regionen und Jahre hinweg, nicht der Wert einzelner Länderzeilen.
    fig, ax = plt.subplots(figsize=figsize)
    # extent braucht eine halbe Zelle Rand auf beiden Seiten, sonst wird die
    # Gesamtbreite (y_max - y_min) auf ein Jahr ZU WENIG Spalten aufgeteilt
    # und jede Zelle etwas schmaler als ein Jahr. Der Fehler ist pro Zelle
    # winzig, akkumuliert sich aber über die Jahre: bei 1970-2024 (55 Spalten)
    # landet das Jahr 2020 ohne diese Korrektur bei x ~ 2019.6, sodass die
    # COVID-Spalte optisch wie 2019 statt 2020 aussieht.
    im = ax.imshow(mat.values, aspect="auto", cmap=cmap,
                   vmin=-clip, vmax=clip,
                   extent=[mat.columns.min() - 0.5, mat.columns.max() + 0.5,
                           len(mat), 0])
    # Regionsgrenzen + Beschriftung, größere Schrift für die Lesbarkeit aus
    # der Distanz (Beamer/Bildschirm).
    regions = mat.index.get_level_values(0)
    bounds = np.flatnonzero(regions[1:] != regions[:-1]) + 1
    for b in bounds:
        ax.axhline(b, color=SURFACE, linewidth=2.0)
    seen = {}
    for i, r in enumerate(regions):
        seen.setdefault(r, []).append(i)
    label_pos = [np.mean(v) + 0.5 for v in seen.values()]
    # Kleine Regionen (z. B. North America mit nur 2 Ländern) liegen bei
    # kompakter Höhe so nah an ihren Nachbarn, dass sich die Beschriftungen
    # überlappen würden. Positionen werden deshalb auf einen Mindestabstand
    # auseinandergezogen: erst vorwärts, dann rückwärts, damit auch das
    # letzte Label nicht über den unteren Rand hinausrutscht.
    min_gap = len(mat) * 0.09
    for i in range(1, len(label_pos)):
        label_pos[i] = max(label_pos[i], label_pos[i - 1] + min_gap)
    for i in range(len(label_pos) - 2, -1, -1):
        label_pos[i] = min(label_pos[i], label_pos[i + 1] - min_gap)
    # Unter 8 Zoll Breite (z. B. eine kleine Präsentationskachel statt eines
    # Breitbild-Banners) braucht jedes Element weniger Platz: kürzere
    # Beschriftungen, kleinere Schrift, schräge x-Achsen-Ticks und eine
    # schmalere Farbskala - sonst kollidieren Titel, Ticks und Farbskala.
    compact = figsize[0] < 8

    ax.set_yticks(label_pos,
                  labels=[r if len(r) < (20 if compact else 26)
                          else r[:(17 if compact else 23)] + "..."
                          for r in seen.keys()], fontsize=9 if compact else 11)
    # Runde Tick-Schrittweite statt der automatischen Tick-Wahl, für eine
    # gleichmäßig lesbare x-Achse. Die Schrittweite skaliert mit der
    # Fensterbreite, damit auch ein kurzer Zeitraum (z. B. since nahe am
    # aktuellen Rand) mehr als einen sichtbaren Tick bekommt.
    span = mat.columns.max() - mat.columns.min()
    if span > 40:
        step = 10
    elif span > 15:
        step = 5
    elif span > 6:
        step = 2
    else:
        step = 1
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.tick_params(axis="x", labelsize=9 if compact else 11)
    if compact:
        # Schräg statt waagerecht, sonst laufen die Jahreszahlen bei wenig
        # Breite ineinander.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Jahr", fontsize=10 if compact else 12)
    if compact:
        # Kurzer, zweizeiliger Titel statt des vollen Erklärsatzes - der
        # passt bei wenig Breite nicht mehr in eine Zeile.
        title = f"BIP-Wachstum je Land\n(rot = Rezession, Skala ±{clip:.0f}%)"
    else:
        title = (f"BIP-Wachstum aller Länder (blau = Wachstum, rot = "
                 f"Rezession; Farbskala bei ±{clip:.0f}% gekappt)")
    ax.set_title(title, loc="left", color=INK, fontsize=11 if compact else 13)
    ax.grid(False)
    if compact:
        # fraction/pad/aspect statt shrink: eine schmalere Farbskala lässt
        # der eigentlichen Heatmap mehr von der knappen Breite.
        cbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.03, aspect=16)
    else:
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("% p.a.", fontsize=9 if compact else 11)
    cbar.ax.tick_params(labelsize=8 if compact else 10)
    fig.tight_layout()
    return fig, ax


def plot_persistence(panel: pd.DataFrame):
    """Lag-1-Autokorrelation des BIP-Wachstums je Land: Wie viel
    'Gedächtnis' hat die Zielgröße überhaupt? Werte nahe 0 bedeuten:
    kaum ausbeutbare Dynamik -> der Random Walk ist schwer zu schlagen."""
    ac = (panel.sort_values(["country", "year"])
          .groupby("country")[config.TARGET]
          .apply(lambda s: s.autocorr(lag=1)))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(ac.dropna(), bins=30, color=SEQ_BLUES[3], edgecolor=SURFACE)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.axvline(ac.median(), color=INK, linewidth=1.5)
    ax.annotate(f"Median {ac.median():.2f}", (ac.median(), ax.get_ylim()[1]),
                xytext=(6, -12), textcoords="offset points",
                fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel("Lag-1-Autokorrelation des BIP-Wachstums")
    ax.set_ylabel("Anzahl Länder")
    ax.set_title("Persistenz der Zielgröße je Land", loc="left", color=INK)
    fig.tight_layout()
    return fig, ax, ac


def plot_volatility_by_income(panel: pd.DataFrame, static: pd.DataFrame):
    """Wachstums-Volatilität (Std je Land) nach Einkommensgruppe:
    begründet skalenfreie Metriken (MASE) und per-Serie-Skalierung."""
    d = panel.merge(static[["country", "income_group"]], on="country",
                    how="left")
    vol = (d.groupby(["income_group", "country"])[config.TARGET]
           .std().reset_index(name="vol"))
    order = ["High income", "Upper middle income",
             "Lower middle income", "Low income"]
    order = [o for o in order if o in set(vol["income_group"])]
    data = [vol.loc[vol["income_group"] == g, "vol"].dropna() for g in order]

    fig, ax = plt.subplots(figsize=(8, 3.4))
    bp = ax.boxplot(data, tick_labels=[o.replace(" income", "") for o in order],
                    vert=True, patch_artist=True, widths=0.5,
                    medianprops={"color": INK, "linewidth": 1.6},
                    flierprops={"marker": "o", "markersize": 4,
                                "markerfacecolor": MUTED,
                                "markeredgecolor": "none"})
    for box in bp["boxes"]:
        box.set(facecolor=SEQ_BLUES[1], edgecolor=SEQ_BLUES[4])
    ax.set_ylabel("Std des BIP-Wachstums je Land (%)")
    ax.set_title("Wachstums-Volatilität nach Einkommensgruppe",
                 loc="left", color=INK)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Interaktive Gapminder-Animation (Notebook 06) - benötigt plotly
# ---------------------------------------------------------------------------
# Feste Zuordnung Region -> Projektfarbe (Farbe folgt der Entität, nicht dem
# Rang). Alphabetische Regionsnamen -> stabile Slots aus SERIES_COLORS.
REGION_ORDER = [
    "East Asia & Pacific",
    "Europe & Central Asia",
    "Latin America & Caribbean",
    "Middle East, North Africa, Afghanistan & Pakistan",
    "North America",
    "South Asia",
    "Sub-Saharan Africa",
]
REGION_COLORS = {r: SERIES_COLORS[i] for i, r in enumerate(REGION_ORDER)}

VIZ_LABELS = {
    "gdp_pc": "BIP pro Kopf (konst. US$)",
    "gdp_total": "BIP gesamt (konst. US$)",
    "working_age_share": "Anteil 15–64-Jährige (%)",
    "dependency_ratio": "Abhängigenquote (%)",
    "population": "Bevölkerung",
    "year": "Jahr",
    "region": "Region",
}


def animate_gapminder(viz, x="gdp_pc", y="working_age_share",
                      size="population", color="region",
                      year_range=(1990, 2024), log_x=True,
                      size_max=60, title=None):
    """Rosling-artige animierte Bubble-Chart (plotly express).

    Eine Blase je Land, Größe = Bevölkerung, Farbe = Region, Play-Button
    animiert über die Jahre. Achsen werden über alle Frames FIXIERT, damit
    sich die Blasen bewegen statt die Skala zu springen.

    Parameters
    ----------
    viz : DataFrame aus data_utils.build_viz_panel()
    x, y, size, color : Spaltennamen
    year_range : (erstes, letztes) Jahr der Animation
    log_x : x-Achse logarithmisch (empfohlen für BIP pro Kopf)

    Returns
    -------
    plotly.graph_objects.Figure  (fig.show() / fig.write_html(...))
    """
    import plotly.express as px

    d = viz[(viz["year"] >= year_range[0]) & (viz["year"] <= year_range[1])]
    d = d.dropna(subset=[x, y, size, color]).copy()
    d = d[d[size] > 0]
    d = d.sort_values("year")

    # Achsen über ALLE Frames fixieren. Bei log_x erwartet plotly express die
    # range in DATENEINHEITEN und rechnet selbst nach log10 um; ein hier bereits
    # geloggter Wert würde doppelt geloggt, sodass alle Blasen ausserhalb des
    # sichtbaren Bereichs lägen.
    pad_y = (d[y].max() - d[y].min()) * 0.05
    range_y = [d[y].min() - pad_y, d[y].max() + pad_y]
    if log_x:
        range_x = [d[x].min() * 0.9, d[x].max() * 1.1]
    else:
        pad_x = (d[x].max() - d[x].min()) * 0.05
        range_x = [d[x].min() - pad_x, d[x].max() + pad_x]

    fig = px.scatter(
        d, x=x, y=y, size=size, color=color,
        animation_frame="year", animation_group="country",
        hover_name="name", size_max=size_max, log_x=log_x,
        range_x=range_x, range_y=range_y,
        color_discrete_map=REGION_COLORS,
        category_orders={"region": REGION_ORDER},
        labels=VIZ_LABELS,
        title=title or "BIP-Entwicklung im Rosling-Stil",
    )
    fig.update_traces(marker=dict(line=dict(width=0.5, color=SURFACE),
                                  opacity=0.8))
    fig.update_layout(
        template="plotly_white",
        legend_title_text="Region",
        font=dict(family="sans-serif"),
        title_x=0.02,
    )
    # Animation etwas ruhiger takten
    if fig.layout.updatemenus:
        for b in fig.layout.updatemenus[0].buttons:
            if b.args and isinstance(b.args[1], dict):
                b.args[1].setdefault("frame", {})["duration"] = 600
                b.args[1].setdefault("transition", {})["duration"] = 300
    return fig


def plot_variable_importance(names: list[str], values: np.ndarray,
                             title: str, top_n: int | None = None):
    """Zeigt ein horizontales Balkendiagramm für Merkmalswichtigkeiten, etwa
    die Gewichte der Variablenauswahl eines Temporal Fusion Transformers,
    im Stil des Projekts.

    values wird auf eine Summe von 1 normiert, damit verschiedene Modelle
    und Folds vergleichbar sind. Die Gewichte der Variablenauswahl eines
    Temporal Fusion Transformers sind bereits nichtnegativ, aber nicht
    notwendigerweise bereits normiert.
    """
    values = np.asarray(values, float)
    values = values / values.sum() if values.sum() > 0 else values
    order = np.argsort(values)
    if top_n is not None:
        order = order[-top_n:]
    names_sorted = [names[i] for i in order]
    vals_sorted = values[order]

    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(names_sorted) + 1))
    ax.barh(names_sorted, vals_sorted, color=SEQ_BLUES[3], height=0.62)
    for y, v in enumerate(vals_sorted):
        ax.annotate(f"{v:.1%}", (v, y), xytext=(4, 0),
                    textcoords="offset points", va="center",
                    fontsize=8.5, color=INK_SECONDARY)
    ax.set_xlim(0, max(vals_sorted.max() * 1.25, 0.05))
    ax.set_title(title, loc="left", color=INK)
    fig.tight_layout()
    return fig, ax


def plot_forecast(train: pd.DataFrame, preds: pd.DataFrame, country: str,
                  models: list[str] | None = None,
                  history_years: int = 25,
                  ax: plt.Axes | None = None,
                  show_legend: bool = True,
                  show_origin_label: bool = True,
                  title: str | None = None):
    """Forecast vs. Ist für EIN Land: Historie (Ist) + je Modell die
    Punktprognose, optional mit q10-q90-Band.

    ``preds`` im Standard-Ergebnisschema (config.RESULTS_COLUMNS).

    ``ax`` erlaubt es, den Plot in ein bestehendes Raster einzubetten
    (z.B. eine Übersicht über viele Länder oder Gruppen). Ohne Angabe
    wird wie bisher eine eigene Figur erzeugt (``fig`` ist dann nicht
    ``None``). ``show_legend=False`` unterdrückt die Legende, und
    ``show_origin_label=False`` unterdrückt die Textbeschriftung
    "Prognosestart" (die vertikale Linie bleibt erhalten), beides
    sinnvoll bei vielen kleinen Teilplots in einem Raster. ``title``
    ersetzt die Standardüberschrift, z.B. für kürzere Beschriftungen in
    einem Raster. Ein Titel sollte NICHT zusätzlich per ``ax.set_title()``
    von außen gesetzt werden, da sich sonst die linksbündige
    Standardüberschrift und eine von außen zentriert gesetzte Überschrift
    überlagern (zwei unterschiedliche Titelplätze in Matplotlib).
    """
    d_hist = (train[train["country"] == country]
              .dropna(subset=[config.TARGET])
              .tail(history_years))
    d_pred = preds[preds["country"] == country]
    models = models or sorted(d_pred["model"].unique())

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(d_hist["year"], d_hist[config.TARGET],
            color=MODEL_COLORS["actual"], label="Ist", linewidth=2.0)

    # Ist-Werte im Prognosezeitraum (gepunktet angebunden)
    d_true = d_pred.drop_duplicates("year").sort_values("year")
    if d_true["y_true"].notna().any():
        ax.plot(d_true["year"], d_true["y_true"], color=MODEL_COLORS["actual"],
                linewidth=1.2, linestyle=":", label="Ist (Prognosezeitraum)")

    for m in models:
        dm = d_pred[d_pred["model"] == m].sort_values("year")
        color = MODEL_COLORS.get(m) or _fallback_color(m)
        if dm[["q10", "q90"]].notna().all(axis=None):
            ax.fill_between(dm["year"], dm["q10"], dm["q90"],
                            color=color, alpha=0.15, linewidth=0)
        ax.plot(dm["year"], dm["y_pred"], color=color, marker="o",
                markersize=4, label=config.MODEL_NAMES.get(m, m))

    origin = int(d_pred["origin_year"].iloc[0]) if len(d_pred) else None
    if origin:
        ax.axvline(origin + 0.5, color=BASELINE, linewidth=1)
        if show_origin_label:
            ax.annotate("Prognosestart", (origin + 0.5, ax.get_ylim()[1]),
                        xytext=(5, -12), textcoords="offset points",
                        color=MUTED, fontsize=8, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE,
                                  edgecolor="none", alpha=0.85))
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_xlabel("Jahr")
    ax.set_ylabel("BIP-Wachstum (%)")
    ax.set_title(title if title is not None else f"Prognose vs. Ist - {country}",
                 loc="left", color=INK)
    if show_legend:
        add_legend(ax)
    if fig is not None:
        fig.tight_layout()
    return fig, ax
