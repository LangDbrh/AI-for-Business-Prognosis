"""Zentrale Konfiguration für das AI4BP-Capstone: Multi-Country-BIP-Prognose.

Alle Notebooks und Module beziehen Pfade, Indikator-Codes und
Backtest-Konventionen AUSSCHLIESSLICH aus dieser Datei, damit alle
Teammitglieder mit identischen Einstellungen arbeiten.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Pfade (relativ zur Repo-Wurzel; funktionieren lokal und in Colab,
# solange die Notebooks aus notebooks/ heraus gestartet werden)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

RAW_WDI_CSV = RAW_DIR / "wdi_raw_long.csv"
RAW_META_CSV = RAW_DIR / "countries_meta.csv"
PANEL_CSV = PROCESSED_DIR / "gdp_panel.csv"
PANEL_META_JSON = PROCESSED_DIR / "panel_meta.json"

# Prognosen auf dem WELTAGGREGAT liegen bewusst in einem Unterordner und NICHT
# neben den Länderprognosen: evaluation.load_all_predictions() sammelt
# data/results/preds_*.csv nicht rekursiv ein. Eine Weltdatei direkt in
# RESULTS_DIR würde dort als weiteres "Modell" mit 25 statt 2.900 Zeilen
# auftauchen und den Ländervergleich in Notebook 05 verfälschen.
WORLD_RESULTS_DIR = RESULTS_DIR / "world"


def rel(pfad) -> str:
    """Pfad relativ zur Projektwurzel als String.

    Damit sind Notebook-Ausgaben unabhängig davon, wer das Notebook auf
    welchem Rechner laufen lässt. Liegt der Pfad ausserhalb der Wurzel,
    wird er unverändert zurückgegeben.
    """
    pfad = Path(pfad)
    try:
        return pfad.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(pfad)


# ---------------------------------------------------------------------------
# World-Bank-Indikatoren (WDI)
# ---------------------------------------------------------------------------
# Ziel der Prognose
TARGET = "gdp_growth"

# Code -> (Spaltenname im Panel, Beschreibung)
INDICATORS: dict[str, tuple[str, str]] = {
    # --- Ziel ---
    "NY.GDP.MKTP.KD.ZG": ("gdp_growth", "Reales BIP-Wachstum (% p.a.)"),
    # --- Referenz / Skalen-Feature ---
    "NY.GDP.PCAP.KD": ("gdp_pc", "Reales BIP pro Kopf (konst. 2015 US$)"),
    # --- Exogen, Gruppe A: im Voraus bekannt (Demografie, UN-Projektionen) ---
    "SP.POP.GROW": ("pop_growth", "Bevölkerungswachstum (% p.a.)"),
    "SP.POP.1564.TO.ZS": ("working_age_share", "Anteil 15-64-Jährige (%)"),
    "SP.POP.DPND": ("dependency_ratio", "Abhängigenquote (% der 15-64-J.)"),
    # --- Exogen, Gruppe B: nur Vergangenheit bekannt -> wird gelaggt ---
    "NE.GDI.TOTL.ZS": ("investment_gdp", "Bruttoinvestitionen (% des BIP)"),
    "NE.TRD.GNFS.ZS": ("trade_gdp", "Handel, Exporte+Importe (% des BIP)"),
    "FP.CPI.TOTL.ZG": ("inflation", "Inflation, Verbraucherpreise (% p.a.)"),
    # --- Referenz, NUR für Visualisierung (Notebook 06), KEIN Modell-Feature ---
    "SP.POP.TOTL": ("population", "Bevölkerung, gesamt (Personen)"),
}

# Exogene Treiber, die zur Forecast-Zeit als bekannt gelten dürfen
# (Demografie: UN World Population Prospects liefern Projektionen >5 Jahre).
#
# BEWUSST OHNE dependency_ratio: Die Abhängigenquote ist definitorisch aus
# working_age_share berechenbar, es gilt
#     dependency_ratio = (100 - working_age_share) / working_age_share * 100.
# Auf dem Panel nachgerechnet stimmt das bis auf Rundung (Spearman exakt -1,00),
# beide Variablen tragen also dieselbe Information. Als exogene Regressoren
# gemeinsam verwendet, treiben sie die Konditionszahl der SARIMAX-Designmatrix
# im Median von rund 1.450 auf rund 56.000 (75 von 116 Ländern über 30.000),
# was die Koeffizienten instabil und uninterpretierbar macht. Siehe die
# Korrelationsmatrix in Notebook 01, Abschnitt 3.5.
#
# dependency_ratio bleibt als Spalte im Panel erhalten (Exploration), wird aber
# von keinem Modell mehr als Feature genutzt.
KNOWN_FUTURE = ["pop_growth", "working_age_share"]

# Treiber, deren Zukunftswerte NICHT bekannt sind -> nur gelaggt verwenden!
PAST_ONLY = ["gdp_pc", "investment_gdp", "trade_gdp", "inflation"]
LAG_YEARS = 1  # Standard-Lag für PAST_ONLY-Variablen

# Reine Referenz-/Visualisierungsspalten: werden heruntergeladen, aber NIE
# als Modell-Feature genutzt (Notebook 06 Gapminder-Animation: Blasengröße).
# Aus dem Modell-Panel (gdp_panel.csv) werden sie entfernt, damit die
# Modelle 02-04 exakt dieselben Feature-Spalten wie bisher sehen.
REFERENCE_ONLY = ["population"]

# ---------------------------------------------------------------------------
# Zeitraum & Länderfilter
# ---------------------------------------------------------------------------
START_YEAR = 1960
LAST_YEAR = 2024          # letztes Jahr mit WDI-Daten (Stand Juli 2026)
MIN_YEARS = 35            # Mindestanzahl Jahre mit gültigem BIP-Wachstum
MAX_GAP_INTERPOLATE = 2   # innere Lücken bis zu dieser Länge interpolieren

# Exogen-Abdeckung: Länder, deren Treiber-Variablen im modellrelevanten
# Fenster überwiegend fehlen (Mikrostaaten wie Monaco/Tuvalu), fliegen raus -
# sonst prognostizieren die Modelle stillschweigend unterschiedliche
# Ländermengen und der Vergleich in Notebook 05 wäre unfair.
EXOG_SINCE = 1995         # Abdeckung wird ab diesem Jahr gemessen
MIN_EXOG_COVERAGE = 0.8   # jede Treiber-Variable braucht >= 80% Abdeckung

# ---------------------------------------------------------------------------
# Backtesting-Konvention (walk-forward, expanding window)
# ---------------------------------------------------------------------------
HORIZON = 5                                   # Prognosehorizont in Jahren
# Trainingsende der Backtest-Folds
BACKTEST_ORIGINS = [1999, 2004, 2009, 2014]
FINAL_ORIGIN = 2019                           # Training <= 2019, Test 2020-2024
ALL_ORIGINS = BACKTEST_ORIGINS + [FINAL_ORIGIN]

# Quantile für Unsicherheitsintervalle (probabilistische Modelle)
QUANTILES = [0.1, 0.5, 0.9]

# ---------------------------------------------------------------------------
# Ergebnis-Schema: JEDES Modell-Notebook speichert seine Prognosen über
# src.evaluation.save_predictions() in data/results/preds_<modell>.csv
# mit exakt diesen Spalten (q10/q90 dürfen leer sein):
# ---------------------------------------------------------------------------
RESULTS_COLUMNS = [
    "model",        # z.B. "rw_drift", "sarimax", "tft", "chronos2"
    "country",      # ISO3-Code, z.B. "DEU" (WORLD_ID beim Weltaggregat)
    "origin_year",  # letztes Trainingsjahr des Folds
    "year",         # prognostiziertes Jahr
    "h",            # Horizont-Schritt 1..HORIZON
    "y_true",       # tatsächliches BIP-Wachstum
    "y_pred",       # Punktprognose (bzw. Median)
    "q10",          # unteres Quantil (optional, NaN erlaubt)
    "q90",          # oberes Quantil (optional, NaN erlaubt)
]

MODEL_NAMES = {
    "rw_drift": "Random Walk mit Drift (Baseline)",
    "mean": "Historischer Mittelwert (Baseline)",
    "arima": "ARIMA",
    "tft": "Temporal Fusion Transformer",
    "chronos2": "Chronos-2 (zero-shot)",
    # Weltaggregat
    "rw_drift_world": "Random Walk mit Drift (Welt)",
    "mean_world": "Historischer Mittelwert (Welt)",
    "chronos2_world": "Chronos-2, top-down",
    "chronos2_world_ctx": "Chronos-2, top-down mit Länderkontext",
    "chronos2_world_uni": "Chronos-2, top-down ohne Kovariaten",
    "chronos2_bottom_up": "Chronos-2, bottom-up",
}

# ---------------------------------------------------------------------------
# Weltaggregat
# ---------------------------------------------------------------------------
# Platzhalter in der Spalte 'country', wenn eine Zeile nicht ein Land, sondern
# die aggregierte Weltreihe beschreibt. Bewusst der offizielle Weltbank-Code
# für "World", damit er sich nie mit einem echten ISO3-Ländercode überschneidet.
WORLD_ID = "WLD"

# Jahre, in denen alle Modellländer ein BIP-Niveau haben. Davor beruht die
# Weltreihe auf einer kleineren Länderstichprobe (siehe build_world_panel).
WORLD_COMPLETE_SINCE = 1989
