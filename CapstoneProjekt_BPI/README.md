# AI4BP Capstone — Multi-Country-BIP-Prognose

Prognose des realen BIP-Wachstums vieler Länder über 1–5 Jahre mit einem
global über alle Länder trainierten Modell und im Voraus bekannten
exogenen Treibern (v. a. Demografie).

**Team:** Antoine Hunou, Deborah Lang, Marius Riesle

## Projekt auf einen Blick

- **Ziel:** Prognose des realen BIP-Wachstums für viele Länder über **1–5 Jahre**
- **Datenbasis:** World Bank `WDI` plus Länder-Metadaten
- **Modelle:** `SARIMAX`, `Temporal Fusion Transformer (TFT)`, `Chronos-2`
- **Vergleichsrahmen:** gemeinsames Panel, identische Backtest-Folds,
  einheitliches Output-Schema
- **Hauptmetrik:** `MASE`, ergänzt um `RMSE`, `Pinball-Loss` und
  Intervall-Coverage

## Leitfrage des Projekts

Wie gut schneiden klassische, Deep-Learning- und Foundation-Model-Ansätze bei
der mehrjährigen Multi-Country-BIP-Prognose ab, wenn sie unter identischen
Bedingungen miteinander verglichen werden?

## Notebooks im Überblick

| Notebook                                 | Rolle im Projekt                                                    | Zentrale Ausgabe                                  |
| ---------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------- |
| `01_Datenbeschaffung_Aufbereitung.ipynb` | Beschafft und bereinigt die Daten, erstellt das Modellierungs-Panel | `data/processed/gdp_panel.csv`, `panel_meta.json` |
| `02_SARIMAX.ipynb`                       | Klassischer Benchmark mit exogenen Variablen                        | `data/results/preds_arima.csv`                    |
| `03_TFT.ipynb`                           | Globales Deep-Learning-Modell für Multi-Horizon-Forecasts           | `data/results/preds_tft.csv`                      |
| `04_Chronos.ipynb`                       | Zero-Shot-Benchmark mit `Chronos-2`                                 | `data/results/preds_chronos2.csv`                 |
| `05_Vergleich.ipynb`                     | Gemeinsame Evaluation aller Modelle                                 | Vergleichstabellen und Plots                      |
| `06_BIP-Animations.ipynb`                | Visualisiert Prognosen und Länderentwicklungen                      | `data/results/bip_animation.html`                 |

## Projektstruktur

```text
CapstoneProjekt_BPI/
├── data/
│   ├── raw/                # Rohdaten und Metadaten
│   ├── processed/          # aufbereitetes Panel für alle Modelle
│   └── results/            # Modellprognosen und Vergleichsergebnisse
├── figures/                # exportierte Abbildungen
├── src/                    # gemeinsame Python-Module für alle Notebooks
│   ├── config.py           # Pfade, Indikatoren, Backtest-Regeln, Ergebnisschema
│   ├── data_utils.py       # Datenbeschaffung, Panel-Aufbereitung, Splits
│   ├── evaluation.py       # MASE, RMSE, Pinball, Save/Load der Prognosen
│   ├── plotting.py         # einheitliche Visualisierungen
│   └── tft_utils.py        # Hilfsfunktionen für das TFT-Modell
├── pitch and plan/
├── 01_Datenbeschaffung_Aufbereitung.ipynb
├── 02_SARIMAX.ipynb
├── 03_TFT.ipynb
├── 04_Chronos.ipynb
├── 05_Vergleich.ipynb
├── 06_BIP-Animations.ipynb
└── Präsentation_capstone_BIP.pdf
```

## Empfohlener Einstieg

Für einen schnellen Überblick empfiehlt sich folgende Reihenfolge:

1. `01_Datenbeschaffung_Aufbereitung.ipynb` für Datenbasis und Feature-Logik
2. `02-04_*` für die verschiedenen Modelle
3. `05_Vergleich.ipynb` für die modellübergreifende Bewertung
4. `06_BIP-Animations.ipynb` für die visuelle Aufbereitung der Ergebnisse

## Setup und Reproduzierbarkeit

Die Abhängigkeiten sind in der `pyproject.toml` im **Repo-Root** definiert.
Das Standard-Setup erfolgt mit:

```bash
uv sync
```

Anschließend sollten die Notebooks mit dem `.venv`-Kernel des Repositories
gestartet werden. In VS Code genügt es, den passenden Interpreter auszuwählen.

## CRISP-DM-orientierter Workflow

Unser Projekt folgt der Logik von **CRISP-DM**. Die einzelnen Phasen sind im
Repository klar auf Dateien und Notebooks abgebildet:

1. **Business Understanding**
   - `README.md` und `pitch and plan/` definieren die Zielsetzung des Projekts:
     Prognose des realen BIP-Wachstums mehrerer Länder über **1–5 Jahre** und
     Vergleich unterschiedlicher Modellklassen unter identischen Bedingungen.

2. **Data Understanding & Data Preparation**
   - `01_Datenbeschaffung_Aufbereitung.ipynb` lädt die World-Bank-Daten
     (`WDI`, CC-BY-4.0), prüft und bereinigt die Daten und erstellt das
     gemeinsame Länder-Panel als Grundlage für alle Modelle.

3. **Modeling**
   - `02_SARIMAX.ipynb` implementiert den klassischen Zeitreihen-Benchmark.
   - `03_TFT.ipynb` implementiert den `Temporal Fusion Transformer` als
     globales Deep-Learning-Modell.
   - `04_Chronos.ipynb` implementiert den Zero-Shot-Benchmark mit `Chronos-2`.
   - Alle drei Notebooks nutzen dieselbe Datenbasis, dieselben Backtest-Folds
     und speichern ihre Prognosen im gemeinsamen Standard-Schema.

4. **Evaluation**
   - `05_Vergleich.ipynb` lädt alle erzeugten `preds_*.csv`, berechnet die
     zentralen Metriken und ermöglicht den konsistenten Vergleich der Modelle.

5. **Deployment / Communication**
   - Im Projektkontext verstehen wir diese Phase als unnötig.

## Verbindliche Konventionen (`src/config.py`)

- **Horizont:** $H = 5$ Jahre
- **Backtesting:** Walk-forward mit expanding window; Origins
  `1999 / 2004 / 2009 / 2014` plus finaler Holdout `2019 → Test 2020–2024`
- **Leakage-Schutz:** reine Zeit-Splits, kein Shuffling; `PAST_ONLY`-Variablen
  werden ausschließlich gelaggt verwendet
- **Primärmetrik:** `MASE`
- **Unsicherheit:** Quantile `q10` und `q90`
