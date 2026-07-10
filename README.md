# AI-for-Business-Prognosis-S-ARIMA-X-

Willkommen im Projekt-Repository für das Modul "AI for Business Prognosis" (Prof. Dr. Carsten Lanquillon). In diesem Repo teilen wir unsere Methodentutorials und das Abschlussprojekt.

## 👥 Team

- Antoine Hunou
- Deborah Lang
- Marius Riesle

## 📈 Methoden

- (S)ARIMA(X) (klassische Zeitreihenanalyse)
- N-BEATS (Deep Learning)

## 📦 Installation

### Base environment für die aktuelle Aufgabe

```bash
uv sync

or

uv sync --group nbeats
```

### SARIMA environment

```bash
uv sync --group sarima
```

### Live-demo environment

```bash
uv sync --group live_demos
```

## 📁 Projektstruktur

```text
AI-for-Business-Prognosis-S-ARIMA-X-/
├── README.md
├── pyproject.toml                                      # Projektabhängigkeiten und Gruppendefinitionen
├── DeepLearning_NBEATS/                                # N-BEATS als Deep-Learning-Methode
│   ├── Präsentation_NBEATS.pdf                         # Präsentation als PDF
│   └── HandsOnÜbung/
│       ├── N_BEATS_HandsOnÜbung.ipynb                  # Notebook zur Hands-On-Übung
│       └── data/
└── KlassischeMethode_S_ARIMA_X/                        # (S)ARIMA(X) als klassische Methode
    │── Präsentation_SArimaX.pdf                        # Präsentation als PDF
    │── ElectricDemand_Daten/                           # Daten für die Live-Demo
    ├── BHKWs_Daten/                                    # Daten für die HandsOn-Übung
    ├── S_ARIMA_X_LiveDemo.ipynb                        # Notebook zu der LiveDemo
    └── S_ARIMA_X_HandsOnÜbung_BhkwPfettscher.ipynb     # Notebook zu der HandsOn-Übung
```
