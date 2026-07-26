# AI-for-Business-Prognosis

Willkommen im Projekt-Repository für das Modul **"AI for Business Prognosis"**
(Prof. Dr. Carsten Lanquillon). Das Repo bündelt unsere Methodentutorials,
Live-Demos und das Capstone-Projekt zur mehrjährigen BIP-Prognose.

## 👥 Team

- Antoine Hunou
- Deborah Lang
- Marius Riesle

## 📈 Inhalte und Methoden

### Tutorial- und Demo-Teile

- `(S)ARIMA(X)` für klassische Zeitreihenanalyse
- `N-BEATS` als Deep-Learning-Ansatz

### Capstone-Projekt `CapstoneProjekt_BPI`

Im Capstone wird das **reale BIP-Wachstum mehrerer Länder** prognostiziert.
Dafür werden einheitliche Datenpipelines, klassische Modelle und moderne
Deep-Learning-/Foundation-Model-Ansätze kombiniert:

- `SARIMAX`
- `Temporal Fusion Transformer (TFT)`
- `Chronos-2`
- modellübergreifender Vergleich in einem separaten Evaluations-Notebook

Weitere Details stehen in `CapstoneProjekt_BPI/README.md`.

## 📦 Installation

> ℹ️ **PyTorch (CPU vs. CUDA):** Standardmäßig wird die **CPU-Variante** von PyTorch installiert.
> Wer eine NVIDIA-GPU nutzen möchte, kann stattdessen die **CUDA-Variante** installieren (siehe unten).

### Standard-Setup

```bash
uv sync

or

uv sync --group capstone
```

### Zusätzliche Gruppen

```bash
uv sync --group capstone
uv sync --group nbeats
uv sync --group sarima
uv sync --group live_demos
```

### PyTorch mit CUDA (optional, NVIDIA-GPU)

Standard (`uv sync`) installiert die CPU-Variante. Für die CUDA-Variante die
`cpu`-Gruppe deaktivieren und die `cu124`-Gruppe aktivieren:

```bash
uv sync --no-group cpu --group cu124
```

Beispiel in Kombination mit einer Fachgruppe:

```bash
uv sync --group nbeats --no-group cpu --group cu124
```

> ⚠️ Die CUDA-Variante benötigt eine passende NVIDIA-GPU samt Treiber. Der Index `cu124`
> steht für CUDA 12.4; bei Bedarf lässt sich der PyTorch-Index in `pyproject.toml`
> (`[[tool.uv.index]]`) auf eine andere CUDA-Version anpassen.

## 📁 Projektstruktur

```text
AI-for-Business-Prognosis/
├── README.md
├── pyproject.toml
├── CapstoneProjekt_BPI/                                # Abschlussprojekt zur Multi-Country-BIP-Prognose
│   ├── Präsentation_capstone_BIP.pdf
│   ├── 01_Datenbeschaffung_Aufbereitung.ipynb          # Datenbeschaffung und Panel-Aufbereitung
│   ├── 02_SARIMAX.ipynb                                # Klassisches SARIMAX Forecasting
│   ├── 03_TFT.ipynb                                    # Temporal Fusion Transformer
│   ├── 04_Chronos.ipynb                                # Chronos-2
│   ├── 05_Vergleich.ipynb                              # Vergleich der Modellgüte
│   ├── 06_BIP-Animations.ipynb                         # Visualisierung/Animation
│   ├── README.md
│   ├── data/
│   │   ├── processed/                                  # Aufbereitete Paneldaten
│   │   ├── raw/                                        # Rohdaten und Metadaten
│   │   └── results/                                    # Modellvorhersagen und Ergebnisse
│   ├── figures/
│   ├── pitch and plan/
│   └── src/                                            # Gemeinsame Python-Module für die Notebooks
├── DeepLearning_NBEATS/                                # N-BEATS als Deep-Learning-Methode
│   ├── Präsentation_NBEATS.pdf
│   ├── HandsOnÜbung/                                   # Notebook + Daten für HandsOn-Übung
│   │   ├── N_BEATS_HandsOnÜbung.ipynb
│   │   └── data/
│   └── LiveDemo/                                       # Notebook + Daten für Live-Demo
│       ├── N_BEATS_LiveDemo.ipynb
│       ├── Bilder/
│       └── Daten/
└── KlassischeMethode_S_ARIMA_X/                        # (S)ARIMA(X) als klassische Methode
    ├── Präsentation_SArimaX.pdf
    ├── S_ARIMA_X_LiveDemo.ipynb                        # Notebook zu der Live-Demo
    ├── S_ARIMA_X_HandsOnÜbung_BhkwPfettscher.ipynb     # Notebook zu der HandsOn-Übung
    ├── BHKWs_Daten/                                    # Daten für die HandsOn-Übung
    └── ElectricDemand_Daten/                           # Daten für die Live-Demo
```
