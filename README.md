# AI-for-Business-Prognosis

Willkommen im Projekt-Repository für das Modul "AI for Business Prognosis" (Prof. Dr. Carsten Lanquillon). In diesem Repo teilen wir unsere Methodentutorials und das Abschlussprojekt.

## 👥 Team

- Antoine Hunou
- Deborah Lang
- Marius Riesle

## 📈 Methoden

- (S)ARIMA(X) (klassische Zeitreihenanalyse)
- N-BEATS (Deep Learning)

## 📦 Installation

> ℹ️ **PyTorch (CPU vs. CUDA):** Standardmäßig wird die **CPU-Variante** von PyTorch installiert.
> Wer eine NVIDIA-GPU nutzen möchte, kann stattdessen die **CUDA-Variante** installieren (siehe unten).

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

### PyTorch mit CUDA (optional, NVIDIA-GPU)

Standard (`uv sync`) installiert die CPU-Variante. Für die CUDA-Variante die
`cpu`-Gruppe deaktivieren und die `cu124`-Gruppe aktivieren:

```bash
uv sync --no-group cpu --group cu124
```

Das lässt sich mit den übrigen Gruppen kombinieren, z.B.:

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
├── pyproject.toml                                      # Projektabhängigkeiten und Gruppendefinitionen
├── uv.lock                                             # Gelockte Abhängigkeiten (uv)
├── DeepLearning_NBEATS/                                # N-BEATS als Deep-Learning-Methode
│   ├── Präsentation_NBEATS.pdf                         # Präsentation als PDF
│   ├── HandsOnÜbung/
│   │   ├── N_BEATS_HandsOnÜbung.ipynb                  # Notebook zur Hands-On-Übung
│   │   └── data/                                       # Daten für die HandsOn-Übung
│   └── LiveDemo/
│       ├── N_BEATS_LiveDemo.ipynb                      # Notebook zu der Live-Demo
│       ├── Bilder/                                     # Grafiken für die Live-Demo
│       └── Daten/                                      # Daten (Export aus dem SARIMAX-Notebook)
└── KlassischeMethode_S_ARIMA_X/                        # (S)ARIMA(X) als klassische Methode
    ├── Präsentation_SArimaX.pdf                        # Präsentation als PDF
    ├── S_ARIMA_X_LiveDemo.ipynb                        # Notebook zu der Live-Demo
    ├── S_ARIMA_X_HandsOnÜbung_BhkwPfettscher.ipynb     # Notebook zu der HandsOn-Übung
    ├── BHKWs_Daten/                                    # Daten für die HandsOn-Übung
    └── ElectricDemand_Daten/                           # Daten für die Live-Demo
```
