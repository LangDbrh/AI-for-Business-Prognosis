---
title: "Team Plan"
---

**Course: AI for Business Prognosis (AI4BP)**

**Name / Team:** X-Team

# Team Plan

*One per team. Fill this in AFTER the team has chosen a project.*

---

## Team and members

Team name: X-Team

Members: Antoine Hunou, Marius Riesle, Deborah Lang

## Chosen problem  (one paragraph)

Wir prognostizieren das reale BIP-Wachstum vieler Länder über einen mehrjährigen Horizont von einem bis fünf Jahren, jährlich. Statt für jedes Land ein eigenes Modell zu bauen, trainieren wir ein global über alle Länder gemeinsames Modell (Cross-Learning), das die kurzen Länderhistorien bündelt. Als Prädiktoren nutzen wir bewusst langsam veränderliche, im Voraus bekannte exogene Treiber, vor allem die Bevölkerungsaltersstruktur und den Erwerbsanteil, die über UN-Projektionen bekannt sind, sowie Investitionsquote und Handel als gelaggte Größen, um die klassische Leakage-Falle zu vermeiden. Die Prognose unterstützt mittelfristige Haushalts-, Investitions- und Kapitalallokationsentscheidungen im Stil von IWF- oder OECD-Projektionen. Weil langfristiges BIP-Wachstum nahe an einem Random Walk liegt, definieren wir Erfolg von Beginn an ehrlich als Schlagen der Random-Walk-Baseline bei gleichzeitig kalibrierten Unsicherheitsintervallen, nicht als perfekte Punktprognose. Die finale Auswertung über 2.900 Prognosen je Modell bestätigt dieses Bild: SARIMAX (MASE 0,846) und der Zero-Shot-Benchmark Chronos-2 (MASE 0,845) liegen praktisch gleichauf vorn, das eigens trainierte TFT folgt mit MASE 0,922, und alle drei schlagen die Baseline (MASE 1,083) statistisch signifikant. Im finalen, pandemiegeprägten Holdout 2020 bis 2024 rücken alle Verfahren nahe an die Baseline heran, was die anfängliche Risikoeinschätzung bestätigt.

## Data

Source: World Bank – World Development Indicators (WDI), Zugriff via Python-Paket `wbgapi`, Lizenz CC-BY-4.0. Zielgröße ist das reale BIP-Wachstum, exogen sind Altersstruktur und Erwerbsanteil (im Voraus bekannt), Investitionsquote, Handel und Inflation (gelaggt). Nach Data Gate (Mindestlänge, Exogen-Abdeckung ab 1995) umfasst das finale Panel 116 Volkswirtschaften, jährlich von 1960 bis 2024.

Access CONFIRMED?  [x] yes  *(WDI-Panel vollständig aufbereitet unter `data/processed/gdp_panel.csv`; Data Gate und Anti-Leakage-Regeln in Notebook 01 geprüft)*

Where it lives (repo / path): GitHub-Repo `Ant01neH/AI-for-Business-Prognosis`, Ordner `CapstoneProjekt_BPI/` (Notebooks `01`–`06`, gemeinsame Module in `src/`, Offline-Test in `tests/smoke_test.py`)

## Method-coverage matrix

*Compare AT LEAST baseline vs. classical vs. deep learning. At least ONE method must be NEW to the team.*

| Layer                 | Method chosen | Library | Owner | NEW to us? (Y/N) |
|------------------------|---------------|---------|-------|------------------|
| Baseline               | Random Walk mit Drift (auf Wachstumsraten) | eigene Implementierung (`src/evaluation.py`) | Antoine Hunou | N |
| Classical              | ARIMA/SARIMA/SARIMAX im internen Modellvergleich je Land, final SARIMAX mit Demografie-Regressoren | statsmodels | Marius Riesle | N |
| Deep Learning          | TFT (Temporal Fusion Transformer), global trainiert | PyTorch Forecasting | Deborah Lang | **Y** |
| DL – Zusatz-Benchmark  | Chronos-2 (Foundation-Model, Zero-Shot mit Kovariaten) | chronos-forecasting (Amazon Science) | Antoine Hunou | **Y** |

## Evaluation

Primary metric: MASE (skalenfrei, länderübergreifend vergleichbar, zunächst je Land und Origin gebildet und dann gemittelt), ergänzt um RMSE/MAE sowie Pinball-Loss und 80%-Intervallabdeckung für die probabilistischen Modelle (TFT, Chronos-2).

Backtesting scheme:

[x] walk-forward   [x] expanding window   [ ] sliding window   *(Origins 1999/2004/2009/2014, finaler Holdout Training ≤2019 / Test 2020–2024; wachsende Historie über mehrere Konjunkturregime)*

**Finale Ergebnisse (Gesamtvergleich, 2.900 Prognosen je Modell, 116 Länder)**

| Modell | MASE | MASE vs. Baseline | RMSE | MAE | 80%-Abdeckung |
|---|---|---|---|---|---|
| Random Walk mit Drift (Baseline) | 1,083 | – | 6,874 | 4,005 | – |
| SARIMAX | 0,846 | +0,219 | 5,351 | 3,113 | – |
| Temporal Fusion Transformer | 0,922 | +0,149 | 5,593 | 3,364 | 62,7 % |
| Chronos-2 (Zero-Shot) | 0,845 | +0,220 | 5,460 | 3,083 | 84,3 % |

Alle drei Modelle schlagen die Baseline statistisch signifikant (Wilcoxon-Vorzeichen-Rang-Test, je p < 0,002). Der Unterschied zwischen Chronos-2 und SARIMAX ist bei nahezu identischer mittlerer MASE im Wilcoxon-Test signifikant (p = 0,013), im gepaarten t-Test dagegen nicht (p = 0,958), was auf viele kleine, konsistente Unterschiede statt weniger großer Ausreißer hindeutet. Chronos-2 liefert zudem die besser kalibrierten Unsicherheitsintervalle, während das TFT mit engeren, aber zu selten treffenden Intervallen sein Risiko unterschätzt.

## Roles

*Some students own two roles.*

- Baseline owner: Antoine Hunou
- Classical owner (ARIMA/SARIMA/SARIMAX): Marius Riesle
- Deep-learning owner (TFT): Deborah Lang
- Deep-learning owner (Chronos-2, Zusatz-Benchmark): Antoine Hunou
- Evaluation and uncertainty owner (Notebook 05, Residualdiagnostik, Kalibrierung): Marius Riesle
- Write-up and slides owner: gesamtes Team

## Milestones to the final presentation

*Ordered checklist. Keyed to CRISP-DM phases 1-5 and MVP-first. Build baseline + classical end-to-end BEFORE adding deep learning. No dates -- work the order.*

- [x] Phase 1 – Business understanding: decision, target, success criteria fixed
- [x] Phase 2 – Data understanding: WDI geladen, Zugriff bestätigt, exploriert (Notebook 01)
- [x] Phase 3 – Data preparation: bereinigt, Panel gebaut, Zeit-Split + walk-forward aufgesetzt (Notebook 01)
- [x] MVP: Random-Walk-Baseline + SARIMAX end-to-end mit Evaluation (Notebooks 01, 02)
- [x] Phase 4 – Modeling: TFT (die neue Methode) ergänzt, danach Chronos-2 als Zero-Shot-Benchmark (Notebooks 03, 04)
- [x] Phase 5 – Evaluation: alle Methoden inkl. Baseline verglichen; Multi-Horizon- und Exogen-Challenge adressiert (Notebook 05)
- [x] Deliverables assembled: slides + dokumentierte Notebooks *(Notebooks dokumentiert & lauffähig sowie Foliensatz für die Abschlusspräsentation finalisiert)*
- [x] Dry-run of final presentation

## Top 3 risks and mitigation

| # | Risk | Mitigation |
|---|------|------------|
| 1 | Langfristiges BIP-Wachstum liegt nahe am Random Walk, Baseline schwer zu schlagen | Erfolg als „Baseline schlagen + kalibrierte Intervalle" definiert; bestätigt: SARIMAX/Chronos-2 schlagen Baseline signifikant um rund 0,22 MASE-Punkte |
| 2 | Strukturbrüche (Finanzkrise 2008/09, Corona-Pandemie 2020) verzerren Training und Evaluation | Krisenjahre über mehrere Walk-Forward-Origins abgedeckt; finaler Holdout 2020–2024 bewusst als härtester Test verwendet, Modelle bleiben dort nah an, aber nicht unter der Baseline |
| 3 | Exogenen-Leakage: Treiber wären fünf Jahre voraus eigentlich unbekannt | Strikte Trennung in `KNOWN_FUTURE` (Demografie, UN-Projektionen) und `PAST_ONLY` (Investitionsquote, Handel, Inflation, gelaggt); Data Gate erzwingt identische Länderabdeckung über alle Modelle |
| 4 | Wenige und lückenhafte Daten je Land: viele Volkswirtschaften melden kurze oder unvollständige WDI-Reihen, wodurch länderspezifische Schätzungen instabil würden | Data Gate mit Mindestlänge von 35 Jahren gültigem Wachstum und mindestens 80 % Exogen-Abdeckung (116 von rund 217 WDI-Volkswirtschaften verbleiben); innere Lücken bis zu zwei Jahren werden interpoliert; globales Cross-Learning (TFT, Chronos-2) gleicht kurze Einzelhistorien zusätzlich aus |

## Definition of done

- [x] Slides completed
- [x] Documented, reproducible notebook(s) *(01–06, gemeinsame Module in `src/`, Offline-Smoke-Test in `tests/`)*
- [x] Honest evaluation, including comparison against the baseline *(Notebook 05: MASE, RMSE, Pinball, Coverage, Wilcoxon- und t-Tests, Segmentierung nach Region/Einkommensgruppe)*
- [x] At least one method is NEW (differs from the team's tutorial method)  *(TFT und Chronos-2; Team kannte bisher SARIMAX und N-BEATS)*
- [x] Chosen challenge(s) explicitly addressed  *(Long/Multi-Horizon, Exogene Treiber, Uncertainty Quantification)*
