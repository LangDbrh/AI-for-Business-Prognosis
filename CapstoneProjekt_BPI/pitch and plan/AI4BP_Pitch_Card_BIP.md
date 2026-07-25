---
title: "Pitch Card (One Idea)"
---

**Course: AI for Business Prognosis (AI4BP)**

**Name / Team:** Antoine Hunou, Marius Riesle, Deborah Lang — Team X-Team (Tutorial-Vorwissen: SARIMAX & N-BEATS)

# Pitch Card

*One idea, one glance. Use this to pitch a candidate project to your teammates.*

---

**Problem (one sentence)**

Wie entwickelt sich das reale BIP-Wachstum vieler Länder über die nächsten ein bis fünf Jahre? Ein einziges, global über alle Länder trainiertes Modell prognostiziert mehrere Jahre auf einmal, gestützt auf langsam veränderliche, im Voraus bekannte exogene Treiber, vor allem die Demografie.

**Business decision it supports**

Mittelfristige Haushalts-, Investitions- und Kapitalallokationsplanung. Wer Budgets oder Kapital über Länder verteilt, etwa Finanzministerien, IWF- oder OECD-artige Projektionen oder die strategische Asset-Allocation, handelt nach dem erwarteten Wachstum und dessen Unsicherheit.

**Challenge(s) tackled**  (pick at least one)

- [x] Long / multi-horizon
- [x] Uncertainty quantification  *(TFT und Chronos-2 liefern kalibrierbare 80%-Prognoseintervalle über q10/q90)*
- [x] Multivariate / multi-series (multi-value)
- [x] Exogenous drivers
- [ ] Intermittent / sparse demand

**Data**

Source: World Bank – World Development Indicators (WDI), Zugriff via Python-Paket `wbgapi`, Lizenz CC-BY-4.0. Zielgröße ist das reale BIP-Wachstum, exogene Treiber sind Bevölkerungswachstum und Erwerbsanteil (im Voraus bekannt, Demografie), Investitionsquote, Handelsoffenheit und Inflation (nur Vergangenheit bekannt, daher gelaggt). Nach dem Data Gate (Mindestlänge von 35 Jahren gültigem Wachstum, mindestens 80 Prozent Abdeckung der Treiber seit 1995) verbleiben 116 Volkswirtschaften von ursprünglich rund 217 WDI-Einheiten, jährlich von 1960 bis 2024.

Ready?  [x] yes   [ ] partly   [ ] no  — Panel aufbereitet, Data Gate bestanden, Walk-Forward-Folds fixiert.

**Reference paper (short)**

Laborda, Ruano & Zamanillo (2023), *„Multi-Country and Multi-Horizon GDP Forecasting …"* (Mathematics 11(12):2625), nutzt selbst TFT für Multi-Country-GDP und deckt sich damit mit unserer Hauptmethode. Diente als methodischer Referenzrahmen für Framing und Metrikwahl, nicht zur vollständigen Reproduktion.

**Proposed methods**

Baseline: Random Walk mit Drift auf Wachstumsraten, der ehrliche Makro-Anker, den es zu schlagen gilt.

Classical: ARIMA, SARIMA und SARIMAX je Land im direkten Methodenvergleich, das finale Modell SARIMAX nutzt zusätzlich die im Voraus bekannten Demografie-Regressoren.

Deep learning: TFT (Temporal Fusion Transformer), global über alle Länder trainiert, mit nativer Multi-Horizon-Prognose, exogenen Kovariaten und Quantil-Ausgabe. Ergänzt um einen Zero-Shot-Benchmark mit Chronos-2, einem vortrainierten Zeitreihen-Foundation-Model von Amazon Science mit Kovariaten-Unterstützung.

Which one is NEW to us:  [ ] classical   [x] deep learning   *(TFT als trainierte Hauptmethode und Chronos-2 als Zusatz-Benchmark; aus dem Tutorial bekannt waren SARIMAX und N-BEATS)*

**Why this is exciting / worth doing**

Makroökonomische BIP-Prognose ist gesellschaftlich hoch relevant und methodisch anspruchsvoll, weil einzelne Länderreihen kurze und starke Schwankungen aufweisen. Ein einziges globales TFT lernt über alle Länder hinweg (Cross-Learning) und mildert so das Problem zu kurzer Länderhistorien, und liefert statt einer reinen Punktprognose kalibrierte Unsicherheitsintervalle, was mittelfristige Wirtschaftsprognosen verbessert. Der Vergleich zeigt folgendes Ergebnis: Chronos-2 erreicht im Zero-Shot-Betrieb die global beste MASE von 0,845 und liegt damit praktisch gleichauf mit dem eigens angepassten SARIMAX (0,846), während das aufwendig trainierte TFT mit 0,922 dahinter zurückbleibt. Bei der Kalibrierung der 80-Prozent-Intervalle erreicht Chronos-2 eine Abdeckung von rund 84 Prozent, während das TFT mit rund 63 Prozent Abdeckung sein Risiko systematisch unterschätzt. Dieser Befund macht sichtbar, wie stark ein aktuelles Foundation-Model bereits mit spezialisierten, trainierten Verfahren mithalten kann.

**Biggest risk**

Langfristiges BIP-Wachstum liegt nahe am Random Walk und wird von Strukturbrüchen wie Finanzkrisen oder der Corona-Pandemie geprägt, sodass die Modelle die Baseline nur knapp schlagen könnten. Dieses Risiko hat sich im finalen, zuvor nicht für die Modellwahl genutzten Holdout 2020 bis 2024 bestätigt. Alle Verfahren rückten dort mit einer MASE zwischen 1,23 und 1,30 nah an die Baseline (1,29) heran. Die im Vorfeld gewählte Definition von Erfolg als kalibrierte Intervalle und das Übertreffen der Baseline statt einer perfekten Punktprognose hat sich dadurch als tragfähig erwiesen.

Ein zweites Risiko sind wenige und lückenhafte Daten je Land. Viele Volkswirtschaften melden kurze oder unvollständige WDI-Reihen, wodurch länderspezifische Schätzungen instabil würden. Das wurde minimiert durch ein Data Gate mit Mindestlänge von 35 Jahren gültigem Wachstum und mindestens 80 Prozent Exogen-Abdeckung (116 von rund 217 WDI-Volkswirtschaften verbleiben). Kleine innere Lücken bis zu zwei Jahren werden interpoliert und das globale Cross-Learning von TFT und Chronos-2 gleicht kurze Einzelhistorien zusätzlich aus.

---

## Listener feedback  (for teammates)

+------------------------------------+------------------------------------+
| One strength                       | One risk                           |
+====================================+====================================+
|                                    |                                    |
|                                    |                                    |
|                                    |                                    |
+------------------------------------+------------------------------------+
