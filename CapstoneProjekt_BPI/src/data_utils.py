"""Datenbeschaffung (World Bank WDI) und Panel-Aufbereitung.

Wird von Notebook 01 zum Erstellen des Panels genutzt und von den
Modell-Notebooks (02-04) zum Laden über load_panel() / train_test_split().
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_wdi(force: bool = False) -> pd.DataFrame:
    """Lädt alle Indikatoren aus config.INDICATORS für alle echten Länder
    (ohne Aggregate wie 'World' oder 'Euro area') und speichert sie als
    tidy Long-Format-CSV unter data/raw/.

    Nutzt das offizielle Python-Paket ``wbgapi``; falls es fehlt, wird auf
    die World-Bank-REST-API (JSON) zurückgegriffen. Bereits vorhandene
    Rohdaten werden wiederverwendet (Cache), außer force=True.

    Inkrementell: Kommt ein neuer Indikator in config.INDICATORS hinzu, werden
    beim Cache-Treffer NUR die fehlenden Indikatoren nachgeladen und angehängt,
    statt alle neu herunterzuladen. Ein einzelner Indikator kostet über die
    World-Bank-API rund 100 Sekunden, ein vollständiger Neu-Download ein
    Vielfaches davon.

    Returns
    -------
    DataFrame mit Spalten: country (ISO3), year, indicator (Panel-Name), value
    """
    all_codes = list(config.INDICATORS.keys())

    if config.RAW_WDI_CSV.exists() and not force:
        cached = pd.read_csv(config.RAW_WDI_CSV)
        have = set(cached["indicator"].unique())
        missing = [c for c in all_codes if config.INDICATORS[c][0] not in have]
        if not missing:
            print(f"Cache-Treffer: {config.RAW_WDI_CSV} (force=True zum Neuladen)")
            return cached

        # Nur die fehlenden Indikatoren nachladen und anhängen, nicht alles neu.
        missing_names = [config.INDICATORS[c][0] for c in missing]
        print(f"Cache unvollständig - lade nur fehlende Indikatoren nach: "
              f"{missing_names}")
        new = _download_indicators(missing)
        combined = (pd.concat([cached, new], ignore_index=True)
                    .sort_values(["country", "indicator", "year"],
                                 ignore_index=True))
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_csv(config.RAW_WDI_CSV, index=False)
        print(f"Aktualisiert: {config.RAW_WDI_CSV}  ({len(combined):,} Zeilen, "
              f"{combined['indicator'].nunique()} Indikatoren)")
        return combined

    raw = _download_indicators(all_codes)
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(config.RAW_WDI_CSV, index=False)
    print(f"Gespeichert: {config.RAW_WDI_CSV}  ({len(raw):,} Zeilen)")
    return raw


def _download_indicators(codes: list[str]) -> pd.DataFrame:
    """Lädt genau die angegebenen Indikator-Codes über wbgapi mit
    REST-API-Fallback. Gemeinsamer Pfad für Voll- und Teil-Downloads."""
    try:
        return _download_via_wbgapi(codes)
    except ImportError:
        print("wbgapi nicht installiert - nutze REST-API-Fallback ...")
        return _download_via_rest(codes)


def _download_via_wbgapi(codes: list[str] | None = None) -> pd.DataFrame:
    import wbgapi as wb  # noqa: F401 - Import prüft Verfügbarkeit

    codes = list(config.INDICATORS.keys()) if codes is None else list(codes)
    years = range(config.START_YEAR, config.LAST_YEAR + 1)

    print(f"Lade {len(codes)} Indikator(en), {config.START_YEAR}-{config.LAST_YEAR}, "
          f"alle Länder (ohne Aggregate) ...")
    df = wb.data.DataFrame(
        codes, economy="all", time=years,
        skipAggs=True,            # nur echte Volkswirtschaften
        columns="series",         # eine Spalte je Indikator
        numericTimeKeys=True,     # Jahres-Index als int statt 'YR1960'
    )
    # Index (economy, time) -> Spalten normalisieren
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "country", df.columns[1]: "year"})
    long = df.melt(id_vars=["country", "year"],
                   var_name="code", value_name="value")
    long["indicator"] = long["code"].map({c: n for c, (n, _) in config.INDICATORS.items()})
    long = long.dropna(subset=["indicator"])[["country", "year", "indicator", "value"]]
    long["year"] = long["year"].astype(int)
    return long.sort_values(["country", "indicator", "year"], ignore_index=True)


def _download_via_rest(codes: list[str] | None = None) -> pd.DataFrame:
    """Fallback ohne wbgapi: offizielle World-Bank-REST-API, paginiert."""
    import requests

    codes = list(config.INDICATORS.keys()) if codes is None else list(codes)
    frames = []
    date_range = f"{config.START_YEAR}:{config.LAST_YEAR}"
    for code in codes:
        name = config.INDICATORS[code][0]
        page, pages = 1, 1
        rows = []
        while page <= pages:
            r = requests.get(
                f"https://api.worldbank.org/v2/country/all/indicator/{code}",
                params={"format": "json", "per_page": 20000,
                        "page": page, "date": date_range},
                timeout=60,
            )
            r.raise_for_status()
            meta, data = r.json()
            pages = meta["pages"]
            rows.extend(data or [])
            page += 1
            time.sleep(0.3)  # API nicht fluten
        frames.append(pd.DataFrame({
            "country": [d["countryiso3code"] for d in rows],
            "year": [int(d["date"]) for d in rows],
            "indicator": name,
            "value": [d["value"] for d in rows],
        }))
        print(f"  {code} -> {name}: {len(frames[-1]):,} Zeilen")
    long = pd.concat(frames, ignore_index=True)
    # REST liefert auch Aggregate -> über Länder-Metadaten filtern
    meta = download_country_meta()
    long = long[long["country"].isin(meta["country"])]
    return long.sort_values(["country", "indicator", "year"], ignore_index=True)


def download_country_meta(force: bool = False) -> pd.DataFrame:
    """Statische Länder-Metadaten: Name, Region, Einkommensgruppe."""
    if config.RAW_META_CSV.exists() and not force:
        return pd.read_csv(config.RAW_META_CSV)

    try:
        import wbgapi as wb
        eco = wb.economy.DataFrame(skipAggs=True, labels=True).reset_index()
        cols = {c.lower(): c for c in eco.columns}
        meta = pd.DataFrame({
            "country": eco[cols.get("id", "id")],
            "name": eco[cols.get("name", "name")],
            "region": eco[cols.get("region", "region")],
            "income_group": eco[cols.get("incomelevel", "incomeLevel")],
        })
    except ImportError:
        import requests
        r = requests.get("https://api.worldbank.org/v2/country",
                         params={"format": "json", "per_page": 400}, timeout=60)
        r.raise_for_status()
        data = r.json()[1]
        rows = [d for d in data if d["region"]["value"] != "Aggregates"]
        meta = pd.DataFrame({
            "country": [d["id"] for d in rows],
            "name": [d["name"] for d in rows],
            "region": [d["region"]["value"] for d in rows],
            "income_group": [d["incomeLevel"]["value"] for d in rows],
        })

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    meta.to_csv(config.RAW_META_CSV, index=False)
    return meta


# ---------------------------------------------------------------------------
# Aufbereitung
# ---------------------------------------------------------------------------
def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Long-Format -> breites Panel: eine Zeile je (country, year),
    eine Spalte je Indikator."""
    panel = (raw.pivot_table(index=["country", "year"], columns="indicator",
                             values="value", aggfunc="first")
                .reset_index()
                .rename_axis(columns=None)
                .sort_values(["country", "year"], ignore_index=True))
    return panel


def filter_countries(panel: pd.DataFrame,
                     min_years: int = config.MIN_YEARS,
                     require_recent: bool = True,
                     min_exog_coverage: float = config.MIN_EXOG_COVERAGE,
                     exog_since: int = config.EXOG_SINCE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Behält nur Länder mit ausreichend langer, aktueller BIP-Historie
    UND brauchbarer Abdeckung der exogenen Treiber.

    Kriterien:
      1. mindestens ``min_years`` Jahre mit gültigem BIP-Wachstum
      2. gültiger Wert im letzten Datenjahr (require_recent)
      3. JEDE Treiber-Variable hat ab ``exog_since`` mindestens
         ``min_exog_coverage`` Abdeckung (sonst fehlen z.B. Mikrostaaten
         im Testfenster die Regressoren, und die Modelle würden
         stillschweigend unterschiedliche Ländermengen prognostizieren)

    Returns
    -------
    (gefiltertes Panel, Report-DataFrame mit Behalten/Verworfen je Land)
    """
    g = panel.dropna(subset=[config.TARGET]).groupby("country")[config.TARGET]
    n_years = g.count()
    last_valid = (panel.dropna(subset=[config.TARGET])
                       .groupby("country")["year"].max())

    keep = n_years[n_years >= min_years].index
    if require_recent:
        recent = last_valid[last_valid >= config.LAST_YEAR - 1].index
        keep = keep.intersection(recent)

    # Exogen-Abdeckung ab exog_since: Minimum über alle Treiber-Spalten
    exog_cols = [c for c in config.KNOWN_FUTURE + config.PAST_ONLY
                 if c in panel.columns]
    cov = (panel[panel["year"] >= exog_since]
           .groupby("country")[exog_cols]
           .apply(lambda d: d.notna().mean().min()))
    if min_exog_coverage is not None and exog_cols:
        keep = keep.intersection(cov[cov >= min_exog_coverage].index)

    report = pd.DataFrame({
        "n_years": n_years,
        "last_valid_year": last_valid,
        "exog_coverage": cov.round(2),
    })
    report["kept"] = report.index.isin(keep)

    filtered = panel[panel["country"].isin(keep)].copy()
    return filtered, report.sort_values("n_years", ascending=False)


def interpolate_gaps(panel: pd.DataFrame,
                     columns: list[str] | None = None,
                     max_gap: int = config.MAX_GAP_INTERPOLATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Interpoliert nur KLEINE innere Lücken (<= max_gap Jahre) linear,
    getrennt je Land. Ränder werden NICHT extrapoliert, das Ziel
    (gdp_growth) wird NIE imputiert.

    Returns
    -------
    (Panel, Report mit Anzahl gefüllter Werte je Spalte)
    """
    if columns is None:
        columns = [c for c in config.KNOWN_FUTURE + config.PAST_ONLY
                   if c in panel.columns]

    panel = panel.sort_values(["country", "year"]).copy()
    filled = {}
    for col in columns:
        before = panel[col].isna().sum()
        panel[col] = (panel.groupby("country")[col]
                      .transform(lambda s: s.interpolate(
                          method="linear", limit=max_gap,
                          limit_area="inside")))
        filled[col] = int(before - panel[col].isna().sum())
    report = pd.DataFrame.from_dict(filled, orient="index",
                                    columns=["values_filled"])
    return panel, report


def add_lagged_features(panel: pd.DataFrame,
                        lag: int = config.LAG_YEARS) -> pd.DataFrame:
    """Erzeugt für alle PAST_ONLY-Variablen gelaggte Spalten (_lag1)
    und ENTFERNT die kontemporären Spalten - so ist Leakage über
    unbekannte Zukunftswerte konstruktiv ausgeschlossen.

    Transformationen vor dem Laggen:
      * gdp_pc     -> log(gdp_pc)                    (Konvergenz-Feature)
      * inflation  -> sign(x) * log1p(|x|)           (staucht Hyperinflation:
        23.000% -> ~10, normale 3% -> ~1.4; ohne das dominieren Ausreißer
        wie Bolivien 1985 jede Feature-Skalierung im TFT)
    """
    panel = panel.sort_values(["country", "year"]).copy()

    if "gdp_pc" in panel.columns:
        panel["log_gdp_pc"] = np.log(panel["gdp_pc"])
    if "inflation" in panel.columns:
        panel["inflation_log"] = (np.sign(panel["inflation"])
                                  * np.log1p(panel["inflation"].abs()))

    rename = {"gdp_pc": "log_gdp_pc", "inflation": "inflation_log"}
    lag_sources = [rename.get(c, c) for c in config.PAST_ONLY
                   if c in panel.columns]

    for col in lag_sources:
        panel[f"{col}_lag{lag}"] = panel.groupby("country")[col].shift(lag)

    # Kontemporäre PAST_ONLY-Spalten entfernen (inkl. Hilfsspalten) sowie
    # reine Referenz-/Viz-Spalten (z.B. population) - sie sind KEINE
    # Modell-Features und sollen das Modell-Panel nicht aufblähen.
    drop = ([c for c in config.PAST_ONLY if c in panel.columns]
            + ["log_gdp_pc", "inflation_log"]
            + [c for c in config.REFERENCE_ONLY if c in panel.columns])
    panel = panel.drop(columns=[c for c in drop if c in panel.columns])
    return panel


def add_crisis_dummy(panel: pd.DataFrame,
                     crisis_years: tuple[int, ...] = (2009, 2020)) -> pd.DataFrame:
    """Dummy für globale Krisenjahre (Finanzkrise, COVID).

    ACHTUNG: Für ZUKÜNFTIGE Jahre ist eine Krise unbekannt -> bei der
    Prognose immer 0 setzen. Historisch hilft der Dummy den Modellen,
    Extremjahre zu erklären, statt sie als Rauschen zu lernen.
    """
    panel = panel.copy()
    panel["is_global_crisis"] = panel["year"].isin(crisis_years).astype(int)
    return panel


# ---------------------------------------------------------------------------
# Schnittstelle für die Modell-Notebooks (02-04)
# ---------------------------------------------------------------------------
def load_panel() -> pd.DataFrame:
    """Lädt das fertige Panel aus data/processed/ (von Notebook 01 erzeugt)."""
    if not config.PANEL_CSV.exists():
        raise FileNotFoundError(
            f"{config.PANEL_CSV} fehlt - bitte zuerst Notebook "
            "01_Datenbeschaffung_Aufbereitung.ipynb ausführen.")
    return pd.read_csv(config.PANEL_CSV)


def load_panel_meta() -> dict:
    with open(config.PANEL_META_JSON, encoding="utf-8") as f:
        return json.load(f)


def train_test_split(panel: pd.DataFrame, origin_year: int,
                     horizon: int = config.HORIZON) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zeitlicher Split (expanding window): Training bis einschließlich
    origin_year, Test = die ``horizon`` Folgejahre. NIEMALS shufflen!"""
    train = panel[panel["year"] <= origin_year].copy()
    test = panel[(panel["year"] > origin_year)
                 & (panel["year"] <= origin_year + horizon)].copy()
    return train, test


def iter_backtest_folds(panel: pd.DataFrame):
    """Generator über alle Backtest-Folds inkl. finalem Holdout:
    ``for origin, train, test in iter_backtest_folds(panel): ...``"""
    for origin in config.ALL_ORIGINS:
        train, test = train_test_split(panel, origin)
        yield origin, train, test


# ---------------------------------------------------------------------------
# Weltaggregat (Notebook 04 Abschnitt 6, Notebook 05)
# ---------------------------------------------------------------------------
def build_world_panel(panel: pd.DataFrame | None = None
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baut die BIP-gewichtete Weltreihe aus den WDI-Rohdaten.

    Wachstumsraten dürfen NICHT ungewichtet gemittelt werden - Tuvalu zählte
    sonst so viel wie die USA. Die Weltreihe entsteht deshalb aus BIP-Niveaus::

        gdp_level = gdp_pc * population        (real, konst. 2015 US$)
        g_welt(t) = sum_i BIP(i,t) / sum_i BIP(i,t-1) - 1

    Beide Niveauspalten entfernt add_lagged_features() aus dem Modell-Panel
    (sie sind REFERENCE_ONLY bzw. werden gelaggt), deshalb wird hier direkt
    config.RAW_WDI_CSV gelesen.

    Entscheidend ist die PAARWEISE Stichprobe: In die Summe eines Jahres gehen
    nur Länder ein, die sowohl in t-1 als auch in t ein BIP-Niveau haben. Sonst
    erschiene der bloße Eintritt eines Landes in die Datenbasis als
    Wachstumssprung. Ab config.WORLD_COMPLETE_SINCE liegen für alle Länder
    Werte vor; in den Testfenstern der Backtests ist die Zusammensetzung also
    ohnehin konstant.

    Die Kovariaten werden mit der jeweils passenden Gewichtung aggregiert:
    Bevölkerungswachstum aus der Summe der Bevölkerung, working_age_share
    bevölkerungsgewichtet, investment_gdp und trade_gdp BIP-gewichtet. Die
    Inflation wird ERST je Land gedämpft (sign(x)*log1p(|x|), wie in
    add_lagged_features) und DANN BIP-gewichtet gemittelt - andernfalls
    dominierte eine einzelne Hyperinflation den Weltwert.

    Parameters
    ----------
    panel : Modell-Panel aus load_panel(); legt die Ländermenge fest.
        None -> load_panel().

    Returns
    -------
    (world_panel, gdp_levels)
        world_panel: eine Zeile je Jahr mit exakt den Spaltennamen des
          Länderpanels (country=config.WORLD_ID, year, config.TARGET, die
          KNOWN_FUTURE- und gelaggten PAST_ONLY-Treiber, is_global_crisis),
          zusätzlich n_countries und world_gdp zur Diagnose. Dadurch läuft die
          Weltreihe ohne Anpassung durch dieselben Fold-/Backtest-Funktionen
          wie ein Länderpanel.
        gdp_levels: reales BIP je Jahr (Index) und Land (Spalten); wird für
          die Bottom-up-Aggregation in evaluation.aggregate_bottom_up()
          gebraucht.
    """
    panel = load_panel() if panel is None else panel
    raw = pd.read_csv(config.RAW_WDI_CSV)
    countries = sorted(panel["country"].unique())

    wide = (raw.pivot_table(index=["country", "year"], columns="indicator",
                            values="value", aggfunc="first")
               .reset_index().rename_axis(columns=None))
    wide = wide[wide["country"].isin(countries)].sort_values(["country", "year"])
    wide["gdp_level"] = wide["gdp_pc"] * wide["population"]
    # Inflation je Land dämpfen, BEVOR sie gewichtet gemittelt wird.
    wide["inflation_log"] = (np.sign(wide["inflation"])
                             * np.log1p(wide["inflation"].abs()))

    gdp = wide.pivot(index="year", columns="country", values="gdp_level").sort_index()
    pop = wide.pivot(index="year", columns="country", values="population").sort_index()

    rows = []
    for year in gdp.index[1:]:
        gdp_now, gdp_prev = gdp.loc[year], gdp.loc[year - 1]
        pop_now, pop_prev = pop.loc[year], pop.loc[year - 1]
        m_gdp = gdp_now.notna() & gdp_prev.notna()
        m_pop = pop_now.notna() & pop_prev.notna()
        rows.append({
            "year": int(year),
            config.TARGET: (gdp_now[m_gdp].sum() / gdp_prev[m_gdp].sum() - 1) * 100,
            "pop_growth": (pop_now[m_pop].sum() / pop_prev[m_pop].sum() - 1) * 100,
            "n_countries": int(m_gdp.sum()),
        })
    world = pd.DataFrame(rows)

    def _weighted_mean(group: pd.DataFrame, col: str, weight: str) -> float:
        d = group[[col, weight]].dropna()
        if d.empty or d[weight].sum() == 0:
            return float("nan")
        return float(np.average(d[col], weights=d[weight]))

    levels = []
    for year, g in wide.groupby("year"):
        levels.append({
            "year": int(year),
            "world_gdp": g["gdp_level"].sum(min_count=1),
            "world_pop": g["population"].sum(min_count=1),
            "working_age_share": _weighted_mean(g, "working_age_share", "population"),
            "investment_gdp": _weighted_mean(g, "investment_gdp", "gdp_level"),
            "trade_gdp": _weighted_mean(g, "trade_gdp", "gdp_level"),
            "inflation_log": _weighted_mean(g, "inflation_log", "gdp_level"),
        })
    levels = pd.DataFrame(levels).sort_values("year")
    levels["log_gdp_pc"] = np.log(levels["world_gdp"] / levels["world_pop"])

    world = world.merge(levels, on="year", how="left").sort_values("year")
    # Past-only-Treiber genau wie im Länderpanel um LAG_YEARS laggen.
    for col in ["log_gdp_pc", "investment_gdp", "trade_gdp", "inflation_log"]:
        world[f"{col}_lag{config.LAG_YEARS}"] = world[col].shift(config.LAG_YEARS)

    world["country"] = config.WORLD_ID
    world = add_crisis_dummy(world)

    lagged = [f"{c}_lag{config.LAG_YEARS}" for c in
              ["log_gdp_pc", "investment_gdp", "trade_gdp", "inflation_log"]]
    keep = (["country", "year", config.TARGET] + config.KNOWN_FUTURE + lagged
            + ["is_global_crisis", "n_countries", "world_gdp"])
    return world[keep].reset_index(drop=True), gdp


# ---------------------------------------------------------------------------
# Visualisierung (Notebook 06): kontemporäres Panel für die Gapminder-Animation
# ---------------------------------------------------------------------------
def build_viz_panel(raw: pd.DataFrame, static: pd.DataFrame,
                    only_modeling_countries: bool = True) -> pd.DataFrame:
    """Baut ein Panel mit KONTEMPORÄREN Werten für die Bubble-Animation.

    Anders als das Modell-Panel werden hier KEINE Lags gebildet und die
    kontemporären Spalten (gdp_pc, working_age_share, population, ...)
    bleiben erhalten - genau das braucht die Animation je Jahr.

    Parameters
    ----------
    raw : Long-Format aus download_wdi() (muss 'population' enthalten,
          d.h. SP.POP.TOTL muss in config.INDICATORS stehen).
    static : Länder-Metadaten aus download_country_meta() (Region etc.).
    only_modeling_countries : True -> dieselbe gefilterte 116-Länder-Menge
          wie die Modelle (Projekt-konsistent); False -> alle Länder, die
          die nötigen Viz-Spalten haben (vollere Rosling-Optik).

    Returns
    -------
    DataFrame mit country, name, region, income_group, year und den
    kontemporären Kennzahlen (u.a. gdp_pc, working_age_share, population).
    """
    panel = build_panel(raw)

    if only_modeling_countries:
        panel, _ = filter_countries(panel)
    else:
        # Nur Länder mit einer minimal brauchbaren BIP-Historie behalten
        n_years = (panel.dropna(subset=[config.TARGET])
                        .groupby("country")[config.TARGET].count())
        keep = n_years[n_years >= config.MIN_YEARS].index
        panel = panel[panel["country"].isin(keep)].copy()

    # kleine innere Lücken der Viz-Spalten glätten (ruhigere Animation)
    viz_cols = [c for c in ["gdp_pc", "working_age_share", "population",
                            "dependency_ratio", "pop_growth"]
                if c in panel.columns]
    panel, _ = interpolate_gaps(panel, columns=viz_cols)

    out = panel.merge(static[["country", "name", "region", "income_group"]],
                      on="country", how="left")
    return out.sort_values(["country", "year"], ignore_index=True)
