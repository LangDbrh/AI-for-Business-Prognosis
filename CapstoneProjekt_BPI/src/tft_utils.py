"""Pipeline für den Temporal Fusion Transformer in Notebook 03.

Bündelt Merkmalsrollen, Vorkehrungen gegen das Einfließen zukünftiger
Information, Training und Vorhersage, damit das Notebook selbst auf
Methodik und Ergebnisse fokussiert bleiben kann. Dieses Modul folgt
derselben Konvention wie die übrigen Module unter src. Alle Konstanten
stammen ausschließlich aus config.py.

Merkmalsrollen, vergleiche Notebook 01, Abschnitt zur Auswahl der Daten.
Die Gruppe time_varying_known_reals entspricht config.KNOWN_FUTURE
zusammen mit dem Krisendummy. Diese Werte gelten bei der Prognose für
Jahre nach dem Origin als real bekannt beziehungsweise werden auf 0
gesetzt, siehe mask_future_crisis. Die Gruppe time_varying_unknown_reals
entspricht dem Ziel sowie den vier Treibern mit der Endung lag1. Diese
Variablen werden von pytorch forecasting ausschließlich im Encoder
verwendet, niemals im Decoder. Das ist die konservative, gegen das
Einfließen zukünftiger Information abgesicherte Lesart von Regel 3 aus
Notebook 01, wonach die gelaggten Variablen nur so weit verwendet werden
dürfen, wie sie zum jeweiligen Zeitpunkt tatsächlich verfügbar waren. Die
Gruppe static_categoricals umfasst region und income_group.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss

from . import config, evaluation

SEED = 42

# ===========================================================================
# Merkmalsrollen, fest aus config.py abgeleitet, siehe Modulbeschreibung oben.
# ===========================================================================
LAG_COLS = [f"log_gdp_pc_lag{config.LAG_YEARS}",
            f"investment_gdp_lag{config.LAG_YEARS}",
            f"trade_gdp_lag{config.LAG_YEARS}",
            f"inflation_log_lag{config.LAG_YEARS}"]
MISSING_FLAG_COLS = [f"{c}_missing" for c in LAG_COLS]
STATIC_CATEGORICALS = ["region", "income_group"]
KNOWN_REALS = ["time_idx", *config.KNOWN_FUTURE, "is_global_crisis"]
UNKNOWN_REALS = [config.TARGET, *LAG_COLS, *MISSING_FLAG_COLS]


def validate_feature_roles() -> None:
    """Prüft, dass keine Variable gleichzeitig als bekannt zukünftig und als
    nur vergangen deklariert ist. Diese Prüfung läuft beim Import des
    Moduls und schützt vor Widersprüchen beim Erweitern der
    Merkmalslisten."""
    overlap = set(KNOWN_REALS) & set(UNKNOWN_REALS)
    if overlap:
        raise ValueError(
            f"Es besteht ein Risiko des Einfließens zukünftiger Information. "
            f"Die folgenden Variablen sind in beiden Rollen deklariert {sorted(overlap)}")


validate_feature_roles()


# ===========================================================================
# Datenaufbereitung je Fold
# ===========================================================================
def add_model_features(panel: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """Ergänzt statische Ländermerkmale und einen fortlaufenden Zeitindex.

    time_idx muss für TimeSeriesDataSet ein Integer ohne Bezug zum
    Kalenderjahr sein, mit dem Wert 0 für config.START_YEAR. Die
    Rückrechnung auf Kalenderjahre erfolgt in predictions_to_results.
    """
    df = panel.merge(static[["country", "region", "income_group"]],
                      on="country", how="left")
    df["time_idx"] = df["year"] - config.START_YEAR
    df["region"] = df["region"].astype(str)
    df["income_group"] = df["income_group"].astype(str)
    return df


def mask_future_crisis(df: pd.DataFrame, origin_year: int) -> pd.DataFrame:
    """Setzt is_global_crisis für alle Jahre nach dem Origin auf 0, gemäß
    Regel 4 aus Notebook 01. Ein Forecaster im Jahr origin_year kennt keine
    zukünftigen Krisen. Konkret betrifft dies den finalen Holdout Fold mit
    Origin 2019, dessen Testfenster von 2020 bis 2024 das reale COVID Jahr
    2020 enthält. Ohne diese Maskierung würde der Decoder die Krise bereits
    kennen, bevor sie eingetreten ist."""
    df = df.copy()
    df.loc[df["year"] > origin_year, "is_global_crisis"] = 0
    return df


def prepare_fold(panel: pd.DataFrame, origin_year: int,
                  horizon: int = config.HORIZON) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bereitet einen Backtest Fold vor, bestehend aus Trainingsfenster und
    Testfenster.

    Die Funktion maskiert zukünftige Krisenjahre über mask_future_crisis,
    füllt fehlende Werte bei den Treibern mit der Endung lag1 mit dem
    Median aus dem Trainingsfenster auf, niemals aus Test oder
    Zukunftsdaten, da sonst die Verteilung der Zukunft in die Aufbereitung
    der Vergangenheit einfließen würde, und ergänzt je Spalte eine
    Kennzeichnung fehlender Werte statt Zeilen zu verwerfen. Zeilen ohne
    gültigen Zielwert werden entfernt, da sie weder trainiert noch
    bewertet werden können.

    Die Funktion gibt ein Tupel aus train_df und fold_df zurück. train_df
    enthält nur Jahre bis einschließlich origin_year. fold_df enthält
    zusätzlich das Testfenster für die Vorhersage, mit echten historischen
    Werten der Exogenen und ohne Platzhalter.
    """
    fold = panel[panel["year"] <= origin_year + horizon].copy()
    fold = mask_future_crisis(fold, origin_year)

    train_mask = fold["year"] <= origin_year
    medians = fold.loc[train_mask, LAG_COLS].median()
    for c in LAG_COLS:
        fold[f"{c}_missing"] = fold[c].isna().astype(int)
        fold[c] = fold[c].fillna(medians[c])

    fold = fold.dropna(subset=[config.TARGET])
    train_df = fold[fold["year"] <= origin_year].copy()
    return train_df, fold


# ===========================================================================
# TimeSeriesDataSet und Modell
# ===========================================================================
def make_training_dataset(df: pd.DataFrame, max_encoder_length: int) -> TimeSeriesDataSet:
    """Baut das TimeSeriesDataSet mit den festen Merkmalsrollen des
    Projekts.

    Die categorical_encoders werden mit add_nan gleich True erzeugt, weil
    einzelne Länder, etwa Nachfolgestaaten der Sowjetunion wie Russland,
    erst mitten im Panel auftauchen. Ein Land, das im kleineren
    Trainingsfenster einer zweistufigen Fold Konstruktion noch fehlt, aber
    im größeren Fenster für die interne Validierung oder das finale
    Training bereits Daten besitzt, würde beim Kodieren der Kategorien
    sonst zu einem Abbruch mit einem Fehler wegen einer unbekannten
    Kategorie führen. Mit add_nan gleich True wird eine bislang unbekannte
    Kategorie stattdessen als unbekannt kodiert, statt das gesamte
    Training des Folds abzubrechen.
    """
    encoder = lambda: NaNLabelEncoder(add_nan=True)
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=config.TARGET,
        group_ids=["country"],
        max_encoder_length=max_encoder_length,
        min_encoder_length=max(3, max_encoder_length // 2),
        max_prediction_length=config.HORIZON,
        min_prediction_length=config.HORIZON,
        static_categoricals=STATIC_CATEGORICALS,
        time_varying_known_reals=KNOWN_REALS,
        time_varying_unknown_reals=UNKNOWN_REALS,
        target_normalizer=GroupNormalizer(groups=["country"]),
        categorical_encoders={"country": encoder(), **{c: encoder() for c in STATIC_CATEGORICALS}},
        add_relative_time_idx=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )


def build_model(training: TimeSeriesDataSet, hp: dict[str, Any]) -> TemporalFusionTransformer:
    """Erzeugt einen Temporal Fusion Transformer aus einem TimeSeriesDataSet
    und einer Hyperparameterkonfiguration."""
    return TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=hp.get("learning_rate", 0.03),
        hidden_size=hp.get("hidden_size", 16),
        attention_head_size=hp.get("attention_head_size", 2),
        dropout=hp.get("dropout", 0.15),
        hidden_continuous_size=hp.get("hidden_continuous_size", 8),
        loss=QuantileLoss(config.QUANTILES),
        log_interval=0,
        optimizer="adam",
    )


def fit_model(model: TemporalFusionTransformer, train_dataset: TimeSeriesDataSet,
              val_dataset: TimeSeriesDataSet, max_epochs: int = 60, patience: int = 6,
              batch_size: int = 64, seed: int = SEED, enable_progress_bar: bool = False
              ) -> tuple[pl.Trainer, EarlyStopping]:
    """Trainiert ein Modell mit Early Stopping auf val_dataset. Gibt den
    Trainer für den Zugriff auf die aktuelle Epoche sowie den Early
    Stopping Callback für die tatsächlich beste Epoche zurück."""
    pl.seed_everything(seed, verbose=False)
    train_dl = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dl = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    early_stop = EarlyStopping(monitor="val_loss", patience=patience, mode="min")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gradient_clip_val=0.1,
        callbacks=[early_stop],
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    return trainer, early_stop


@dataclass
class FoldResult:
    origin_year: int
    hp: dict[str, Any]
    best_epoch: int
    val_loss: float
    model: TemporalFusionTransformer
    predict_dataset: TimeSeriesDataSet
    fold_df: pd.DataFrame


def run_fold(panel: pd.DataFrame, origin_year: int, hp: dict[str, Any],
             max_epochs: int = 60, patience: int = 6, seed: int = SEED,
             refit_on_full_train: bool = True) -> FoldResult:
    """Trainiert und evaluiert einen Backtest Fold im Walk Forward Verfahren
    mit expandierendem Fenster, gemäß dem Standardablauf des Projekts.

    Das Training läuft zweistufig. In der ersten Stufe erfolgt ein Fit auf
    den Trainingsjahren ohne die letzten HORIZON Jahre, mit Early Stopping
    auf genau diesen zurückgehaltenen Jahren als interner Validierung. Das
    simuliert die eigentliche Aufgabe des Folds, ohne die realen Test und
    Zukunftsjahre zu berühren. In der zweiten Stufe wird das Modell für die
    so ermittelte Anzahl an Epochen auf dem vollständigen Trainingsfenster
    neu trainiert, ohne weiteres Early Stopping. Andernfalls blieben die
    jüngsten HORIZON Trainingsjahre systematisch ungenutzt. Mit
    refit_on_full_train gleich False lässt sich dieser zweite Schritt für
    schnelle Testläufe überspringen.
    """
    encoder_len = hp["max_encoder_length"]
    train_df, fold_df = prepare_fold(panel, origin_year)

    val_cutoff = train_df["year"].max() - config.HORIZON
    fit_df = train_df[train_df["year"] <= val_cutoff]
    fit_dataset = make_training_dataset(fit_df, encoder_len)
    val_dataset = TimeSeriesDataSet.from_dataset(
        fit_dataset, train_df, predict=True, stop_randomization=True)

    pl.seed_everything(seed, verbose=False)
    model = build_model(fit_dataset, hp)
    trainer, early_stop = fit_model(model, fit_dataset, val_dataset,
                                     max_epochs=max_epochs, patience=patience, seed=seed)
    best_epoch = trainer.current_epoch - early_stop.wait_count
    val_loss = float(early_stop.best_score)

    if refit_on_full_train and best_epoch > 0:
        full_dataset = TimeSeriesDataSet.from_dataset(
            fit_dataset, train_df, predict=False, stop_randomization=False)
        pl.seed_everything(seed, verbose=False)
        model = build_model(full_dataset, hp)
        pl.seed_everything(seed, verbose=False)
        train_dl = full_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
        trainer = pl.Trainer(
            max_epochs=max(best_epoch, 1), gradient_clip_val=0.1,
            enable_progress_bar=False, enable_model_summary=False,
            logger=False, enable_checkpointing=False,
            deterministic=True,
        )
        trainer.fit(model, train_dataloaders=train_dl)
        base_dataset = full_dataset
    else:
        base_dataset = fit_dataset

    predict_dataset = TimeSeriesDataSet.from_dataset(
        base_dataset, fold_df, predict=True, stop_randomization=True)

    return FoldResult(origin_year=origin_year, hp=hp, best_epoch=max(best_epoch, 1),
                       val_loss=val_loss, model=model, predict_dataset=predict_dataset,
                       fold_df=fold_df)


# ===========================================================================
# Vorhersagen in das Standard Ergebnisschema, config.RESULTS_COLUMNS
# ===========================================================================
def predictions_to_results(fold: FoldResult, model_name: str = "tft") -> pd.DataFrame:
    """Wandelt Quantilsprognosen für einen Fold in das Standardschema um,
    identisch zu src/evaluation.py für alle Modell Notebooks."""
    dl = fold.predict_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
    raw = fold.model.predict(dl, mode="quantiles", return_index=True)
    out = raw.output.numpy()  # Form ist Anzahl Serien, Horizont, Anzahl Quantile
    idx = raw.index  # Enthaelt country und time_idx als ersten Decoder Schritt

    q_lo, q_hi = config.QUANTILES[0], config.QUANTILES[-1]
    q_mid_pos = config.QUANTILES.index(0.5)

    rows = []
    for i, row in idx.reset_index(drop=True).iterrows():
        country = row["country"]
        start_time_idx = int(row["time_idx"])
        for h in range(config.HORIZON):
            year = config.START_YEAR + start_time_idx + h
            rows.append({
                "model": model_name,
                "country": country,
                "origin_year": fold.origin_year,
                "year": year,
                "h": h + 1,
                "y_pred": float(out[i, h, q_mid_pos]),
                "q10": float(out[i, h, config.QUANTILES.index(q_lo)]),
                "q90": float(out[i, h, config.QUANTILES.index(q_hi)]),
            })
    preds = pd.DataFrame(rows)

    truth = fold.fold_df[["country", "year", config.TARGET]].rename(
        columns={config.TARGET: "y_true"})
    preds = preds.merge(truth, on=["country", "year"], how="left")
    return preds[config.RESULTS_COLUMNS]


# ===========================================================================
# Hyperparameter Suche, einmalig auf einem Fenster vor dem ersten Backtest
# Origin, damit keine echten Testjahre in die Modellwahl einfliessen.
# ===========================================================================
def search_hyperparameters(panel: pd.DataFrame, configs: list[dict[str, Any]],
                            tuning_origin: int, max_epochs: int = 20,
                            patience: int = 5, seed: int = SEED) -> pd.DataFrame:
    """Trainiert jede Konfiguration einmal auf tuning_origin, mit interner
    Validierung und ohne erneutes Training auf dem vollen Fenster, und
    misst MASE auf dieser Validierung. tuning_origin sollte vor
    config.BACKTEST_ORIGINS[0] liegen, damit kein offizieller Test Fold
    berührt wird."""
    train_df, _ = prepare_fold(panel, tuning_origin)
    val_cutoff = train_df["year"].max() - config.HORIZON
    fit_df = train_df[train_df["year"] <= val_cutoff]

    results = []
    for i, hp in enumerate(configs):
        fit_dataset = make_training_dataset(fit_df, hp["max_encoder_length"])
        val_dataset = TimeSeriesDataSet.from_dataset(
            fit_dataset, train_df, predict=True, stop_randomization=True)
        pl.seed_everything(seed, verbose=False)
        model = build_model(fit_dataset, hp)
        trainer, early_stop = fit_model(model, fit_dataset, val_dataset,
                                         max_epochs=max_epochs, patience=patience, seed=seed)

        val_dl = val_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
        raw = model.predict(val_dl, mode="quantiles", return_index=True)
        out = raw.output.numpy()
        idx = raw.index.reset_index(drop=True)
        # MASE wird je Land berechnet und danach gemittelt, wie in
        # evaluation.summarize(), damit einzelne hochvolatile Laender die
        # Modellwahl nicht dominieren.
        country_mases = []
        for j, row in idx.iterrows():
            country, start_t = row["country"], int(row["time_idx"])
            years = [config.START_YEAR + start_t + h for h in range(config.HORIZON)]
            truth = (train_df[train_df["country"] == country]
                     .set_index("year")[config.TARGET])
            y_pred_c = out[j, :, config.QUANTILES.index(0.5)]
            y_true_c = [truth.get(y, np.nan) for y in years]
            y_train_c = train_df.loc[(train_df["country"] == country)
                                      & (train_df["year"] <= val_cutoff), config.TARGET]
            country_mases.append(evaluation.mase(y_true_c, y_pred_c, y_train_c))

        val_mase = float(np.nanmean(country_mases))
        results.append({
            "config_id": i, **hp,
            "best_epoch": trainer.current_epoch - early_stop.wait_count,
            "val_loss": float(early_stop.best_score),
            "val_MASE": val_mase,
        })
    return pd.DataFrame(results).sort_values("val_MASE").reset_index(drop=True)


# ===========================================================================
# Diagnostik fuer den Notebook Abschnitt Ergebnisse und Diagnostik.
# ===========================================================================
def compute_row_mase(preds: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Ergänzt je Kombination aus Land und Origin Jahr den MASE, skaliert
    mit dem im Trainingsfenster gemessenen Fehler des naiven Modells für
    einen Schritt bis origin_year, siehe evaluation.summarize(). Der Wert
    ist über die Zeilen für h gleich 1 bis HORIZON einer Kombination aus
    Land und Origin Jahr konstant."""
    rows = []
    for (country, origin), g in preds.groupby(["country", "origin_year"]):
        y_train = panel.loc[(panel["country"] == country)
                             & (panel["year"] <= origin), config.TARGET]
        rows.append({"country": country, "origin_year": origin,
                      "MASE": evaluation.mase(g["y_true"], g["y_pred"], y_train)})
    return preds.merge(pd.DataFrame(rows), on=["country", "origin_year"], how="left")


def summarize_by_origin(preds: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Berechnet Metriken je Backtest Origin und zeigt, ob die
    Prognosegüte über die Zeit schwankt, was insbesondere für den finalen
    COVID Holdout relevant ist."""
    preds = compute_row_mase(preds, panel)
    rows = []
    for origin, g in preds.groupby("origin_year"):
        rows.append({
            "origin_year": int(origin),
            "test_years": f"{origin + 1} bis {origin + config.HORIZON}",
            "MASE": g.drop_duplicates(["country", "origin_year"])["MASE"].mean(),
            "RMSE": evaluation.rmse(g["y_true"], g["y_pred"]),
            "MAE": evaluation.mae(g["y_true"], g["y_pred"]),
            "pinball_q10": evaluation.pinball_loss(g["y_true"], g["q10"], config.QUANTILES[0]),
            "pinball_q90": evaluation.pinball_loss(g["y_true"], g["q90"], config.QUANTILES[-1]),
            "coverage_80": evaluation.interval_coverage(g["y_true"], g["q10"], g["q90"]),
            "n_forecasts": len(g),
        })
    return pd.DataFrame(rows)


def summarize_by_horizon(preds: pd.DataFrame) -> pd.DataFrame:
    """Berechnet RMSE, MAE und Abdeckung je Prognoseschritt h und zeigt, ob
    der Fehler mit dem Horizont wächst, wie für Mehrschrittprognosen
    erwartet."""
    rows = []
    for h, g in preds.groupby("h"):
        rows.append({
            "h": int(h),
            "RMSE": evaluation.rmse(g["y_true"], g["y_pred"]),
            "MAE": evaluation.mae(g["y_true"], g["y_pred"]),
            "coverage_80": evaluation.interval_coverage(g["y_true"], g["q10"], g["q90"]),
            "n_forecasts": len(g),
        })
    return pd.DataFrame(rows).sort_values("h").reset_index(drop=True)


def summarize_by_group(preds: pd.DataFrame, panel: pd.DataFrame,
                        static: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Berechnet den mittleren MASE je statischer Gruppe, etwa region oder
    income_group, und zeigt, für welche Ländergruppen das globale Modell
    besser oder schlechter funktioniert."""
    preds = compute_row_mase(preds, panel)
    unique = preds.drop_duplicates(["country", "origin_year"])
    merged = unique.merge(static[["country", group_col]], on="country", how="left")
    return (merged.groupby(group_col)["MASE"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "MASE_mean", "std": "MASE_std", "count": "n_folds"})
            .sort_values("MASE_mean")
            .reset_index())


def paired_mase_test(preds_a: pd.DataFrame, preds_b: pd.DataFrame,
                      panel: pd.DataFrame) -> dict[str, Any]:
    """Führt den primären Signifikanztest für den Panel Vergleich zweier
    Modelle durch, einen gepaarten Wilcoxon Vorzeichen Rang Test ergänzt um
    einen t Test auf dem gepaarten MASE je Kombination aus Land und Origin
    Jahr. Die methodische Begründung, warum ein Paneltest hier dem
    Diebold Mariano Test vorzuziehen ist, steht in
    evaluation.paired_forecast_test()."""
    a = compute_row_mase(preds_a, panel).drop_duplicates(["country", "origin_year"])
    b = compute_row_mase(preds_b, panel).drop_duplicates(["country", "origin_year"])
    merged = a.merge(b, on=["country", "origin_year"], suffixes=("_a", "_b"))
    return evaluation.paired_forecast_test(merged["MASE_a"], merged["MASE_b"])


def diebold_mariano_per_country(preds_a: pd.DataFrame, preds_b: pd.DataFrame,
                                 min_obs: int = 15) -> pd.DataFrame:
    """Führt den klassischen Diebold Mariano Test je Land auf einer eigenen,
    zeitlich zusammenhängenden Verlustreihe durch. Die fünf Origin Blöcke
    eines Landes überlappen nicht und lassen sich zu einer einzigen,
    25 Jahre umfassenden Fehlerreihe aneinanderreihen. Diese Funktion
    ergänzt paired_mase_test, sie ersetzt sie nicht. Der Diebold Mariano
    Test prüft je Land für sich, ob der Unterschied über die Zeit
    systematisch ist. Der gepaarte Test prüft, ob das Modell b über das
    gesamte Panel im statistischen Mittel besser ist.

    Als Verlustfunktion dient der absolute Fehler, konsistent mit MAE und
    MASE als Primärmetrik des Projekts. Ein positiver Wert von dm_stat
    bedeutet, dass Modell b den kleineren Fehler aufweist.
    """
    merged = preds_a.merge(preds_b, on=["country", "origin_year", "year", "h"],
                            suffixes=("_a", "_b"))
    merged["e_a"] = (merged["y_true_a"] - merged["y_pred_a"]).abs()
    merged["e_b"] = (merged["y_true_a"] - merged["y_pred_b"]).abs()

    rows = []
    for country, g in merged.sort_values("year").groupby("country"):
        g = g.dropna(subset=["e_a", "e_b"])
        if len(g) < min_obs:
            continue
        stat, p = evaluation.diebold_mariano(g["e_a"], g["e_b"], h=config.HORIZON)
        rows.append({"country": country, "n_obs": len(g), "dm_stat": stat, "dm_p": p})
    return pd.DataFrame(rows)


def aggregate_by_group(preds: pd.DataFrame, static: pd.DataFrame,
                        group_col: str) -> pd.DataFrame:
    """Rollt Länderprognosen zu einem gleichgewichteten Gruppenmittel hoch,
    etwa je Region, als Mittelwert von y_true und y_pred über alle Länder
    der Gruppe, je Kombination aus Origin Jahr, Jahr und h. Ohne eine
    Gewichtung nach BIP oder Bevölkerung, die im Panel nicht verfügbar ist
    und im Notebook diskutiert wird, entspricht dies nicht der offiziellen
    Regionalwachstumsrate der Weltbank, sondern einem einfachen,
    gleichgewichteten Rollup von unten nach oben zur Illustration.

    Die Funktion gibt einen Datensatz im Format der Prognosetabellen
    zurück, wobei die Spalte country den Gruppennamen anstelle eines
    ISO3 Codes enthält. Dadurch lassen sich die bestehenden
    Auswertungsfunktionen wie compute_row_mase unverändert weiterverwenden.
    Dafür wird ein passendes Gruppenpanel benötigt, siehe
    build_group_panel.
    """
    merged = preds.merge(static[["country", group_col]], on="country", how="left")
    grouped = (merged.groupby([group_col, "origin_year", "year", "h"])
               .agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"),
                    n_countries=("country", "nunique"))
               .reset_index()
               .rename(columns={group_col: "country"}))
    grouped["model"] = "aggregate"
    grouped["q10"] = np.nan
    grouped["q90"] = np.nan
    return grouped[["model", "country", "origin_year", "year", "h",
                     "y_true", "y_pred", "q10", "q90", "n_countries"]]


def build_group_panel(panel: pd.DataFrame, static: pd.DataFrame,
                       group_col: str) -> pd.DataFrame:
    """Erzeugt ein aggregiertes, gleichgewichtetes Panel je Gruppe und Jahr.
    Es liefert die historische Skala für die MASE Berechnung der
    Gruppenaggregatprognosen aus aggregate_by_group."""
    merged = panel.merge(static[["country", group_col]], on="country", how="left")
    return (merged.groupby([group_col, "year"])[config.TARGET]
            .mean().reset_index()
            .rename(columns={group_col: "country"}))


def summarize_aggregate(group_preds: pd.DataFrame, group_panel: pd.DataFrame) -> pd.DataFrame:
    """Berechnet MASE und RMSE je Gruppenaggregatprognose, gemittelt über
    die Origins."""
    scored = compute_row_mase(group_preds, group_panel)
    per_group_origin = scored.drop_duplicates(["country", "origin_year"])
    rows = []
    for grp, g in per_group_origin.groupby("country"):
        gg = scored[scored["country"] == grp]
        rows.append({
            "group": grp,
            "MASE_aggregate": g["MASE"].mean(),
            "RMSE_aggregate": evaluation.rmse(gg["y_true"], gg["y_pred"]),
            "n_folds": len(g),
        })
    return pd.DataFrame(rows).sort_values("MASE_aggregate").reset_index(drop=True)


def quantile_crossing_rate(preds: pd.DataFrame) -> float:
    """Berechnet den Anteil der Prognosen mit nicht monotonen Quantilen,
    also q10 größer als y_pred oder y_pred größer als q90.
    pytorch forecasting trainiert jedes Quantil mit einem eigenen
    Ausgabekopf und erzwingt Monotonie nicht automatisch. Das ist ein
    bekanntes Risiko bei einer Quantilsregression mit mehreren Quantilen."""
    crossed = (preds["q10"] > preds["y_pred"]) | (preds["y_pred"] > preds["q90"])
    return float(crossed.mean())


def forecast_dispersion(preds: pd.DataFrame) -> pd.DataFrame:
    """Vergleicht je Kombination aus Land und Origin Jahr, wie stark sich
    die Punktprognose über den Horizont bewegt, mit der tatsächlichen
    Bewegung der Zielgröße über denselben Horizont.

    pred_range und true_range sind jeweils die Spannweite, also Maximum
    minus Minimum, von y_pred beziehungsweise y_true über die fünf
    Horizontschritte. ratio ist der Quotient aus beiden Spannweiten. Ein
    Wert deutlich unter 1 bedeutet, dass sich die Prognose über den
    gesamten Prognosehorizont kaum bewegt, während der tatsächliche
    Verlauf deutlich stärker schwankt. Das ist eine eigenständige
    Diagnostik zur Frage, wie informativ eine Prognose über den Verlauf
    innerhalb des Horizonts tatsächlich ist, unabhängig von MASE oder der
    Abdeckung der Quantile.
    """
    rows = []
    for (country, origin), g in preds.groupby(["country", "origin_year"]):
        pred_range = float(g["y_pred"].max() - g["y_pred"].min())
        true_range = float(g["y_true"].max() - g["y_true"].min())
        ratio = pred_range / true_range if true_range > 0 else float("nan")
        rows.append({"country": country, "origin_year": origin,
                      "pred_range": pred_range, "true_range": true_range,
                      "ratio": ratio})
    return pd.DataFrame(rows)


def naive_drift_forecast(panel: pd.DataFrame, origin_year: int,
                          horizon: int = config.HORIZON) -> pd.DataFrame:
    """Berechnet einen Random Walk mit Drift je Land, ausschließlich als
    interner Referenzpunkt für dieses Notebook. Dies ist nicht der
    offizielle rw_drift Baseline Fold aus Notebook 05, dessen kanonische
    Datei preds_rw_drift.csv vom Team stammt.

    Die Prognose lautet y_hat(origin plus h) gleich y(origin) plus h mal
    drift, wobei drift die mittlere Jahresveränderung von config.TARGET
    bis einschließlich origin_year ist.
    """
    rows = []
    for country, g in panel.groupby("country"):
        hist = g.loc[g["year"] <= origin_year, config.TARGET].dropna()
        if len(hist) < 2:
            continue
        last_value = hist.iloc[-1]
        drift = hist.diff().mean()
        truth = g.set_index("year")[config.TARGET]
        for h in range(1, horizon + 1):
            year = origin_year + h
            rows.append({
                "model": "rw_drift_intern", "country": country,
                "origin_year": origin_year, "year": year, "h": h,
                "y_true": truth.get(year, np.nan),
                "y_pred": last_value + h * drift,
                "q10": np.nan, "q90": np.nan,
            })
    return pd.DataFrame(rows)
