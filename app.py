"""
Redge Compass AI  -  app.py  (Phase 1)
======================================
Application Streamlit d'analyse des données HVAC (lecture et analyse seulement).

L'application NE COMMANDE AUCUN équipement : elle lit un fichier CSV,
détecte des comportements inhabituels avec 3 modèles PyOD, applique des
règles métier HVAC explicables, puis présente des diagnostics PROBABLES
qui doivent être validés par un technicien.

Organisation du fichier (à lire de haut en bas) :
  1. Réglages généraux (couleurs, noms de colonnes, valeurs par défaut)
  2. Fonctions métier HVAC                (tes fonctions d'origine)
  3. Chargement et préparation des données
  4. Qualité des données
  5. Analyse : modèles IA + règles + événements
  6. Scores, énergie, tableaux par équipement et par site
  7. Retours des techniciens (fichier retours_techniciens.csv)
  8. Aides d'affichage (cartes, graphiques, tableaux)
  9. Les 10 pages
 10. Programme principal
"""

import csv
import html
import io
import os
from datetime import date, datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator

from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.knn import KNN

from sklearn.preprocessing import StandardScaler


# ============================================================
# 0. CONFIGURATION DE LA PAGE (doit être le premier appel Streamlit)
# ============================================================

st.set_page_config(
    page_title="Redge Compass AI",
    page_icon="🧭",
    layout="wide",
)


# ============================================================
# 1. RÉGLAGES GÉNÉRAUX
# ============================================================

# Dossier où se trouve app.py (C:\RedgeAI). Sert à retrouver les fichiers CSV.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILE_DEMO_SMALL = "donnees_hvac.csv"
FILE_DEMO_EXTENDED = "donnees_demo_synthetiques.csv"
FEEDBACK_FILE = os.path.join(BASE_DIR, "retours_techniciens.csv")

# ---- Palette (identité visuelle) ----
COLORS = {
    "navy": "#0B2545",      # bleu foncé
    "blue": "#1F6FB2",      # bleu
    "light": "#8EC5F0",     # bleu clair
    "turquoise": "#14B8A6",
    "green": "#2E9E5B",     # fonctionnement normal
    "orange": "#F59E0B",    # alerte
    "red": "#D64545",       # critique
    "grey": "#6B7A8F",
}

PRIORITY_ORDER = ["Critique", "Élevée", "Moyenne", "Faible"]
PRIORITY_COLORS = {
    "Critique": COLORS["red"],
    "Élevée": COLORS["orange"],
    "Moyenne": COLORS["light"],
    "Faible": COLORS["green"],
}

EQUIPMENT_STATUS_COLORS = {
    "Normal": COLORS["green"],
    "À surveiller": COLORS["orange"],
    "Anomalie probable": "#E67E22",
    "Anomalie forte": COLORS["red"],
}

STATUSES = [
    "Nouvelle",
    "À analyser",
    "Confirmée",
    "Intervention planifiée",
    "Résolue",
    "Faux positif",
]
CLOSED_STATUSES = ["Résolue", "Faux positif"]   # anomalies qui ne sont plus "ouvertes"

PAGES = [
    "Vue d'ensemble",
    "Sites",
    "Équipements",
    "Anomalies",
    "Recommandations",
    "Comparaison des modèles",
    "Analyse énergétique",
    "Qualité des données",
    "Rapports",
    "Configuration",
]

# ---- Colonnes attendues dans le CSV ----
REQUIRED_COLUMNS = [
    "timestamp",
    "equipment_id",
    "zone_temperature",
    "setpoint",
    "outdoor_temperature",
    "supply_temperature",
    "return_temperature",
    "electrical_power",
    "valve_position",
    "occupancy",
]

NUMERIC_COLUMNS = [
    "zone_temperature",
    "setpoint",
    "outdoor_temperature",
    "supply_temperature",
    "return_temperature",
    "electrical_power",
    "valve_position",
    "occupancy",
]

# Colonnes facultatives : si elles existent, l'application les utilise.
OPTIONAL_COLUMNS = {
    "client": "Client",
    "site": "Site",
    "building": "Bâtiment",
    "city": "Ville ou localisation",
    "equipment_type": "Type d'équipement",
    "brand_model": "Marque et modèle",
    "last_maintenance": "Dernière maintenance",
    "alarm_count": "Nombre d'alarmes",
}

# Colonnes nécessaires aux règles 5 et 6 (prévues en phase 2).
HVAC_EXTRA_COLUMNS = {
    "filter_pressure": "Pression différentielle du filtre",
    "fan_status": "État du ventilateur",
    "airflow": "Débit d'air",
    "compressor_status": "État du compresseur",
}

# Plages plausibles (pour détecter les valeurs hors plage)
VALID_RANGES = {
    "zone_temperature": (-10, 60),
    "setpoint": (5, 40),
    "outdoor_temperature": (-40, 60),
    "supply_temperature": (-10, 80),
    "return_temperature": (-10, 60),
    "electrical_power": (0, 5000),
    "valve_position": (0, 100),
    "occupancy": (0, 1),
}
STUCK_COLUMNS = ["zone_temperature", "supply_temperature", "return_temperature"]
STUCK_RUN = 12            # nombre de mesures identiques de suite = valeur bloquée
SUPPLY_GAP_LIMIT = 3.0    # écart (°C) de la température de soufflage jugé anormal
DQ_LIMIT = 70             # en dessous, le diagnostic est présenté comme "limité"

MODELS = [
    ("Isolation Forest", "isolation_forest"),
    ("ECOD", "ecod"),
    ("KNN", "knn"),
]

# ---- Paramètres de configuration (valeurs par défaut) ----
CONFIG_DEFAULTS = {
    "cfg_sensitivity": 0.10,
    "cfg_temp_threshold": 2.0,
    "cfg_valve_threshold": 95,
    "cfg_power_factor": 1.5,
    "cfg_min_duration": 30,
    "cfg_neighbors": 5,
    "cfg_vote_rule": 2,
    "cfg_price_kwh": 0.0,
    "cfg_currency": "€",
    "cfg_co2_factor": 0.0,
    "cfg_rule1": True,
    "cfg_rule2": True,
    "cfg_rule3": True,
    "cfg_rule4": True,
}

# Paramètres qui influencent l'analyse (les autres, comme le prix du kWh,
# ne relancent pas les modèles).
ANALYSIS_KEYS = [
    "sensitivity", "temp_threshold", "valve_threshold", "power_factor",
    "min_duration", "neighbors", "vote_rule",
    "rule1", "rule2", "rule3", "rule4",
]


def init_config():
    """
    Crée les paramètres au premier lancement et les conserve quand on
    change de page (Streamlit oublie sinon les réglages d'une page
    qui n'est plus affichée).
    """
    for key, default in CONFIG_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default
        else:
            st.session_state[key] = st.session_state[key]


def get_cfg():
    """Retourne les paramètres sous forme de dictionnaire simple."""
    return {key[4:]: st.session_state[key] for key in CONFIG_DEFAULTS}


def reset_config():
    """Remet tous les paramètres à leur valeur par défaut."""
    for key, default in CONFIG_DEFAULTS.items():
        st.session_state[key] = default


# ============================================================
# 2. FONCTIONS MÉTIER HVAC  (conservées de ta version d'origine)
# ============================================================

def calculate_confidence(votes):
    """
    Transforme le nombre de votes en indice d'accord des modèles.

    Attention :
    ce résultat n'est pas une probabilité scientifique de panne.
    """

    if votes == 3:
        return 90

    if votes == 2:
        return 70

    if votes == 1:
        return 40

    return 10


def calculate_ai_level(votes):
    """
    Crée une désignation compréhensible pour le client.
    """

    if votes == 3:
        return "Anomalie forte"

    if votes == 2:
        return "Anomalie probable"

    if votes == 1:
        return "À surveiller"

    return "Comportement probablement normal"


def priority_level(score):
    """Convertit un score de priorité (0 à 100) en niveau lisible."""

    if score >= 80:
        return "Critique"

    if score >= 60:
        return "Élevée"

    if score >= 30:
        return "Moyenne"

    return "Faible"


def calculate_priority(row):
    """
    Calcule la priorité à partir des règles HVAC
    et des votes des modèles.
    (Accepte une ligne de tableau ou un dictionnaire.)
    """

    score = 0

    if row.get("comfort_anomaly", False):
        score += 20

    if row.get("possible_cooling_problem", False):
        score += 25

    if row.get("possible_overconsumption", False):
        score += 20

    if row.get("out_of_hours_operation", False):
        score += 15

    if row.get("model_votes", 0) == 1:
        score += 10

    elif row.get("model_votes", 0) == 2:
        score += 20

    elif row.get("model_votes", 0) == 3:
        score += 30

    score = min(score, 100)

    return pd.Series([score, priority_level(score)], index=["score", "level"])


def create_observation(row):
    """
    Prépare la liste des preuves disponibles.
    """

    observations = []

    if row.get("comfort_anomaly", False):
        observations.append(
            f"écart à la consigne de "
            f"{row.get('temperature_error', 0):.1f} °C"
        )

    if row.get("valve_saturated", False):
        observations.append(
            f"vanne ouverte à "
            f"{row.get('valve_position', 0):.0f} %"
        )

    if row.get("high_power", False):
        observations.append(
            f"puissance élevée : "
            f"{row.get('electrical_power', 0):.1f} kW"
        )

    if row.get("out_of_hours_operation", False):
        observations.append(
            "équipement actif pendant une période non occupée"
        )

    votes = row.get("model_votes", 0)

    if votes > 0:
        observations.append(
            f"{votes} modèle(s) IA sur 3 ont détecté "
            f"un comportement inhabituel"
        )

    if not observations:
        return "Aucune anomalie significative détectée"

    return " | ".join(observations)


def create_probable_cause(row):
    """
    Utilise les règles HVAC pour proposer une cause probable.
    """

    if (
        row.get("possible_cooling_problem", False)
        and row.get("high_power", False)
    ):
        return (
            "Performance frigorifique potentiellement insuffisante, "
            "débit d'air insuffisant ou échange thermique dégradé "
            "(à vérifier)."
        )

    if row.get("possible_cooling_problem", False):
        return (
            "Refroidissement potentiellement insuffisant malgré "
            "une demande élevée de la régulation (à vérifier)."
        )

    if row.get("out_of_hours_operation", False):
        return (
            "Programmation horaire ou calendrier d'occupation "
            "potentiellement incorrect (à vérifier)."
        )

    if (
        row.get("comfort_anomaly", False)
        and row.get("model_votes", 0) >= 2
    ):
        return (
            "Dérive de régulation ou performance thermique "
            "potentiellement insuffisante (à vérifier)."
        )

    if row.get("possible_overconsumption", False):
        return (
            "Rendement potentiellement dégradé, mauvais réglage ou "
            "charge anormalement élevée (à vérifier)."
        )

    if row.get("model_votes", 0) >= 2:
        return (
            "Cause non déterminée automatiquement. "
            "Une analyse technique complémentaire est nécessaire."
        )

    if row.get("comfort_anomaly", False):
        return (
            "Écart de confort persistant sans confirmation par les "
            "modèles IA : cause non déterminée, à vérifier."
        )

    return "Aucune cause probable identifiée."


def create_recommendation(row):
    """
    Génère une action à partir des anomalies et règles HVAC.
    """

    if (
        row.get("possible_cooling_problem", False)
        and row.get("possible_overconsumption", False)
        and row.get("model_votes", 0) >= 2
    ):
        return (
            "Contrôler la production frigorifique, la température "
            "de soufflage, le débit d'air, le filtre, l'échangeur "
            "et le fonctionnement de la vanne."
        )

    if row.get("possible_cooling_problem", False):
        return (
            "Vérifier la température de soufflage, la commande "
            "de la vanne, le compresseur et le débit d'air."
        )

    if row.get("out_of_hours_operation", False):
        return (
            "Vérifier le calendrier BMS, les horaires d'occupation "
            "et les commandes de marche forcée."
        )

    if row.get("possible_overconsumption", False):
        return (
            "Contrôler la régulation, les consignes, l'état du filtre "
            "et l'efficacité énergétique de l'équipement."
        )

    if (
        row.get("comfort_anomaly", False)
        and row.get("model_votes", 0) >= 2
    ):
        return (
            "Contrôler la régulation, la température de soufflage "
            "et le fonctionnement thermique de l'équipement."
        )

    if row.get("comfort_anomaly", False):
        return (
            "Surveiller l'écart à la consigne et vérifier "
            "la régulation si la situation persiste."
        )

    if row.get("model_votes", 0) >= 2:
        return (
            "Examiner les températures, la puissance, la vanne "
            "et les états de fonctionnement."
        )

    if row.get("model_votes", 0) == 1:
        return (
            "Continuer la surveillance. Un seul modèle a identifié "
            "un comportement inhabituel."
        )

    return "Aucune action immédiate recommandée."


def dataframe_to_excel(dataframe):
    """
    Transforme le tableau de résultats en fichier Excel téléchargeable.
    """

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        dataframe.to_excel(
            writer,
            sheet_name="Analyse Redge",
            index=False
        )

    output.seek(0)

    return output.getvalue()


def dataframes_to_excel(sheets):
    """
    Variante : écrit plusieurs feuilles Excel à partir d'un dictionnaire
    {nom_de_feuille: tableau}.
    """

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, table in sheets.items():
            safe_name = name[:31].replace("/", "-").replace("\\", "-")
            table.to_excel(writer, sheet_name=safe_name, index=False)

    output.seek(0)

    return output.getvalue()


# ============================================================
# 3. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================

@st.cache_data
def load_csv_bytes(raw_bytes):
    """
    Lit un CSV à partir de son contenu.
    Détecte automatiquement le séparateur (virgule, point-virgule, tabulation).
    Si le séparateur est ";", la virgule est lue comme séparateur décimal.
    """

    text = raw_bytes.decode("utf-8-sig", errors="replace")

    try:
        separator = csv.Sniffer().sniff(text[:5000], delimiters=",;\t").delimiter
    except csv.Error:
        separator = ","

    decimal = "," if separator == ";" else "."

    return pd.read_csv(io.StringIO(text), sep=separator, decimal=decimal)


def read_local_file(file_name):
    """Lit un fichier CSV situé dans le dossier de l'application."""

    with open(os.path.join(BASE_DIR, file_name), "rb") as file:
        return load_csv_bytes(file.read())


def effective_mapping(df_columns):
    """
    Correspondance {nom attendu par l'app: nom de colonne du fichier}.
    Par défaut, on utilise la colonne du même nom si elle existe.
    """

    saved = st.session_state.get("column_mapping", {})
    expected = (
        REQUIRED_COLUMNS
        + list(OPTIONAL_COLUMNS)
        + list(HVAC_EXTRA_COLUMNS)
    )

    mapping = {}

    for name in expected:
        source = saved.get(name, "")

        if source not in df_columns:
            source = name if name in df_columns else ""

        mapping[name] = source

    return mapping


def apply_mapping(df):
    """Renomme les colonnes du fichier selon la correspondance choisie."""

    mapping = effective_mapping(list(df.columns))

    rename = {
        source: name
        for name, source in mapping.items()
        if source and source != name
    }

    return df.rename(columns=rename)


# ============================================================
# 4. QUALITÉ DES DONNÉES
# ============================================================

def compute_quality(raw):
    """
    Évalue la qualité du fichier AVANT tout nettoyage.
    Retourne un dictionnaire de tableaux et de chiffres.
    """

    check_columns = ["timestamp", "equipment_id"] + NUMERIC_COLUMNS

    d = raw[REQUIRED_COLUMNS].copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d["equipment_id"] = d["equipment_id"].fillna("Inconnu").astype(str)

    for column in NUMERIC_COLUMNS:
        d[column] = pd.to_numeric(d[column], errors="coerce")

    rows = []
    gap_rows = []
    missing_by_column = pd.Series(0, index=check_columns, dtype="float64")
    out_of_range_by_column = {column: 0 for column in VALID_RANGES}
    stuck_by_column = {column: 0 for column in STUCK_COLUMNS}

    for equipment, g in d.groupby("equipment_id"):

        n_raw = len(g)
        missing = g[check_columns].isna().sum()
        missing_by_column = missing_by_column.add(missing, fill_value=0)

        out_of_range = 0
        for column, (low, high) in VALID_RANGES.items():
            count = int(((g[column] < low) | (g[column] > high)).sum())
            out_of_range += count
            out_of_range_by_column[column] += count

        duplicates = int(g.duplicated(subset=["timestamp"]).sum())

        g2 = (
            g.dropna(subset=["timestamp"])
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
        )

        differences = g2["timestamp"].diff().dt.total_seconds().div(60)
        step = differences.median() if len(g2) > 1 else np.nan

        if pd.isna(step) or step <= 0:
            step = 5.0

        gap_mask = differences > step * 2.5
        previous_time = g2["timestamp"].shift()

        for index in g2.index[gap_mask]:
            gap_rows.append({
                "equipment_id": equipment,
                "start": previous_time.loc[index],
                "end": g2.loc[index, "timestamp"],
                "duration_min": float(differences.loc[index]),
            })

        missing_samples = int(
            (differences[gap_mask] / step).round().sub(1).clip(lower=0).sum()
        )

        stuck_total = 0
        for column in STUCK_COLUMNS:
            s = g2[column]
            run_id = (s != s.shift()).cumsum()
            run_length = s.groupby(run_id).transform("size")
            count = int(((run_length >= STUCK_RUN) & s.notna()).sum())
            stuck_total += count
            stuck_by_column[column] += count

        # Parts (0 à 1) de chaque problème
        missing_share = missing.sum() / max(n_raw * len(check_columns), 1)
        range_share = out_of_range / max(n_raw * len(VALID_RANGES), 1)
        duplicate_share = duplicates / max(n_raw, 1)
        stuck_share = stuck_total / max(len(g2) * len(STUCK_COLUMNS), 1)
        gap_share = missing_samples / max(len(g2) + missing_samples, 1)

        # Data Quality Score : 100 moins les pénalités (en points).
        penalty = 100 * (
            missing_share + range_share + duplicate_share
            + stuck_share + gap_share
        )
        score = float(np.clip(100 - penalty, 0, 100))

        rows.append({
            "equipment_id": equipment,
            "rows": n_raw,
            "completeness_pct": 100 * (1 - missing_share),
            "missing_values": int(missing.sum()),
            "out_of_range": out_of_range,
            "duplicates": duplicates,
            "stuck_values": stuck_total,
            "interruptions": int(gap_mask.sum()),
            "missing_samples": missing_samples,
            "step_min": float(step),
            "last_timestamp": g2["timestamp"].max(),
            "dq_score": score,
        })

    per_equipment = pd.DataFrame(rows)

    total_rows = int(per_equipment["rows"].sum())
    global_score = float(
        np.average(per_equipment["dq_score"], weights=per_equipment["rows"])
    )

    summary = {
        "rows": total_rows,
        "completeness_pct": float(
            np.average(
                per_equipment["completeness_pct"],
                weights=per_equipment["rows"],
            )
        ),
        "missing_values": int(per_equipment["missing_values"].sum()),
        "out_of_range": int(per_equipment["out_of_range"].sum()),
        "duplicates": int(per_equipment["duplicates"].sum()),
        "stuck_values": int(per_equipment["stuck_values"].sum()),
        "interruptions": int(per_equipment["interruptions"].sum()),
        "mean_step_min": float(per_equipment["step_min"].mean()),
        "last_timestamp": per_equipment["last_timestamp"].max(),
        "dq_score": global_score,
    }

    return {
        "per_equipment": per_equipment,
        "summary": summary,
        "gaps": pd.DataFrame(
            gap_rows,
            columns=["equipment_id", "start", "end", "duration_min"],
        ),
        "missing_by_column": missing_by_column,
        "out_of_range_by_column": out_of_range_by_column,
        "stuck_by_column": stuck_by_column,
    }


# ============================================================
# 5. ANALYSE : MODÈLES IA + RÈGLES HVAC + ÉVÉNEMENTS
# ============================================================

def clean_data(raw):
    """Prépare les données : dates, nombres, doublons, valeurs manquantes."""

    data = raw.copy()

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp"])
    data["equipment_id"] = data["equipment_id"].fillna("Inconnu").astype(str)

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Les doublons (même équipement, même instant) sont supprimés.
    data = data.drop_duplicates(subset=["equipment_id", "timestamp"])
    data = data.sort_values(["equipment_id", "timestamp"]).reset_index(drop=True)

    # Valeurs manquantes : médiane de l'équipement, sinon médiane générale.
    for column in NUMERIC_COLUMNS:
        data[column] = data.groupby("equipment_id")[column].transform(
            lambda s: s.fillna(s.median())
        )
        data[column] = data[column].fillna(data[column].median()).fillna(0)

    data["occupancy"] = (data["occupancy"] >= 0.5).astype(int)

    # Colonnes facultatives
    for column, default in [
        ("client", "Non renseigné"),
        ("site", "Non renseigné"),
        ("building", "Non renseigné"),
        ("city", "Non renseigné"),
        ("brand_model", "Non disponible"),
        ("last_maintenance", "Non disponible"),
    ]:
        if column not in data.columns:
            data[column] = default
        else:
            data[column] = data[column].fillna(default).astype(str)

    if "equipment_type" not in data.columns:
        # Type déduit du début de l'identifiant (RTU_01 -> RTU)
        data["equipment_type"] = (
            data["equipment_id"].str.split(r"[_\- ]", regex=True).str[0]
        )
    else:
        data["equipment_type"] = (
            data["equipment_type"].fillna("Non renseigné").astype(str)
        )

    if "alarm_count" in data.columns:
        data["alarm_count"] = pd.to_numeric(data["alarm_count"], errors="coerce")

    # Intervalle entre deux mesures, par équipement (en minutes)
    step = data.groupby("equipment_id")["timestamp"].apply(
        lambda s: s.diff().dt.total_seconds().median() / 60
    )
    step = step.fillna(5.0).where(step > 0, 5.0)
    data["dt_min"] = data["equipment_id"].map(step)

    return data


def add_hvac_indicators(data, cfg):
    """Calcule les indicateurs et les règles HVAC sur chaque mesure."""

    data["temperature_error"] = data["zone_temperature"] - data["setpoint"]
    data["air_delta_t"] = data["return_temperature"] - data["supply_temperature"]

    # Niveau "habituel" de puissance et de soufflage, par équipement,
    # calculé pendant les périodes d'occupation (sinon sur tout l'historique).
    for column, name in [
        ("electrical_power", "power_reference"),
        ("supply_temperature", "supply_reference"),
    ]:
        all_median = data.groupby("equipment_id")[column].median()
        occupied_median = (
            data[data["occupancy"] == 1].groupby("equipment_id")[column].median()
        )
        reference = occupied_median.reindex(all_median.index).fillna(all_median)
        data[name] = data["equipment_id"].map(reference)

    data["valve_saturated"] = data["valve_position"] >= cfg["valve_threshold"]

    data["high_power"] = (
        data["electrical_power"] > data["power_reference"] * cfg["power_factor"]
    )

    far_from_setpoint = data["temperature_error"].abs() > cfg["temp_threshold"]
    occupied = data["occupancy"] == 1

    # Règle 1 : inconfort thermique
    data["comfort_anomaly"] = cfg["rule1"] & occupied & far_from_setpoint

    # Règle 2 : refroidissement potentiellement insuffisant
    data["possible_cooling_problem"] = (
        cfg["rule2"]
        & occupied
        & (data["temperature_error"] > cfg["temp_threshold"])
        & data["valve_saturated"]
    )

    # Règle 3 : surconsommation potentielle
    data["possible_overconsumption"] = (
        cfg["rule3"] & data["high_power"] & far_from_setpoint
    )

    # Règle 4 : fonctionnement hors occupation
    data["equipment_active"] = (
        data["electrical_power"]
        > np.maximum(data["power_reference"] * 0.30, 0.1)
    )
    data["out_of_hours_operation"] = (
        cfg["rule4"] & (data["occupancy"] == 0) & data["equipment_active"]
    )

    return data


def run_models(data, cfg):
    """Entraîne les 3 modèles PyOD et ajoute étiquettes, scores et votes."""

    feature_columns = [
        "zone_temperature",
        "setpoint",
        "outdoor_temperature",
        "supply_temperature",
        "return_temperature",
        "electrical_power",
        "valve_position",
        "temperature_error",
        "air_delta_t",
    ]

    X = data[feature_columns].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X_scaled = StandardScaler().fit_transform(X)

    # Le nombre de voisins doit rester inférieur au nombre de mesures.
    neighbors = min(int(cfg["neighbors"]), max(2, len(data) - 1))

    models = {
        "Isolation Forest": IForest(
            n_estimators=200,
            contamination=cfg["sensitivity"],
            random_state=42,
        ),
        "ECOD": ECOD(contamination=cfg["sensitivity"]),
        "KNN": KNN(
            contamination=cfg["sensitivity"],
            n_neighbors=neighbors,
        ),
    }

    for model_name, model in models.items():
        model.fit(X_scaled)
        safe_name = model_name.lower().replace(" ", "_")
        data[f"{safe_name}_anomaly"] = model.labels_ == 1
        data[f"{safe_name}_score"] = model.decision_scores_

    anomaly_columns = [f"{safe}_anomaly" for _, safe in MODELS]

    data["model_votes"] = data[anomaly_columns].astype(int).sum(axis=1)
    data["ai_anomaly"] = data["model_votes"] >= cfg["vote_rule"]
    data["ai_confidence"] = data["model_votes"].apply(calculate_confidence)
    data["ai_level"] = data["model_votes"].apply(calculate_ai_level)

    priority = data.apply(calculate_priority, axis=1)
    data["priority_score"] = priority["score"]
    data["priority"] = priority["level"]

    return data


def build_events(data, cfg, quality_per_equipment, extra_missing):
    """
    Regroupe les mesures consécutives qui posent le même problème
    en UN événement (début, fin, durée, valeurs max et moyennes).
    Un événement est "persistant" s'il dure au moins la durée minimale.
    """

    data["candidate"] = (
        data["ai_anomaly"]
        | data["comfort_anomaly"]
        | data["possible_cooling_problem"]
        | data["possible_overconsumption"]
        | data["out_of_hours_operation"]
    )
    data["event_id"] = ""

    quality_lookup = quality_per_equipment.set_index("equipment_id")["dq_score"]
    rows_by_equipment = data.groupby("equipment_id").size()

    events = []

    for equipment, g in data.groupby("equipment_id", sort=True):

        step = float(g["dt_min"].iloc[0])
        gap = g["timestamp"].diff().dt.total_seconds().div(60) > step * 2.5
        candidate = g["candidate"]
        new_block = (candidate != candidate.shift()) | gap
        block = new_block.cumsum()

        for _, e in g[candidate].groupby(block[candidate]):

            start = e["timestamp"].min()
            end = e["timestamp"].max()
            duration = (end - start).total_seconds() / 60 + step
            event_id = f"{equipment}|{start:%Y-%m-%d %H:%M}"
            data.loc[e.index, "event_id"] = event_id

            votes = int(e["model_votes"].max())
            ai_confirmed = votes >= cfg["vote_rule"]

            # Règles 2 et 3 : elles exigent la confirmation des modèles IA.
            cooling = bool(e["possible_cooling_problem"].any()) and ai_confirmed
            overconsumption = (
                bool(e["possible_overconsumption"].any()) and ai_confirmed
            )
            comfort = bool(e["comfort_anomaly"].any())
            out_of_hours = bool(e["out_of_hours_operation"].any())

            peak = e.sort_values(
                ["model_votes", "priority_score"], ascending=False
            ).iloc[0]
            temp_row = e.loc[e["temperature_error"].abs().idxmax()]
            power_row = e.loc[e["electrical_power"].idxmax()]

            flags = {
                "comfort_anomaly": comfort,
                "valve_saturated": bool(e["valve_saturated"].any()),
                "high_power": bool(e["high_power"].any()),
                "possible_cooling_problem": cooling,
                "possible_overconsumption": overconsumption,
                "out_of_hours_operation": out_of_hours,
                "model_votes": votes,
                "temperature_error": float(temp_row["temperature_error"]),
                "valve_position": float(e["valve_position"].max()),
                "electrical_power": float(power_row["electrical_power"]),
            }

            priority = calculate_priority(flags)

            # ---- Catégorie et règles déclenchées ----
            categories = []
            rules = []
            if comfort:
                categories.append("Confort")
                rules.append("Règle 1 : inconfort thermique")
            if cooling:
                categories.append("Refroidissement")
                rules.append("Règle 2 : refroidissement potentiellement insuffisant")
            if overconsumption:
                categories.append("Consommation")
                rules.append("Règle 3 : surconsommation potentielle")
            if out_of_hours:
                categories.append("Hors occupation")
                rules.append("Règle 4 : fonctionnement hors occupation")
            if not categories:
                categories.append("Détection IA seule")
            if (
                (e["possible_cooling_problem"].any() and not cooling)
                or (e["possible_overconsumption"].any() and not overconsumption)
            ):
                rules.append(
                    "Règle 2 ou 3 non retenue : conditions réunies mais "
                    "non confirmées par les modèles IA"
                )

            # ---- Variables anormales et observation ----
            observation = create_observation(flags)
            abnormal = []

            if comfort or abs(flags["temperature_error"]) > cfg["temp_threshold"]:
                abnormal.append("température de zone")
            if flags["high_power"] or out_of_hours:
                abnormal.append("puissance électrique")
            if flags["valve_saturated"]:
                abnormal.append("position de vanne")

            supply_excess = float(
                (e["supply_temperature"] - e["supply_reference"]).max()
            )
            if supply_excess > SUPPLY_GAP_LIMIT:
                abnormal.append("température de soufflage")
                observation += (
                    f" | température de soufflage anormalement élevée "
                    f"({e['supply_temperature'].max():.1f} °C, habituelle "
                    f"≈ {e['supply_reference'].iloc[0]:.1f} °C)"
                )

            # ---- Valeurs réelles / attendues ----
            power_expected = float(power_row["power_reference"])
            power_gap_pct = (
                100 * (power_row["electrical_power"] - power_expected)
                / power_expected
                if power_expected > 0 else np.nan
            )

            # ---- Énergie ----
            kwh = e["electrical_power"] * e["dt_min"] / 60
            avoidable = 0.0
            if out_of_hours:
                avoidable += float(kwh[e["out_of_hours_operation"]].sum())
            if overconsumption:
                over = e[e["possible_overconsumption"]]
                avoidable += float(
                    (
                        (over["electrical_power"] - over["power_reference"])
                        .clip(lower=0) * over["dt_min"] / 60
                    ).sum()
                )

            # ---- Fiabilité du diagnostic ----
            equipment_quality = float(quality_lookup.get(equipment, 100))
            limited = (
                equipment_quality < DQ_LIMIT
                or rows_by_equipment.get(equipment, 0) < 50
            )

            limits = [
                "Diagnostic indicatif : la cause reste à confirmer par un "
                "technicien."
            ]
            if limited:
                limits.append(
                    f"Fiabilité limitée : qualité des données "
                    f"{equipment_quality:.0f}/100 ou historique de moins de "
                    f"50 mesures ({rows_by_equipment.get(equipment, 0)} mesures)."
                )
            if extra_missing:
                limits.append(
                    "Variables non disponibles pour affiner le diagnostic : "
                    + ", ".join(extra_missing) + "."
                )

            events.append({
                "event_id": event_id,
                "client": e["client"].iloc[0],
                "site": e["site"].iloc[0],
                "building": e["building"].iloc[0],
                "equipment_id": equipment,
                "equipment_type": e["equipment_type"].iloc[0],
                "start": start,
                "end": end,
                "duration_min": duration,
                "n_samples": len(e),
                "persistent": duration >= cfg["min_duration"],
                "votes": votes,
                "models_voted": ", ".join(
                    name for name, safe in MODELS if bool(peak[f"{safe}_anomaly"])
                ) or "Aucun",
                "agreement": calculate_confidence(votes),
                "level": (
                    "Anomalie forte" if votes >= 3
                    else "Anomalie probable" if votes == 2
                    else "À surveiller"
                ),
                "category": " + ".join(categories),
                "rules": "; ".join(rules) if rules else
                         "Aucune règle HVAC (détection IA seule)",
                "observation": observation,
                "abnormal_variables": ", ".join(abnormal) if abnormal else
                                      "aucune variable clairement identifiée",
                "temp_actual": float(temp_row["zone_temperature"]),
                "temp_expected": float(temp_row["setpoint"]),
                "temp_gap": float(temp_row["temperature_error"]),
                "temp_gap_mean": float(e["temperature_error"].mean()),
                "power_actual": float(power_row["electrical_power"]),
                "power_mean": float(e["electrical_power"].mean()),
                "power_expected": power_expected,
                "power_gap_pct": float(power_gap_pct),
                "valve_max": float(e["valve_position"].max()),
                "priority_score": int(priority["score"]),
                "priority": priority["level"],
                "probable_cause": create_probable_cause(flags),
                "recommendation": create_recommendation(flags),
                "reliability": "Limitée" if limited else "Normale",
                "limits": " ".join(limits),
                "energy_kwh": float(kwh.sum()),
                "avoidable_kwh": avoidable,
            })

    columns_order = [
        "event_id", "client", "site", "building", "equipment_id",
        "equipment_type", "start", "end", "duration_min", "n_samples",
        "persistent", "votes", "models_voted", "agreement", "level",
        "category", "rules", "observation", "abnormal_variables",
        "temp_actual", "temp_expected", "temp_gap", "temp_gap_mean",
        "power_actual", "power_mean", "power_expected", "power_gap_pct",
        "valve_max", "priority_score", "priority", "probable_cause",
        "recommendation", "reliability", "limits", "energy_kwh",
        "avoidable_kwh",
    ]

    return pd.DataFrame(events, columns=columns_order)


@st.cache_data(show_spinner="Analyse des données en cours...")
def run_analysis(mapped_df, analysis_cfg):
    """
    Fonction principale d'analyse (mise en cache : elle ne se relance que
    si les données ou les paramètres changent).
    """

    cfg = analysis_cfg

    quality = compute_quality(mapped_df)

    data = clean_data(mapped_df)
    data = add_hvac_indicators(data, cfg)
    data = run_models(data, cfg)

    extra_missing = [
        label for column, label in HVAC_EXTRA_COLUMNS.items()
        if column not in mapped_df.columns
    ]
    if "alarm_count" not in mapped_df.columns:
        extra_missing.append("alarmes")

    events = build_events(
        data, cfg, quality["per_equipment"], extra_missing
    )

    return {
        "data": data,
        "events": events,
        "quality": quality,
        "extra_missing": extra_missing,
    }


# ============================================================
# 6. SCORES, ÉNERGIE, TABLEAUX PAR ÉQUIPEMENT ET PAR SITE
# ============================================================

def format_number(value, decimals=1, unit=""):
    """Format français (virgule décimale). Retourne 'n/a' si pas de valeur."""

    if value is None or pd.isna(value):
        return "n/a"

    text = f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")

    return f"{text} {unit}".strip()


def format_datetime(value):
    """Date et heure au format français."""

    if value is None or pd.isna(value):
        return "n/a"

    return pd.Timestamp(value).strftime("%d/%m/%Y %H:%M")


def format_duration(minutes):
    """Durée lisible : '20 min' ou '2 h 30'."""

    minutes = int(round(minutes))

    if minutes < 60:
        return f"{minutes} min"

    return f"{minutes // 60} h {minutes % 60:02d}"


def build_equipment_table(data, events_open, short_open, quality_eq,
                          reference_time, cfg):
    """
    Une ligne par équipement : état, scores (Health, Comfort, Energy),
    énergie, statut global.
    """

    rows = []

    open_ids = set(events_open["event_id"])
    quality_lookup = quality_eq.set_index("equipment_id")["dq_score"]

    for equipment, g in data.groupby("equipment_id"):

        last = g.iloc[-1]
        occupied = g[g["occupancy"] == 1]
        ev = events_open[events_open["equipment_id"] == equipment]
        short = short_open[short_open["equipment_id"] == equipment]
        step = float(last["dt_min"])
        n = len(g)

        # ---- Comfort Score : % du temps occupé dans la plage acceptable ----
        if len(occupied):
            comfort = 100 * float(
                (occupied["temperature_error"].abs() <= cfg["temp_threshold"]).mean()
            )
            drift = float(occupied["temperature_error"].abs().mean())
        else:
            comfort = np.nan
            drift = 0.0

        # ---- Health Score (formule simple et expliquée dans la page) ----
        share_event = 100 * float(g["event_id"].isin(open_ids).mean())
        share_power = 100 * float(g["high_power"].mean())
        dq = float(quality_lookup.get(equipment, 100))

        penalty_events = min(40.0, 2.0 * share_event)
        penalty_power = min(15.0, share_power)
        penalty_drift = min(20.0, 5.0 * drift)
        penalty_data = min(20.0, (100 - dq) / 5)

        health = float(np.clip(
            100 - penalty_events - penalty_power - penalty_drift - penalty_data,
            0, 100,
        ))

        # ---- Énergie : puissance (kW) x durée de la mesure (h) ----
        kwh = g["electrical_power"] * g["dt_min"] / 60

        # ---- Statut global (anomalies ouvertes uniquement) ----
        if (ev["level"] == "Anomalie forte").any():
            status = "Anomalie forte"
        elif (ev["level"] == "Anomalie probable").any():
            status = "Anomalie probable"
        elif len(ev) or len(short):
            status = "À surveiller"
        else:
            status = "Normal"

        # ---- Communication ----
        silence = (reference_time - last["timestamp"]).total_seconds() / 60
        if silence <= 3 * step:
            communication = "Active"
        else:
            communication = (
                f"Interrompue (dernière donnée : "
                f"{format_datetime(last['timestamp'])})"
            )

        active_threshold = max(last["power_reference"] * 0.30, 0.1)
        operating = (
            "En fonctionnement"
            if last["electrical_power"] > active_threshold
            else "À l'arrêt ou en veille"
        )

        rows.append({
            "equipment_id": equipment,
            "equipment_type": last["equipment_type"],
            "client": last["client"],
            "site": last["site"],
            "building": last["building"],
            "city": last["city"],
            "brand_model": last["brand_model"],
            "last_maintenance": last["last_maintenance"],
            "alarm_count": (
                float(last["alarm_count"])
                if "alarm_count" in g.columns else np.nan
            ),
            "communication": communication,
            "operating": operating,
            "temperature": float(last["zone_temperature"]),
            "setpoint": float(last["setpoint"]),
            "power": float(last["electrical_power"]),
            "valve": float(last["valve_position"]),
            "health_score": health,
            "comfort_score": comfort,
            "last_anomaly": ev["end"].max() if len(ev) else pd.NaT,
            "status": status,
            "n_open_events": len(ev),
            "energy_kwh": float(kwh.sum()),
            "energy_occupied_kwh": float(kwh[g["occupancy"] == 1].sum()),
            "energy_out_of_occupancy_kwh": float(kwh[g["occupancy"] == 0].sum()),
            "max_power": float(g["electrical_power"].max()),
            "mean_power_occupied": (
                float(occupied["electrical_power"].mean())
                if len(occupied) else np.nan
            ),
            "avoidable_kwh": float(ev["avoidable_kwh"].sum()),
            "last_timestamp": last["timestamp"],
            "data_quality": dq,
            "measures": n,
        })

    table = pd.DataFrame(rows)

    # ---- Energy Score : comparaison avec les équipements du même type ----
    table["energy_score"] = np.nan
    for _, group in table.groupby("equipment_type"):
        if len(group) >= 2:
            median_power = group["mean_power_occupied"].median()
            if median_power and median_power > 0:
                excess = (group["mean_power_occupied"] / median_power - 1) * 100
                table.loc[group.index, "energy_score"] = (
                    100 - excess.clip(lower=0)
                ).clip(0, 100)

    return table


def build_site_table(equipment_table, events_open):
    """Une ligne par site."""

    rows = []

    for (client, site), g in equipment_table.groupby(["client", "site"]):
        ev = events_open[
            (events_open["client"] == client) & (events_open["site"] == site)
        ]
        rows.append({
            "client": client,
            "site": site,
            "city": g["city"].iloc[0],
            "equipments": len(g),
            "comfort_score": g["comfort_score"].mean(),
            "energy_score": g["energy_score"].mean(),
            "health_score": g["health_score"].mean(),
            "open_anomalies": len(ev),
            "energy_kwh": g["energy_kwh"].sum(),
            "last_update": g["last_timestamp"].max(),
        })

    return pd.DataFrame(rows)


# ============================================================
# 7. RETOURS DES TECHNICIENS  (fichier retours_techniciens.csv)
# ============================================================

FEEDBACK_COLUMNS = [
    "event_id", "status", "validation", "real_cause",
    "action_done", "comment", "resolution_date", "updated_at",
]


def load_feedback():
    """Lit les retours enregistrés (tableau vide si le fichier n'existe pas)."""

    if os.path.exists(FEEDBACK_FILE):
        try:
            table = pd.read_csv(FEEDBACK_FILE, dtype=str).fillna("")
            for column in FEEDBACK_COLUMNS:
                if column not in table.columns:
                    table[column] = ""
            return table[FEEDBACK_COLUMNS]
        except Exception:
            pass

    return pd.DataFrame(columns=FEEDBACK_COLUMNS)


def save_feedback(event_id, **fields):
    """Ajoute ou met à jour le retour d'un technicien pour un événement."""

    table = load_feedback()

    if event_id in set(table["event_id"]):
        index = table.index[table["event_id"] == event_id][0]
    else:
        index = len(table)
        table.loc[index, "event_id"] = event_id
        for column in FEEDBACK_COLUMNS[1:]:
            table.loc[index, column] = ""

    for key, value in fields.items():
        table.loc[index, key] = "" if value is None else str(value)

    table.loc[index, "updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    try:
        table.to_csv(FEEDBACK_FILE, index=False, encoding="utf-8-sig")
    except PermissionError:
        st.error(
            "Impossible d'écrire retours_techniciens.csv. "
            "Ferme le fichier s'il est ouvert dans Excel puis recommence."
        )


# ============================================================
# 8. AIDES D'AFFICHAGE
# ============================================================

def inject_css():
    """Style général : fond gris clair, cartes blanches, bandeau bleu foncé."""

    st.markdown(
        f"""
        <style>
        .block-container {{ padding-top: 1.2rem; max-width: 1450px; }}
        h1, h2, h3, h4 {{ color: {COLORS['navy']}; }}
        .rc-banner {{
            background: linear-gradient(90deg, {COLORS['navy']} 0%,
                        {COLORS['blue']} 100%);
            color: white; padding: 18px 24px; border-radius: 10px;
            margin-bottom: 14px;
        }}
        .rc-banner h2 {{ color: white; margin: 0; font-size: 1.5rem; }}
        .rc-banner p {{ margin: 4px 0 0 0; color: #DCE9F7; font-size: 0.9rem; }}
        .rc-card {{
            background: white; border-radius: 8px; padding: 12px 14px;
            box-shadow: 0 1px 3px rgba(11,37,69,0.12); margin-bottom: 10px;
            min-height: 92px;
        }}
        .rc-card-label {{ color: {COLORS['grey']}; font-size: 0.8rem; }}
        .rc-card-value {{ color: {COLORS['navy']}; font-size: 1.65rem;
                          font-weight: 700; line-height: 1.25; }}
        .rc-card-note {{ color: {COLORS['grey']}; font-size: 0.72rem; }}
        .rc-badge {{
            display: inline-block; padding: 2px 10px; border-radius: 12px;
            color: white; font-size: 0.78rem; font-weight: 600;
        }}
        .rc-demo {{
            border-left: 6px solid {COLORS['orange']}; background: #FFF6E5;
            padding: 8px 14px; border-radius: 6px; margin-bottom: 12px;
            color: {COLORS['navy']}; font-size: 0.9rem;
        }}
        .rc-warn {{
            border-left: 6px solid {COLORS['red']}; background: #FDECEC;
            padding: 8px 14px; border-radius: 6px; margin-bottom: 12px;
            color: {COLORS['navy']}; font-size: 0.9rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9,
    })


def banner(title, subtitle):
    """Bandeau de titre de page."""

    st.markdown(
        f"""<div class="rc-banner"><h2>{html.escape(title)}</h2>
        <p>{html.escape(subtitle)}</p></div>""",
        unsafe_allow_html=True,
    )


def kpi_card(label, value, color=None, note=""):
    """Carte indicateur (HTML)."""

    color = color or COLORS["blue"]

    return (
        f'<div class="rc-card" style="border-left:6px solid {color}">'
        f'<div class="rc-card-label">{html.escape(str(label))}</div>'
        f'<div class="rc-card-value">{html.escape(str(value))}</div>'
        f'<div class="rc-card-note">{html.escape(str(note))}</div></div>'
    )


def kpi_row(cards):
    """Affiche une rangée de cartes indicateurs."""

    columns = st.columns(len(cards))

    for column, card in zip(columns, cards):
        column.markdown(card, unsafe_allow_html=True)


def badge(text, color):
    """Pastille de couleur (HTML)."""

    return (
        f'<span class="rc-badge" style="background:{color}">'
        f'{html.escape(str(text))}</span>'
    )


def show_table(table, **kwargs):
    """Affiche un tableau pleine largeur (compatible anciennes et nouvelles versions)."""

    try:
        st.dataframe(table, width="stretch", hide_index=True, **kwargs)
    except Exception:
        st.dataframe(table, use_container_width=True, hide_index=True, **kwargs)


def show_figure(fig):
    """Affiche un graphique matplotlib puis le ferme (libère la mémoire)."""

    st.pyplot(fig)
    plt.close(fig)


def draw_lines(df, series, ylabel, title, events=None, height=3.4):
    """
    Courbes dans le temps avec unité sur l'axe vertical.
    series : liste de (colonne, légende, couleur, style de trait).
    events : tableau d'événements à surligner en rouge clair.
    """

    fig, ax = plt.subplots(figsize=(10, height))

    for column, label, color, style in series:
        ax.plot(
            df["timestamp"], df[column],
            label=label, color=color, linestyle=style, linewidth=1.4,
        )

    if events is not None and len(events):
        for _, event in events.iterrows():
            ax.axvspan(event["start"], event["end"],
                       color=COLORS["red"], alpha=0.12)

    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11, color=COLORS["navy"])
    ax.legend(loc="upper left", ncol=4, frameon=False, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    show_figure(fig)


def draw_bars(labels, values, colors, ylabel, title, height=3.0):
    """Diagramme en barres simple avec les valeurs écrites au-dessus."""

    fig, ax = plt.subplots(figsize=(6, height))
    positions = range(len(labels))
    ax.bar(positions, values, color=colors, width=0.6)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11, color=COLORS["navy"])
    ax.grid(axis="x", visible=False)

    top = max([v for v in values if pd.notna(v)] + [1])
    ax.set_ylim(0, top * 1.15)
    for position, value in zip(positions, values):
        if pd.notna(value):
            ax.text(position, value + top * 0.02, f"{value:,.0f}".replace(",", " "),
                    ha="center", fontsize=8)

    fig.tight_layout()
    show_figure(fig)


def draw_grouped_bars(pivot, ylabel, title, height=3.4):
    """Barres groupées : une série par colonne du tableau."""

    fig, ax = plt.subplots(figsize=(10, height))

    n_series = max(len(pivot.columns), 1)
    width = 0.8 / n_series
    palette = [COLORS["navy"], COLORS["turquoise"], COLORS["blue"],
               COLORS["light"], COLORS["orange"], COLORS["grey"]]

    for i, column in enumerate(pivot.columns):
        positions = np.arange(len(pivot)) + i * width
        ax.bar(positions, pivot[column].values, width=width,
               label=str(column), color=palette[i % len(palette)])

    ax.set_xticks(np.arange(len(pivot)) + width * (n_series - 1) / 2)
    ax.set_xticklabels([str(i) for i in pivot.index])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11, color=COLORS["navy"])
    ax.legend(frameon=False, fontsize=8, ncol=4)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    show_figure(fig)


def draw_venn(flags):
    """
    Diagramme de Venn à 3 ensembles, dessiné avec matplotlib
    (aucune bibliothèque supplémentaire nécessaire).
    flags : dictionnaire {nom du modèle: série de booléens}.
    """

    names = list(flags)
    a, b, c = (flags[name].to_numpy() for name in names)

    counts = {
        "a": int((a & ~b & ~c).sum()),
        "b": int((~a & b & ~c).sum()),
        "c": int((~a & ~b & c).sum()),
        "ab": int((a & b & ~c).sum()),
        "ac": int((a & ~b & c).sum()),
        "bc": int((~a & b & c).sum()),
        "abc": int((a & b & c).sum()),
    }

    fig, ax = plt.subplots(figsize=(5, 4.2))

    centers = [(-0.5, 0.3), (0.5, 0.3), (0.0, -0.5)]
    colors = [COLORS["blue"], COLORS["turquoise"], COLORS["orange"]]

    for center, color in zip(centers, colors):
        ax.add_patch(Circle(center, 0.9, color=color, alpha=0.30))

    positions = {
        "a": (-0.95, 0.5), "b": (0.95, 0.5), "c": (0.0, -1.0),
        "ab": (0.0, 0.75), "ac": (-0.5, -0.15), "bc": (0.5, -0.15),
        "abc": (0.0, 0.2),
    }
    for key, (x, y) in positions.items():
        ax.text(x, y, str(counts[key]), ha="center", va="center",
                fontsize=11, color=COLORS["navy"], fontweight="bold")

    ax.text(-1.15, 1.3, names[0], ha="center", fontsize=9, color=COLORS["navy"])
    ax.text(1.15, 1.3, names[1], ha="center", fontsize=9, color=COLORS["navy"])
    ax.text(0.0, -1.65, names[2], ha="center", fontsize=9, color=COLORS["navy"])

    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.9, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Mesures détectées par chaque modèle", loc="left",
                 fontsize=11, color=COLORS["navy"])
    fig.tight_layout()
    show_figure(fig)


def pick_frequency(span):
    """Choisit un pas de regroupement adapté à la durée des données."""

    if span <= pd.Timedelta(hours=3):
        return "5min"

    if span <= pd.Timedelta(days=2):
        return "1h"

    return "1D"


def events_display(events):
    """Tableau d'événements avec des colonnes lisibles (en français)."""

    if events.empty:
        return pd.DataFrame()

    table = pd.DataFrame({
        "Date": events["start"].dt.strftime("%d/%m/%Y %H:%M"),
        "Site": events["site"],
        "Équipement": events["equipment_id"],
        "Catégorie": events["category"],
        "Modèles ayant voté": (
            events["votes"].astype(str) + "/3 : " + events["models_voted"]
        ),
        "Indice d'accord entre les modèles (%)": events["agreement"],
        "Priorité": events["priority"],
        "Statut": events["status"],
        "Durée": events["duration_min"].apply(format_duration),
        "Cause probable": events["probable_cause"],
        "Action recommandée": events["recommendation"],
    })

    return table


def event_label(event):
    """Texte court pour choisir un événement dans une liste."""

    return (
        f"{format_datetime(event['start'])} - {event['equipment_id']} - "
        f"{event['priority']} - {event['status']}"
    )


# ============================================================
# 9. LES 10 PAGES
# ============================================================

def render_data_warnings(ctx):
    """Messages d'avertissement communs à toutes les pages."""

    quality = ctx["analysis"]["quality"]["summary"]
    n_rows = len(ctx["analysis"]["data"])

    if ctx["is_demo"]:
        st.markdown(
            '<div class="rc-demo"><b>Données de démonstration</b> : '
            f'{html.escape(ctx["source_label"])}. Ces données ne proviennent '
            "pas d'équipements réels.</div>",
            unsafe_allow_html=True,
        )

    if quality["dq_score"] < DQ_LIMIT:
        st.markdown(
            f'<div class="rc-warn"><b>Qualité des données insuffisante</b> '
            f'(Data Quality Score : {quality["dq_score"]:.0f}/100). '
            "Les diagnostics sont à considérer avec prudence.</div>",
            unsafe_allow_html=True,
        )

    if n_rows < 50:
        st.markdown(
            '<div class="rc-warn"><b>Historique trop court</b> : '
            f"{n_rows} mesures seulement. Les modèles IA ne peuvent pas "
            "se mettre d'accord de façon fiable ; ce résultat sert à tester "
            "l'application, pas à conseiller un client.</div>",
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------
# Page 1 : Vue d'ensemble
# ------------------------------------------------------------

def page_overview(ctx):
    banner("Vue d'ensemble",
           "Synthèse des équipements, des anomalies et des priorités")
    render_data_warnings(ctx)

    view = ctx["view"]
    eq = view["equipment"]
    events = view["events_open"]
    cfg = ctx["cfg"]

    if eq.empty:
        st.warning("Aucun équipement ne correspond aux filtres choisis.")
        return

    status_counts = eq["status"].value_counts()
    n_sites = eq[["client", "site"]].drop_duplicates().shape[0]
    total_kwh = eq["energy_kwh"].sum()
    avoidable = eq["avoidable_kwh"].sum()

    avoidable_note = "Indicatif : hors occupation + puissance au-dessus du niveau habituel"
    if cfg["price_kwh"] > 0:
        avoidable_note += (
            f" (≈ {format_number(avoidable * cfg['price_kwh'], 0, cfg['currency'])})"
        )

    kpi_row([
        kpi_card("Sites", n_sites, COLORS["navy"]),
        kpi_card("Équipements", len(eq), COLORS["navy"]),
        kpi_card("Fonctionnement normal", int(status_counts.get("Normal", 0)),
                 COLORS["green"]),
        kpi_card("À surveiller", int(status_counts.get("À surveiller", 0)),
                 COLORS["orange"]),
        kpi_card("Anomalies probables", int(status_counts.get("Anomalie probable", 0)),
                 "#E67E22", "équipements concernés"),
    ])
    kpi_row([
        kpi_card("Anomalies fortes", int(status_counts.get("Anomalie forte", 0)),
                 COLORS["red"], "équipements concernés"),
        kpi_card("Alertes critiques",
                 int((events["priority"] == "Critique").sum()),
                 COLORS["red"], "événements ouverts"),
        kpi_card("Consommation totale", format_number(total_kwh, 0, "kWh"),
                 COLORS["blue"], "puissance intégrée dans le temps"),
        kpi_card("Énergie potentiellement évitable",
                 format_number(avoidable, 0, "kWh") if avoidable > 0 else "n/a",
                 COLORS["turquoise"], avoidable_note),
    ])

    left, right = st.columns([3, 2])

    with left:
        data = view["data"]
        freq = pick_frequency(data["timestamp"].max() - data["timestamp"].min())
        counts = (
            data.assign(flag=data["in_open_event"].astype(int))
            .set_index("timestamp")["flag"]
            .resample(freq).sum()
        )

        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.bar(range(len(counts)), counts.values, color=COLORS["orange"])
        step = max(1, len(counts) // 8)
        ticks = list(range(0, len(counts), step))
        label_format = "%d/%m" if freq == "1D" else "%d/%m %H:%M"
        ax.set_xticks(ticks)
        ax.set_xticklabels([counts.index[i].strftime(label_format) for i in ticks],
                           rotation=30, ha="right")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("Mesures en anomalie (nombre)")
        ax.set_title("Évolution des anomalies persistantes", loc="left",
                     fontsize=11, color=COLORS["navy"])
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        show_figure(fig)

    with right:
        counts = [int((events["priority"] == p).sum()) for p in PRIORITY_ORDER]
        draw_bars(PRIORITY_ORDER, counts,
                  [PRIORITY_COLORS[p] for p in PRIORITY_ORDER],
                  "Événements ouverts (nombre)", "Répartition par priorité")

    left, right = st.columns(2)

    with left:
        by_type = events.groupby("equipment_type").size()
        if len(by_type):
            draw_bars(list(by_type.index), list(by_type.values),
                      [COLORS["blue"]] * len(by_type),
                      "Événements ouverts (nombre)",
                      "Répartition par type d'équipement")
        else:
            st.info("Aucun événement ouvert à répartir par type d'équipement.")

    with right:
        st.subheader("Équipements au plus faible Health Score")
        lowest = eq.sort_values("health_score").head(5)
        show_table(
            pd.DataFrame({
                "Équipement": lowest["equipment_id"],
                "Site": lowest["site"],
                "Health Score (/100)": lowest["health_score"].round(0),
                "Statut": lowest["status"],
            })
        )

    st.subheader("Les 5 anomalies les plus importantes")
    if events.empty:
        st.success("Aucune anomalie persistante ouverte avec la configuration actuelle.")
    else:
        top = events.sort_values(
            ["priority_score", "duration_min"], ascending=False
        ).head(5)
        show_table(events_display(top))

        st.subheader("Résumé des recommandations prioritaires")
        for _, event in top.head(3).iterrows():
            st.markdown(
                f"- **{event['equipment_id']}** ({format_datetime(event['start'])}, "
                f"priorité {event['priority'].lower()}) : {event['recommendation']}"
            )


# ------------------------------------------------------------
# Page 2 : Sites
# ------------------------------------------------------------

def page_sites(ctx):
    banner("Sites", "Scores et indicateurs par site")
    render_data_warnings(ctx)

    view = ctx["view"]
    sites = build_site_table(view["equipment"], view["events_open"])

    if sites.empty:
        st.warning("Aucun site ne correspond aux filtres choisis.")
        return

    if (sites["site"] == "Non renseigné").any():
        st.info(
            "Le fichier ne contient pas de colonne « site » ou « client » : "
            "les valeurs sont affichées « Non renseigné ». Tu pourras "
            "associer ces colonnes dans la page Configuration."
        )

    show_table(pd.DataFrame({
        "Client": sites["client"],
        "Site": sites["site"],
        "Ville ou localisation": sites["city"],
        "Équipements": sites["equipments"],
        "Score de confort (/100)": sites["comfort_score"].round(0),
        "Score énergétique (/100)": sites["energy_score"].round(0),
        "Health Score moyen (/100)": sites["health_score"].round(0),
        "Anomalies ouvertes": sites["open_anomalies"],
        "Consommation (kWh)": sites["energy_kwh"].round(0),
        "Dernière mise à jour": sites["last_update"].dt.strftime("%d/%m/%Y %H:%M"),
    }))

    st.caption(
        "Le score énergétique n'est calculable que s'il existe au moins deux "
        "équipements du même type à comparer (sinon « n/a », car il n'y a "
        "aucune référence)."
    )

    with st.expander("Comment sont calculés les scores ?"):
        st.markdown(
            "- **Score de confort** : pourcentage du temps d'occupation pendant "
            "lequel l'écart à la consigne reste dans la tolérance configurée.\n"
            "- **Score énergétique** : puissance moyenne en occupation comparée "
            "à la médiane des équipements du même type.\n"
            "- **Health Score** : 100 moins des pénalités (anomalies persistantes, "
            "puissance élevée, dérive de régulation, qualité des données). "
            "Voir la page Équipements pour le détail."
        )


# ------------------------------------------------------------
# Page 3 : Équipements
# ------------------------------------------------------------

def page_equipment(ctx):
    banner("Équipements", "Fiche détaillée, courbes, énergie et historique")
    render_data_warnings(ctx)

    view = ctx["view"]
    eq = view["equipment"]
    data = view["data"]

    if eq.empty:
        st.warning("Aucun équipement ne correspond aux filtres choisis.")
        return

    selected = st.selectbox("Équipement", list(eq["equipment_id"]))
    row = eq[eq["equipment_id"] == selected].iloc[0]
    g = data[data["equipment_id"] == selected]
    ev_all = view["events_all"]
    ev_eq = ev_all[ev_all["equipment_id"] == selected]
    ev_open = view["events_open"]
    ev_open = ev_open[ev_open["equipment_id"] == selected]

    tabs = st.tabs([
        "Vue générale",
        "Courbes des températures",
        "Énergie et puissance",
        "Anomalies et événements",
        "Recommandations",
        "Historique des actions",
    ])

    with tabs[0]:
        color = EQUIPMENT_STATUS_COLORS.get(row["status"], COLORS["grey"])
        st.markdown(
            f"**{row['equipment_id']}** &nbsp; {badge(row['status'], color)}",
            unsafe_allow_html=True,
        )

        kpi_row([
            kpi_card("Type", row["equipment_type"]),
            kpi_card("Marque et modèle", row["brand_model"]),
            kpi_card("Site", row["site"]),
            kpi_card("Communication",
                     "Active" if row["communication"] == "Active" else "Interrompue",
                     COLORS["green"] if row["communication"] == "Active"
                     else COLORS["red"],
                     "" if row["communication"] == "Active" else row["communication"]),
        ])
        kpi_row([
            kpi_card("État de fonctionnement", row["operating"]),
            kpi_card("Température de zone", format_number(row["temperature"], 1, "°C"),
                     COLORS["navy"], "dernière mesure"),
            kpi_card("Consigne", format_number(row["setpoint"], 1, "°C"),
                     COLORS["navy"]),
            kpi_card("Puissance", format_number(row["power"], 1, "kW"),
                     COLORS["navy"], "dernière mesure"),
        ])
        kpi_row([
            kpi_card("Position de vanne", format_number(row["valve"], 0, "%"),
                     COLORS["navy"]),
            kpi_card("Alarmes",
                     format_number(row["alarm_count"], 0)
                     if pd.notna(row["alarm_count"]) else "Non disponible",
                     COLORS["navy"]),
            kpi_card("Health Score", format_number(row["health_score"], 0, "/100"),
                     COLORS["green"] if row["health_score"] >= 80
                     else COLORS["orange"] if row["health_score"] >= 60
                     else COLORS["red"]),
            kpi_card("Dernière anomalie ouverte", format_datetime(row["last_anomaly"])
                     if pd.notna(row["last_anomaly"]) else "Aucune"),
        ])
        st.caption(
            f"Dernière maintenance : {row['last_maintenance']}. "
            f"Dernière donnée reçue : {format_datetime(row['last_timestamp'])}. "
            f"Score de confort : {format_number(row['comfort_score'], 0, '%')}."
        )

        with st.expander("Comment est calculé le Health Score ?"):
            st.markdown(
                "Le Health Score part de 100 et retire des points :\n"
                "- **Anomalies persistantes** : 2 points par % de mesures concernées (maximum 40) ;\n"
                "- **Puissance élevée** : 1 point par % de mesures au-dessus du seuil (maximum 15) ;\n"
                "- **Dérive de régulation** : 5 points par °C d'écart moyen à la consigne "
                "en occupation (maximum 20) ;\n"
                "- **Qualité des données** : (100 − Data Quality Score) / 5 (maximum 20).\n\n"
                "Les alarmes et les cycles courts ne sont pas pris en compte dans cette "
                "version (variables absentes du fichier). Ces pondérations sont "
                "**indicatives** et à valider avec les experts Redge."
            )

    with tabs[1]:
        draw_lines(
            g,
            [
                ("zone_temperature", "Température de zone", COLORS["navy"], "-"),
                ("setpoint", "Consigne", COLORS["grey"], "--"),
                ("supply_temperature", "Soufflage", COLORS["turquoise"], "-"),
                ("return_temperature", "Reprise", COLORS["blue"], "-"),
                ("outdoor_temperature", "Extérieure", COLORS["light"], ":"),
            ],
            "Température (°C)", f"Températures - {selected}", events=ev_open,
        )
        draw_lines(
            g, [("valve_position", "Position de vanne", COLORS["blue"], "-")],
            "Ouverture (%)", f"Vanne - {selected}", events=ev_open, height=2.6,
        )
        st.caption("Les zones rouges clair correspondent aux anomalies persistantes ouvertes.")

    with tabs[2]:
        draw_lines(
            g, [("electrical_power", "Puissance électrique", COLORS["navy"], "-")],
            "Puissance (kW)", f"Puissance - {selected}", events=ev_open,
        )
        st.caption(
            f"Énergie sur la période : {format_number(row['energy_kwh'], 0, 'kWh')} "
            "(puissance supposée en kW, intégrée dans le temps)."
        )

    with tabs[3]:
        if ev_eq.empty:
            st.success("Aucun événement détecté pour cet équipement.")
        else:
            table = events_display(ev_eq)
            table.insert(
                1, "Persistant",
                np.where(ev_eq["persistent"], "Oui", "Non (trop court)"),
            )
            show_table(table)

    with tabs[4]:
        if ev_open.empty:
            st.success("Aucune recommandation en cours pour cet équipement.")
        for _, event in ev_open.sort_values("priority_score", ascending=False).iterrows():
            with st.container(border=True):
                st.markdown(
                    f"**{event['category']}** - {format_datetime(event['start'])} "
                    f"&nbsp; {badge(event['priority'], PRIORITY_COLORS[event['priority']])}",
                    unsafe_allow_html=True,
                )
                st.write(f"**Cause probable :** {event['probable_cause']}")
                st.write(f"**Action recommandée :** {event['recommendation']}")

    with tabs[5]:
        history = ev_eq[ev_eq["status"] != "Nouvelle"]
        if history.empty:
            st.info("Aucune action enregistrée pour cet équipement.")
        else:
            show_table(pd.DataFrame({
                "Événement": history["start"].dt.strftime("%d/%m/%Y %H:%M"),
                "Statut": history["status"],
                "Validation du diagnostic": history["validation"],
                "Cause réelle": history["real_cause"],
                "Action réalisée": history["action_done"],
                "Commentaire": history["comment"],
                "Date de résolution": history["resolution_date"],
                "Mis à jour le": history["updated_at"],
            }))


# ------------------------------------------------------------
# Page 4 : Anomalies
# ------------------------------------------------------------

def page_anomalies(ctx):
    banner("Anomalies",
           "Événements persistants, diagnostic probable et validation par le technicien")
    render_data_warnings(ctx)

    view = ctx["view"]
    events = view["events_all"][view["events_all"]["persistent"]]
    cfg = ctx["cfg"]

    if events.empty:
        st.success(
            "Aucune anomalie persistante avec la configuration actuelle "
            f"(durée minimale : {cfg['min_duration']} min)."
        )
        short = view["events_all"][~view["events_all"]["persistent"]]
        if len(short):
            st.info(
                f"{len(short)} événement(s) trop court(s) ont été écartés. "
                "Réduis la durée minimale dans Configuration pour les voir."
            )
        return

    column_1, column_2 = st.columns(2)
    priorities = column_1.multiselect(
        "Priorité (vide = toutes)", PRIORITY_ORDER)
    statuses = column_2.multiselect("Statut (vide = tous)", STATUSES)

    filtered = events
    if priorities:
        filtered = filtered[filtered["priority"].isin(priorities)]
    if statuses:
        filtered = filtered[filtered["status"].isin(statuses)]

    filtered = filtered.sort_values(
        ["priority_score", "start"], ascending=[False, False]
    )

    st.write(f"**{len(filtered)}** anomalie(s) affichée(s).")

    if filtered.empty:
        st.info("Aucune anomalie ne correspond à ces filtres.")
        return

    show_table(events_display(filtered))

    # ---- Détail d'une anomalie ----
    st.subheader("Ouvrir une anomalie")

    labels = {row["event_id"]: event_label(row) for _, row in filtered.iterrows()}
    chosen = st.selectbox(
        "Choisir une anomalie", list(labels), format_func=lambda key: labels[key]
    )
    event = filtered[filtered["event_id"] == chosen].iloc[0]

    st.markdown(
        f"**Équipement :** {event['equipment_id']} ({event['site']})  \n"
        f"**Date :** {format_datetime(event['start'])} → "
        f"{format_datetime(event['end'])} "
        f"(durée : {format_duration(event['duration_min'])})  \n"
        f"**Résultat IA :** {event['votes']} modèle(s) sur 3 ont détecté un "
        f"comportement inhabituel ({event['models_voted']}).  \n"
        f"**Indice d'accord entre les modèles :** {event['agreement']} %"
    )

    st.markdown("**Observations :**")
    for item in event["observation"].split(" | "):
        st.markdown(f"- {item}")
    st.markdown(
        f"- Température de zone : {format_number(event['temp_actual'], 1, '°C')} "
        f"(attendue : {format_number(event['temp_expected'], 1, '°C')}, "
        f"écart {event['temp_gap']:+.1f} °C)".replace(".", ",")
    )
    if pd.notna(event["power_gap_pct"]):
        st.markdown(
            f"- Puissance : {format_number(event['power_actual'], 1, 'kW')} "
            f"(niveau habituel ≈ {format_number(event['power_expected'], 1, 'kW')}, "
            f"écart {event['power_gap_pct']:+.0f} %)".replace(".", ",")
        )

    st.markdown(
        f"**Variables anormales :** {event['abnormal_variables']}  \n"
        f"**Règle(s) HVAC déclenchée(s) :** {event['rules']}  \n"
        f"**Cause probable :** {event['probable_cause']}  \n"
        f"**Action recommandée :** {event['recommendation']}  \n"
        f"**Priorité :** {event['priority']}  \n"
        f"**Fiabilité du diagnostic :** {event['reliability']}"
    )
    st.caption(event["limits"])

    # ---- Retour du technicien ----
    st.subheader("Retour du technicien")

    def current(field, default=""):
        value = event.get(field, default)
        return default if pd.isna(value) else value

    with st.form(f"feedback_{chosen}"):
        status = st.selectbox(
            "Statut", STATUSES, index=STATUSES.index(current("status", "Nouvelle")))
        validation_options = ["Non évalué", "Diagnostic confirmé", "Diagnostic rejeté"]
        validation = st.radio(
            "Validation du diagnostic", validation_options, horizontal=True,
            index=validation_options.index(current("validation", "Non évalué"))
            if current("validation", "Non évalué") in validation_options else 0,
        )
        real_cause = st.text_input("Cause réelle", value=current("real_cause"))
        action_done = st.text_area("Action réalisée", value=current("action_done"))
        comment = st.text_area("Commentaire du technicien", value=current("comment"))
        resolution = st.date_input("Date de résolution (facultative)", value=None)
        submitted = st.form_submit_button("Enregistrer")

    if submitted:
        save_feedback(
            chosen,
            status=status,
            validation=validation,
            real_cause=real_cause,
            action_done=action_done,
            comment=comment,
            resolution_date=resolution.strftime("%d/%m/%Y") if resolution else "",
        )
        st.success("Retour enregistré dans retours_techniciens.csv.")
        st.rerun()


# ------------------------------------------------------------
# Page 5 : Recommandations
# ------------------------------------------------------------

def page_recommendations(ctx):
    banner("Recommandations",
           "Actions proposées, à valider par un technicien avant toute intervention")
    render_data_warnings(ctx)

    view = ctx["view"]
    events = view["events_all"][view["events_all"]["persistent"]]

    show_closed = st.checkbox("Afficher aussi les anomalies résolues ou faux positifs")
    if not show_closed:
        events = events[~events["status"].isin(CLOSED_STATUSES)]

    events = events.sort_values("priority_score", ascending=False)

    if events.empty:
        st.success("Aucune recommandation en cours.")
        return

    maximum = st.slider("Nombre de cartes à afficher", 1, min(20, len(events)),
                        min(6, len(events)))
    cfg = ctx["cfg"]

    for _, event in events.head(maximum).iterrows():
        key = event["event_id"]

        with st.container(border=True):
            st.markdown(
                f"**{event['category']} - {event['equipment_id']}** &nbsp; "
                f"{badge(event['priority'], PRIORITY_COLORS[event['priority']])} "
                f"&nbsp; {badge(event['status'], COLORS['grey'])}",
                unsafe_allow_html=True,
            )

            left, right = st.columns([3, 2])
            with left:
                st.write(f"**Preuve principale :** {event['observation']}")
                st.write(f"**Cause probable :** {event['probable_cause']}")
                st.write(f"**Action recommandée :** {event['recommendation']}")
            with right:
                st.write(
                    f"**Indice d'accord entre les modèles :** {event['agreement']} %"
                )
                if event["avoidable_kwh"] > 0:
                    gain = f"≈ {format_number(event['avoidable_kwh'], 0, 'kWh')} potentiellement évitables"
                    if cfg["price_kwh"] > 0:
                        gain += (
                            f" (≈ {format_number(event['avoidable_kwh'] * cfg['price_kwh'], 0, cfg['currency'])})"
                        )
                else:
                    gain = "non calculable"
                st.write(f"**Gain énergétique éventuel :** {gain}")
                st.write(f"**Période :** {format_datetime(event['start'])}")

            with st.expander("Voir les détails"):
                st.markdown(
                    f"- **Données utilisées :** température de zone, consigne, "
                    f"température de soufflage et de reprise, puissance, vanne, "
                    f"occupation.\n"
                    f"- **Règles HVAC déclenchées :** {event['rules']}\n"
                    f"- **Modèles ayant voté :** {event['models_voted']} "
                    f"({event['votes']}/3)\n"
                    f"- **Niveau d'accord :** {event['agreement']} % "
                    f"(ce n'est pas une probabilité de panne)\n"
                    f"- **Limites du diagnostic :** {event['limits']}"
                )

            b1, b2, b3, _ = st.columns([1, 1, 1.4, 3])
            if b1.button("Confirmer", key=f"confirm_{key}"):
                save_feedback(key, status="Confirmée", validation="Diagnostic confirmé")
                st.rerun()
            if b2.button("Faux positif", key=f"false_{key}"):
                save_feedback(key, status="Faux positif", validation="Diagnostic rejeté")
                st.rerun()
            if b3.button("Intervention réalisée", key=f"done_{key}"):
                save_feedback(
                    key, status="Résolue",
                    resolution_date=date.today().strftime("%d/%m/%Y"),
                )
                st.rerun()


# ------------------------------------------------------------
# Page 6 : Comparaison des modèles
# ------------------------------------------------------------

def page_models(ctx):
    banner("Comparaison des modèles",
           "Comprendre pourquoi une mesure est retenue ou rejetée")
    render_data_warnings(ctx)

    data = ctx["view"]["data"]
    cfg = ctx["cfg"]
    rule = cfg["vote_rule"]

    flags = {name: data[f"{safe}_anomaly"] for name, safe in MODELS}
    confirmed = data["model_votes"] >= rule
    disagreement = (data["model_votes"] >= 1) & (data["model_votes"] < 3)

    kpi_row([
        kpi_card(name, int(flag.sum()), COLORS["blue"], "mesures détectées")
        for name, flag in flags.items()
    ] + [
        kpi_card(f"Confirmées par {rule} modèle(s) sur 3", int(confirmed.sum()),
                 COLORS["red"]),
        kpi_card("Désaccords", int(disagreement.sum()), COLORS["orange"],
                 "au moins 1 modèle mais pas les 3"),
    ])

    left, right = st.columns(2)

    with left:
        draw_bars(
            list(flags), [int(f.sum()) for f in flags.values()],
            [COLORS["blue"], COLORS["turquoise"], COLORS["orange"]],
            "Mesures détectées (nombre)", "Anomalies par modèle",
        )
    with right:
        draw_venn(flags)

    # ---- Accord moyen entre les modèles ----
    names = list(flags)
    overlaps = []
    for i in range(3):
        for j in range(i + 1, 3):
            union = (flags[names[i]] | flags[names[j]]).sum()
            if union > 0:
                overlaps.append(
                    (flags[names[i]] & flags[names[j]]).sum() / union
                )
    mean_overlap = 100 * float(np.mean(overlaps)) if overlaps else np.nan

    events = ctx["view"]["events_all"]
    validated = events[events["status"].isin(["Confirmée", "Faux positif", "Résolue"])]
    false_positives = int((events["status"] == "Faux positif").sum())

    st.markdown(
        f"**Accord moyen entre les modèles :** {format_number(mean_overlap, 0, '%')} "
        "(part des mesures signalées par un modèle qui sont aussi signalées par un autre).  \n"
        f"**Faux positifs après validation technicien :** {false_positives} "
        f"sur {len(validated)} anomalie(s) validée(s)."
    )

    st.subheader("Tableau comparatif des scores")
    rows = []
    for name, safe in MODELS:
        score = data[f"{safe}_score"]
        rows.append({
            "Modèle": name,
            "Mesures détectées": int(data[f"{safe}_anomaly"].sum()),
            "Part des mesures (%)": round(100 * data[f"{safe}_anomaly"].mean(), 1),
            "Score moyen": round(float(score.mean()), 3),
            "Score maximum": round(float(score.max()), 3),
        })
    show_table(pd.DataFrame(rows))
    st.caption(
        "Les scores des trois modèles n'ont pas la même échelle : on compare "
        "les mesures détectées, pas les valeurs de score entre modèles."
    )

    # ---- Listes séparées ----
    st.subheader("Détail des mesures")
    tabs = st.tabs(list(flags) + ["Confirmées", "Désaccords"])

    def detail_table(mask):
        table = data[mask].sort_values("model_votes", ascending=False).head(300)
        return pd.DataFrame({
            "Date": table["timestamp"].dt.strftime("%d/%m/%Y %H:%M"),
            "Équipement": table["equipment_id"],
            "Écart consigne (°C)": table["temperature_error"].round(1),
            "Puissance (kW)": table["electrical_power"].round(1),
            "Vanne (%)": table["valve_position"].round(0),
            "Votes (/3)": table["model_votes"],
        })

    masks = list(flags.values()) + [confirmed, disagreement]
    for tab, mask in zip(tabs, masks):
        with tab:
            if mask.sum() == 0:
                st.info("Aucune mesure dans cette catégorie.")
            else:
                show_table(detail_table(mask))
                if mask.sum() > 300:
                    st.caption("Les 300 premières mesures sont affichées.")

    # ---- Pourquoi retenue ou rejetée ? ----
    st.subheader("Pourquoi cette mesure est-elle retenue ou rejetée ?")

    candidates = data[data["model_votes"] >= 1].sort_values(
        ["model_votes", "timestamp"], ascending=[False, True]
    ).head(300)

    if candidates.empty:
        st.info("Aucun modèle n'a signalé de mesure.")
        return

    chosen = st.selectbox(
        "Choisir une mesure",
        list(candidates.index),
        format_func=lambda i: (
            f"{format_datetime(data.loc[i, 'timestamp'])} - "
            f"{data.loc[i, 'equipment_id']} - {data.loc[i, 'model_votes']}/3 votes"
        ),
    )
    row = data.loc[chosen]

    for name, safe in MODELS:
        verdict = "a signalé un comportement inhabituel" if row[f"{safe}_anomaly"] \
            else "n'a rien signalé"
        st.markdown(f"- **{name}** {verdict} (score : {row[f'{safe}_score']:.2f})")

    if row["model_votes"] >= rule:
        st.success(
            f"Mesure **retenue** : {int(row['model_votes'])} modèle(s) sur 3 "
            f"sont d'accord (règle de vote : au moins {rule})."
        )
    else:
        st.warning(
            f"Mesure **rejetée** comme anomalie IA : seulement "
            f"{int(row['model_votes'])} modèle(s) sur 3 (règle de vote : au moins {rule}). "
            "Elle reste à surveiller."
        )


# ------------------------------------------------------------
# Page 7 : Analyse énergétique
# ------------------------------------------------------------

def page_energy(ctx):
    banner("Analyse énergétique",
           "Puissance, consommation et énergie potentiellement évitable")
    render_data_warnings(ctx)

    view = ctx["view"]
    data = view["data"].copy()
    eq = view["equipment"]
    cfg = ctx["cfg"]

    if eq.empty:
        st.warning("Aucun équipement ne correspond aux filtres choisis.")
        return

    st.info(
        "Aucune énergie cumulée n'est disponible dans le fichier : la "
        "consommation est calculée en intégrant la puissance dans le temps "
        "(puissance supposée en kW x durée entre deux mesures). Les "
        "interruptions de communication ne sont pas comblées. "
        "Aucun prix du kWh n'est supposé : il se règle dans Configuration."
    )

    data["kwh"] = data["electrical_power"] * data["dt_min"] / 60

    total = eq["energy_kwh"].sum()
    out_of_occupancy = eq["energy_out_of_occupancy_kwh"].sum()
    cards = [
        kpi_card("Consommation totale", format_number(total, 0, "kWh"), COLORS["blue"]),
        kpi_card("Hors occupation", format_number(out_of_occupancy, 0, "kWh"),
                 COLORS["orange"],
                 f"{format_number(100 * out_of_occupancy / total, 0, '%')} du total"
                 if total > 0 else ""),
        kpi_card("Puissance maximale", format_number(eq["max_power"].max(), 1, "kW"),
                 COLORS["navy"]),
        kpi_card("Énergie potentiellement évitable",
                 format_number(eq["avoidable_kwh"].sum(), 0, "kWh")
                 if eq["avoidable_kwh"].sum() > 0 else "n/a",
                 COLORS["turquoise"], "indicatif, à valider"),
    ]
    if cfg["price_kwh"] > 0:
        cards.append(kpi_card(
            "Coût estimé", format_number(total * cfg["price_kwh"], 0, cfg["currency"]),
            COLORS["navy"], f"au prix de {format_number(cfg['price_kwh'], 3)} {cfg['currency']}/kWh",
        ))
    if cfg["co2_factor"] > 0:
        cards.append(kpi_card(
            "Émissions de CO2 estimées",
            format_number(total * cfg["co2_factor"], 0, "kg"),
            COLORS["navy"], f"facteur : {format_number(cfg['co2_factor'], 3)} kg/kWh",
        ))
    kpi_row(cards)

    # ---- Puissance au cours du temps ----
    fig, ax = plt.subplots(figsize=(10, 3.4))
    palette = [COLORS["navy"], COLORS["turquoise"], COLORS["blue"],
               COLORS["orange"], COLORS["grey"], COLORS["light"]]
    for i, (equipment, g) in enumerate(data.groupby("equipment_id")):
        ax.plot(g["timestamp"], g["electrical_power"], label=equipment,
                color=palette[i % len(palette)], linewidth=1.2)
    ax.set_ylabel("Puissance (kW)")
    ax.set_title("Puissance électrique au cours du temps", loc="left",
                 fontsize=11, color=COLORS["navy"])
    ax.legend(frameon=False, fontsize=8, ncol=6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    show_figure(fig)

    # ---- Consommation quotidienne et hebdomadaire ----
    left, right = st.columns(2)

    daily = (
        data.assign(day=data["timestamp"].dt.strftime("%d/%m"))
        .pivot_table(index="day", columns="equipment_id", values="kwh",
                     aggfunc="sum", sort=False)
    )
    with left:
        draw_grouped_bars(daily, "Énergie (kWh)", "Consommation quotidienne")

    weekly = (
        data.assign(week=data["timestamp"].dt.to_period("W").dt.start_time
                    .dt.strftime("Sem. du %d/%m"))
        .pivot_table(index="week", columns="equipment_id", values="kwh",
                     aggfunc="sum", sort=False)
    )
    with right:
        draw_grouped_bars(weekly, "Énergie (kWh)", "Consommation hebdomadaire")

    # ---- Par équipement, occupé / hors occupation ----
    fig, ax = plt.subplots(figsize=(10, 3.2))
    positions = np.arange(len(eq))
    ax.bar(positions, eq["energy_occupied_kwh"], color=COLORS["navy"],
           label="En occupation")
    ax.bar(positions, eq["energy_out_of_occupancy_kwh"],
           bottom=eq["energy_occupied_kwh"], color=COLORS["orange"],
           label="Hors occupation")
    ax.set_xticks(positions)
    ax.set_xticklabels(eq["equipment_id"])
    ax.set_ylabel("Énergie (kWh)")
    ax.set_title("Consommation par équipement", loc="left", fontsize=11,
                 color=COLORS["navy"])
    ax.set_ylim(0, float(eq["energy_kwh"].max()) * 1.25)
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    show_figure(fig)

    # ---- Équipements les plus consommateurs et comparaison ----
    st.subheader("Équipements les plus consommateurs")
    ranking = eq.sort_values("energy_kwh", ascending=False)
    show_table(pd.DataFrame({
        "Équipement": ranking["equipment_id"],
        "Type": ranking["equipment_type"],
        "Consommation (kWh)": ranking["energy_kwh"].round(0),
        "Hors occupation (kWh)": ranking["energy_out_of_occupancy_kwh"].round(0),
        "Puissance maximale (kW)": ranking["max_power"].round(1),
        "Puissance moyenne en occupation (kW)": ranking["mean_power_occupied"].round(1),
        "Score énergétique (/100)": ranking["energy_score"].round(0),
        "Énergie potentiellement évitable (kWh)": ranking["avoidable_kwh"].round(0),
    }))

    if eq["energy_score"].notna().any():
        st.caption(
            "Comparaison entre équipements similaires : le score énergétique "
            "compare la puissance moyenne en occupation à la médiane des "
            "équipements du même type."
        )
    else:
        st.caption(
            "Comparaison entre équipements similaires impossible : il faut au "
            "moins deux équipements du même type."
        )


# ------------------------------------------------------------
# Page 8 : Qualité des données
# ------------------------------------------------------------

def page_quality(ctx):
    banner("Qualité des données",
           "Fiabilité des mesures utilisées par les modèles")
    render_data_warnings(ctx)

    quality = ctx["analysis"]["quality"]
    summary = quality["summary"]
    per_eq = quality["per_equipment"]

    score = summary["dq_score"]
    color = COLORS["green"] if score >= 85 else COLORS["orange"] if score >= DQ_LIMIT \
        else COLORS["red"]

    kpi_row([
        kpi_card("Data Quality Score", f"{score:.0f} / 100", color),
        kpi_card("Données disponibles", format_number(summary["completeness_pct"], 1, "%"),
                 COLORS["blue"]),
        kpi_card("Valeurs manquantes", summary["missing_values"], COLORS["orange"]),
        kpi_card("Valeurs hors plage", summary["out_of_range"], COLORS["orange"]),
    ])
    kpi_row([
        kpi_card("Valeurs bloquées", summary["stuck_values"], COLORS["orange"],
                 f"{STUCK_RUN} mesures identiques ou plus"),
        kpi_card("Doublons", summary["duplicates"], COLORS["orange"]),
        kpi_card("Interruptions de communication", summary["interruptions"],
                 COLORS["red"] if summary["interruptions"] else COLORS["green"]),
        kpi_card("Fréquence moyenne",
                 f"1 mesure / {format_number(summary['mean_step_min'], 0)} min",
                 COLORS["navy"],
                 f"dernière valeur : {format_datetime(summary['last_timestamp'])}"),
    ])

    st.subheader("Qualité par équipement")
    show_table(pd.DataFrame({
        "Équipement": per_eq["equipment_id"],
        "Mesures": per_eq["rows"],
        "Données disponibles (%)": per_eq["completeness_pct"].round(1),
        "Valeurs manquantes": per_eq["missing_values"],
        "Hors plage": per_eq["out_of_range"],
        "Valeurs bloquées": per_eq["stuck_values"],
        "Doublons": per_eq["duplicates"],
        "Interruptions": per_eq["interruptions"],
        "Dernière valeur reçue": per_eq["last_timestamp"].dt.strftime("%d/%m/%Y %H:%M"),
        "Data Quality Score (/100)": per_eq["dq_score"].round(0),
    }))

    left, right = st.columns(2)

    with left:
        st.subheader("Valeurs manquantes par variable")
        missing = quality["missing_by_column"]
        missing = missing[missing > 0]
        if missing.empty:
            st.success("Aucune valeur manquante.")
        else:
            show_table(pd.DataFrame({
                "Variable": missing.index, "Valeurs manquantes": missing.astype(int).values,
            }))

        st.subheader("Valeurs hors plage par variable")
        out_of_range = {k: v for k, v in quality["out_of_range_by_column"].items() if v > 0}
        if not out_of_range:
            st.success("Aucune valeur hors plage.")
        else:
            show_table(pd.DataFrame({
                "Variable": list(out_of_range),
                "Valeurs hors plage": list(out_of_range.values()),
            }))

    with right:
        st.subheader("Interruptions de communication")
        gaps = quality["gaps"]
        if gaps.empty:
            st.success("Aucune interruption détectée.")
        else:
            show_table(pd.DataFrame({
                "Équipement": gaps["equipment_id"],
                "Dernière mesure avant": gaps["start"].dt.strftime("%d/%m/%Y %H:%M"),
                "Reprise": gaps["end"].dt.strftime("%d/%m/%Y %H:%M"),
                "Durée": gaps["duration_min"].apply(format_duration),
            }))

        st.subheader("Valeurs bloquées par variable")
        stuck = {k: v for k, v in quality["stuck_by_column"].items() if v > 0}
        if not stuck:
            st.success("Aucune valeur bloquée détectée.")
        else:
            show_table(pd.DataFrame({
                "Variable": list(stuck), "Valeurs bloquées": list(stuck.values()),
            }))

    with st.expander("Comment est calculé le Data Quality Score ?"):
        st.markdown(
            "Le score part de 100 et retire des points proportionnels à la part "
            "de valeurs manquantes, de valeurs hors plage, de doublons, de valeurs "
            "bloquées et de mesures absentes à cause d'interruptions. En dessous de "
            f"{DQ_LIMIT}, les diagnostics sont signalés comme « fiabilité limitée »."
        )


# ------------------------------------------------------------
# Page 9 : Rapports
# ------------------------------------------------------------

REPORT_KINDS = [
    "Rapport quotidien",
    "Rapport hebdomadaire",
    "Rapport mensuel",
    "Rapport par site",
    "Rapport par équipement",
    "Rapport des anomalies",
    "Rapport des recommandations",
    "Rapport énergétique",
]


def report_events_table(events):
    """
    Tableau de rapport qui sépare clairement :
    observation mesurée / anomalie IA / règle HVAC / cause probable /
    action recommandée / validation du technicien.
    """

    if events.empty:
        return pd.DataFrame()

    validation = (
        events["status"].astype(str)
        + " - " + events["validation"].astype(str)
        + np.where(events["comment"].astype(str) != "",
                   " - " + events["comment"].astype(str), "")
    )

    return pd.DataFrame({
        "Date": events["start"].dt.strftime("%d/%m/%Y %H:%M"),
        "Site": events["site"],
        "Équipement": events["equipment_id"],
        "Observation mesurée": events["observation"],
        "Anomalie IA": (
            events["votes"].astype(str) + "/3 modèles (" + events["models_voted"]
            + "), indice d'accord " + events["agreement"].astype(str) + " %"
        ),
        "Règle HVAC": events["rules"],
        "Cause probable": events["probable_cause"],
        "Action recommandée": events["recommendation"],
        "Priorité": events["priority"],
        "Validation du technicien": validation,
    })


def page_reports(ctx):
    banner("Rapports", "Exports Excel et CSV (PDF prévu dans une version ultérieure)")
    render_data_warnings(ctx)

    view = ctx["view"]
    data = view["data"]
    eq = view["equipment"]
    events = view["events_all"][view["events_all"]["persistent"]]
    cfg = ctx["cfg"]

    if eq.empty:
        st.warning("Aucun équipement ne correspond aux filtres choisis.")
        return

    kind = st.selectbox("Type de rapport", REPORT_KINDS)

    scope_events = events
    scope_eq = eq
    scope_label = "Tous les équipements filtrés"
    period_label = (
        f"{format_datetime(data['timestamp'].min())} → "
        f"{format_datetime(data['timestamp'].max())}"
    )

    if kind in ("Rapport quotidien", "Rapport hebdomadaire", "Rapport mensuel"):
        days = {"Rapport quotidien": 1, "Rapport hebdomadaire": 7,
                "Rapport mensuel": 30}[kind]
        end = data["timestamp"].max()
        start = end - pd.Timedelta(days=days)
        scope_events = events[events["end"] >= start]
        period_label = f"{format_datetime(start)} → {format_datetime(end)}"
        energy_data = data[data["timestamp"] >= start]
        scope_eq = eq.copy()
        scope_eq["energy_kwh"] = scope_eq["equipment_id"].map(
            (energy_data["electrical_power"] * energy_data["dt_min"] / 60)
            .groupby(energy_data["equipment_id"]).sum()
        ).fillna(0)

    elif kind == "Rapport par site":
        site = st.selectbox("Site", sorted(eq["site"].unique()))
        scope_eq = eq[eq["site"] == site]
        scope_events = events[events["site"] == site]
        scope_label = f"Site : {site}"

    elif kind == "Rapport par équipement":
        equipment = st.selectbox("Équipement", list(eq["equipment_id"]))
        scope_eq = eq[eq["equipment_id"] == equipment]
        scope_events = events[events["equipment_id"] == equipment]
        scope_label = f"Équipement : {equipment}"

    # ---- Construction des tableaux ----
    summary = pd.DataFrame({
        "Indicateur": [
            "Type de rapport", "Périmètre", "Période", "Équipements",
            "Anomalies persistantes", "dont priorité critique ou élevée",
            "Consommation (kWh)", "Data Quality Score (/100)",
            "Données", "Avertissement",
        ],
        "Valeur": [str(v) for v in [
            kind, scope_label, period_label, len(scope_eq),
            len(scope_events),
            int(scope_events["priority"].isin(["Critique", "Élevée"]).sum()),
            round(float(scope_eq["energy_kwh"].sum()), 1),
            round(ctx["analysis"]["quality"]["summary"]["dq_score"], 0),
            "DÉMONSTRATION (données non réelles)" if ctx["is_demo"]
            else "Données importées",
            "Diagnostics probables, à valider par un technicien. L'indice "
            "d'accord n'est pas une probabilité de panne.",
        ]],
    })

    events_table = report_events_table(scope_events)

    recommendations = pd.DataFrame()
    if not scope_events.empty:
        recommendations = pd.DataFrame({
            "Équipement": scope_events["equipment_id"],
            "Problème détecté": scope_events["category"],
            "Preuve principale": scope_events["observation"],
            "Cause probable": scope_events["probable_cause"],
            "Action recommandée": scope_events["recommendation"],
            "Priorité": scope_events["priority"],
            "Indice d'accord (%)": scope_events["agreement"],
            "Énergie potentiellement évitable (kWh)":
                scope_events["avoidable_kwh"].round(1),
            "Limites du diagnostic": scope_events["limits"],
        })

    energy_table = pd.DataFrame({
        "Équipement": scope_eq["equipment_id"],
        "Consommation (kWh)": scope_eq["energy_kwh"].round(1),
        "Hors occupation (kWh)": scope_eq["energy_out_of_occupancy_kwh"].round(1),
        "Puissance maximale (kW)": scope_eq["max_power"].round(1),
        "Énergie potentiellement évitable (kWh)": scope_eq["avoidable_kwh"].round(1),
    })

    # ---- Affichage selon le type ----
    st.subheader("Résumé")
    show_table(summary)

    if kind == "Rapport énergétique":
        st.subheader("Énergie")
        show_table(energy_table)
        sheets = {"Résumé": summary, "Énergie": energy_table}
        csv_table = energy_table

    elif kind == "Rapport des recommandations":
        st.subheader("Recommandations")
        if recommendations.empty:
            st.info("Aucune recommandation sur ce périmètre.")
        else:
            show_table(recommendations)
        sheets = {"Résumé": summary, "Recommandations": recommendations}
        csv_table = recommendations

    else:
        st.subheader("Anomalies")
        if events_table.empty:
            st.info("Aucune anomalie persistante sur ce périmètre.")
        else:
            show_table(events_table)
        sheets = {"Résumé": summary, "Anomalies": events_table,
                  "Énergie": energy_table}
        if not recommendations.empty:
            sheets["Recommandations"] = recommendations
        csv_table = events_table

    # ---- Exports ----
    st.subheader("Exporter")

    column_1, column_2 = st.columns(2)
    column_1.download_button(
        "Télécharger le rapport en Excel",
        data=dataframes_to_excel(sheets),
        file_name="rapport_redge_compass.xlsx",
        mime=("application/vnd.openxmlformats-officedocument."
              "spreadsheetml.sheet"),
    )
    column_2.download_button(
        "Télécharger le tableau principal en CSV",
        data=csv_table.to_csv(index=False).encode("utf-8-sig"),
        file_name="rapport_redge_compass.csv",
        mime="text/csv",
    )

    st.subheader("Export complet de l'analyse")
    st.caption("Toutes les mesures avec les étiquettes des modèles, les règles et les scores.")

    full = view["data"].drop(columns=["candidate"], errors="ignore")
    column_1, column_2 = st.columns(2)
    column_1.download_button(
        "Analyse complète en Excel",
        data=dataframe_to_excel(full),
        file_name="resultats_redge_compass.xlsx",
        mime=("application/vnd.openxmlformats-officedocument."
              "spreadsheetml.sheet"),
    )
    column_2.download_button(
        "Analyse complète en CSV",
        data=full.to_csv(index=False).encode("utf-8-sig"),
        file_name="resultats_redge_compass.csv",
        mime="text/csv",
    )


# ------------------------------------------------------------
# Page 10 : Configuration
# ------------------------------------------------------------

def render_column_mapping(df_raw):
    """Associe les colonnes du fichier aux noms attendus par l'application."""

    st.subheader("Correspondance des colonnes du CSV")
    st.caption(
        "Si ton fichier utilise d'autres noms (par exemple ZoneTemp au lieu de "
        "zone_temperature), choisis ici la colonne du fichier qui correspond "
        "à chaque variable."
    )

    options = [""] + list(df_raw.columns)
    current = effective_mapping(list(df_raw.columns))
    saved = dict(st.session_state.get("column_mapping", {}))

    all_columns = [
        (name, name, True) for name in REQUIRED_COLUMNS
    ] + [
        (name, f"{label} (facultatif)", False)
        for name, label in {**OPTIONAL_COLUMNS, **HVAC_EXTRA_COLUMNS}.items()
    ]

    left, right = st.columns(2)

    for position, (name, label, required) in enumerate(all_columns):
        column = left if position % 2 == 0 else right
        choice = column.selectbox(
            f"{label}" + (" *" if required else ""),
            options,
            index=options.index(current[name]),
            format_func=lambda value: "(aucune colonne)" if value == "" else value,
            key=f"mapping_{name}_{hash(tuple(df_raw.columns))}",
        )
        saved[name] = choice

    st.session_state["column_mapping"] = saved


def page_config(ctx):
    banner("Configuration", "Réglages de l'analyse, des règles HVAC et de l'énergie")

    st.button("Remettre les valeurs par défaut", on_click=reset_config)

    st.subheader("Modèles d'intelligence artificielle")
    st.slider("Sensibilité des modèles", 0.01, 0.20, step=0.01, key="cfg_sensitivity")
    st.caption(
        "Part des mesures que chaque modèle considère comme inhabituelles. "
        "0,10 = environ 10 %. Plus la valeur est haute, plus on détecte "
        "d'anomalies (et de fausses alertes). À calibrer avec les données réelles."
    )

    st.number_input("Nombre de voisins du modèle KNN", 2, 50, step=1, key="cfg_neighbors")
    st.caption(
        "Le modèle KNN compare chaque mesure à ses voisins les plus proches. "
        "5 convient pour commencer."
    )

    st.selectbox(
        "Règle de vote", [1, 2, 3], key="cfg_vote_rule",
        format_func=lambda n: f"{n} modèle(s) sur 3",
    )
    st.caption(
        "Nombre minimal de modèles d'accord pour confirmer une anomalie IA. "
        "Par défaut : 2 sur 3."
    )

    st.subheader("Seuils des règles HVAC")
    st.number_input("Tolérance par rapport à la consigne (°C)", 0.5, 5.0,
                    step=0.5, key="cfg_temp_threshold")
    st.caption("Écart de température au-delà duquel on parle d'inconfort (règles 1 à 3).")

    st.number_input("Seuil de saturation de la vanne (%)", 50, 100,
                    step=5, key="cfg_valve_threshold")
    st.caption("Position de vanne à partir de laquelle on considère la vanne « très ouverte ».")

    st.number_input("Seuil de puissance (x niveau habituel)", 1.1, 5.0,
                    step=0.1, key="cfg_power_factor")
    st.caption(
        "La puissance est jugée élevée au-delà de ce multiple de la puissance "
        "habituelle de l'équipement (médiane en période d'occupation)."
    )

    st.number_input("Durée minimale d'une anomalie (minutes)", 5, 240,
                    step=5, key="cfg_min_duration")
    st.caption(
        "Une anomalie doit durer au moins ce temps pour être retenue (30 minutes "
        "par défaut, soit 6 mesures de 5 minutes). Les événements plus courts "
        "sont ignorés pour éviter les fausses alertes. Pour tester le petit "
        "fichier donnees_hvac.csv (anomalie de 20 minutes), descends à 15 minutes."
    )

    st.markdown("**Activation des règles HVAC**")
    st.checkbox("Règle 1 : inconfort thermique", key="cfg_rule1")
    st.checkbox("Règle 2 : refroidissement potentiellement insuffisant", key="cfg_rule2")
    st.checkbox("Règle 3 : surconsommation potentielle", key="cfg_rule3")
    st.checkbox("Règle 4 : fonctionnement hors occupation", key="cfg_rule4")

    columns = ctx["mapped_columns"]
    st.markdown("**Règles 5 et 6 (prévues en phase 2)**")
    for name, label in HVAC_EXTRA_COLUMNS.items():
        state = "présente dans le fichier" if name in columns else "absente"
        st.write(f"- {label} : {state}")
    st.caption(
        "Les règles 5 (filtre encrassé) et 6 (production frigorifique) "
        "nécessitent ces variables. Elles sont désactivées dans cette version."
    )

    st.subheader("Énergie")
    st.number_input("Prix du kWh (0 = non défini)", 0.0, 10.0, step=0.01,
                    format="%.3f", key="cfg_price_kwh")
    st.caption("Aucun prix n'est supposé : tant que la valeur est 0, aucun coût n'est affiché.")
    st.text_input("Unité monétaire", key="cfg_currency")
    st.number_input("Facteur d'émission de CO2 (kg par kWh, 0 = non défini)", 0.0, 5.0,
                    step=0.01, format="%.3f", key="cfg_co2_factor")
    st.caption("À renseigner selon le pays et le fournisseur d'énergie du client.")

    st.subheader("Prévu en phase 2")
    st.write(
        "- Horaires d'occupation (aujourd'hui, la colonne `occupancy` du fichier est utilisée)\n"
        "- Choix des unités\n"
        "- Période d'analyse pour l'entraînement des modèles "
        "(le filtre de période de la barre latérale ne limite que l'affichage)"
    )

    render_column_mapping(ctx["df_raw"])


# ============================================================
# 10. PROGRAMME PRINCIPAL
# ============================================================

def make_view(analysis, filters, feedback):
    """
    Applique les filtres de la barre latérale et les retours des techniciens,
    puis calcule les tableaux (équipements, sites) pour les pages.
    """

    data = analysis["data"]
    events = analysis["events"].copy()

    # ---- Retours des techniciens ----
    events = events.merge(feedback, on="event_id", how="left")
    for column in FEEDBACK_COLUMNS[1:]:
        events[column] = events[column].fillna("")
    events["status"] = events["status"].replace("", "Nouvelle")
    events["validation"] = events["validation"].replace("", "Non évalué")

    # ---- Filtres ----
    mask = pd.Series(True, index=data.index)
    event_mask = pd.Series(True, index=events.index)

    for column, selected in [
        ("client", filters["clients"]),
        ("site", filters["sites"]),
        ("equipment_type", filters["types"]),
    ]:
        if selected:
            mask &= data[column].isin(selected)
            event_mask &= events[column].isin(selected)

    start, end = filters["period"]
    mask &= (data["timestamp"].dt.date >= start) & (data["timestamp"].dt.date <= end)
    event_mask &= (events["start"].dt.date >= start) & (events["start"].dt.date <= end)

    data_f = data[mask].copy()
    events_f = events[event_mask].copy()

    persistent = events_f[events_f["persistent"]]
    open_events = persistent[~persistent["status"].isin(CLOSED_STATUSES)]
    # Événements trop courts : on ne les garde "à surveiller" que s'ils
    # déclenchent une règle HVAC (les détections IA isolées sont ignorées,
    # sinon chaque équipement serait toujours "à surveiller").
    short_open = events_f[
        ~events_f["persistent"]
        & ~events_f["status"].isin(CLOSED_STATUSES)
        & (events_f["category"] != "Détection IA seule")
    ]

    data_f["in_open_event"] = data_f["event_id"].isin(set(open_events["event_id"]))

    quality = analysis["quality"]["per_equipment"]
    reference_time = data["timestamp"].max()

    if data_f.empty:
        equipment = pd.DataFrame()
    else:
        equipment = build_equipment_table(
            data_f, open_events, short_open, quality, reference_time, {
                "temp_threshold": filters["temp_threshold"],
            },
        )

    return {
        "data": data_f,
        "events_all": events_f,
        "events_open": open_events,
        "equipment": equipment,
    }


def main():
    inject_css()
    init_config()
    cfg = get_cfg()

    # ---------------- Barre latérale : navigation ----------------
    st.sidebar.markdown(
        f"<h3 style='margin-bottom:0'>Redge Compass AI</h3>"
        f"<span style='color:{COLORS['grey']};font-size:0.8rem'>"
        "Analyse HVAC - lecture seule</span>",
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio("Navigation", PAGES, key="page",
                            label_visibility="collapsed")

    # ---------------- Barre latérale : source de données ----------------
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Source des données**")

    source_options = []
    if os.path.exists(os.path.join(BASE_DIR, FILE_DEMO_EXTENDED)):
        source_options.append("Démo étendue (données synthétiques)")
    if os.path.exists(os.path.join(BASE_DIR, FILE_DEMO_SMALL)):
        source_options.append("Démo : donnees_hvac.csv")
    source_options.append("Données réelles : déposer un CSV")

    source = st.sidebar.radio("Source", source_options, key="source",
                              label_visibility="collapsed")

    df_raw = None
    is_demo = True
    source_label = source

    if source.startswith("Démo étendue"):
        df_raw = read_local_file(FILE_DEMO_EXTENDED)
        source_label = "données SYNTHÉTIQUES générées par generer_donnees_demo.py"

    elif source.startswith("Démo :"):
        df_raw = read_local_file(FILE_DEMO_SMALL)
        source_label = "fichier donnees_hvac.csv"

    else:
        is_demo = False
        uploaded = st.sidebar.file_uploader("Fichier CSV", type=["csv"])
        if uploaded is not None:
            try:
                df_raw = load_csv_bytes(uploaded.getvalue())
                source_label = uploaded.name
            except Exception as error:
                st.error(f"Impossible de lire le fichier déposé : {error}")

    if df_raw is None:
        banner("Redge Compass AI", "Aucune donnée chargée")
        st.info(
            "Dépose un fichier CSV dans la barre latérale, ou choisis un "
            "mode démonstration."
        )
        st.stop()

    # ---------------- Correspondance des colonnes ----------------
    mapped = apply_mapping(df_raw)
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in mapped.columns]

    if missing_columns:
        banner("Colonnes manquantes",
               "Associe les colonnes de ton fichier aux variables attendues")
        st.error(
            "Colonnes obligatoires absentes : " + ", ".join(missing_columns)
        )
        st.write("Colonnes présentes dans le fichier :")
        st.code(", ".join(map(str, df_raw.columns)))
        render_column_mapping(df_raw)
        st.stop()

    if len(mapped) < 10:
        banner("Données insuffisantes", "")
        st.error(
            "Le fichier contient moins de 10 lignes : il n'y a pas assez de "
            "données pour exécuter les modèles."
        )
        st.stop()

    # ---------------- Analyse (mise en cache) ----------------
    analysis_cfg = {key: cfg[key] for key in ANALYSIS_KEYS}

    try:
        analysis = run_analysis(mapped, analysis_cfg)
    except Exception as error:
        banner("Erreur d'analyse", "")
        st.error(f"L'analyse a échoué : {error}")
        st.stop()

    data = analysis["data"]

    # ---------------- Barre latérale : filtres ----------------
    with st.sidebar.expander("Filtres"):
        clients = st.multiselect("Client (vide = tous)",
                                 sorted(data["client"].unique()))
        sites = st.multiselect("Site (vide = tous)",
                               sorted(data["site"].unique()))
        types = st.multiselect("Type d'équipement (vide = tous)",
                               sorted(data["equipment_type"].unique()))

        first_day = data["timestamp"].min().date()
        last_day = data["timestamp"].max().date()
        period = st.date_input("Période", value=(first_day, last_day),
                               min_value=first_day, max_value=last_day)

    if isinstance(period, (tuple, list)):
        if len(period) == 2:
            period_start, period_end = period
        elif len(period) == 1:
            period_start = period_end = period[0]
        else:
            period_start, period_end = first_day, last_day
    else:
        period_start = period_end = period

    filters = {
        "clients": clients,
        "sites": sites,
        "types": types,
        "period": (period_start, period_end),
        "temp_threshold": cfg["temp_threshold"],
    }

    view = make_view(analysis, filters, load_feedback())

    ctx = {
        "cfg": cfg,
        "analysis": analysis,
        "view": view,
        "is_demo": is_demo,
        "source_label": source_label,
        "df_raw": df_raw,
        "mapped_columns": list(mapped.columns),
    }

    st.sidebar.caption(
        "Résultats indicatifs : chaque diagnostic doit être validé par un "
        "technicien. L'indice d'accord entre les modèles n'est pas une "
        "probabilité de panne."
    )

    # ---------------- Affichage de la page choisie ----------------
    pages = {
        "Vue d'ensemble": page_overview,
        "Sites": page_sites,
        "Équipements": page_equipment,
        "Anomalies": page_anomalies,
        "Recommandations": page_recommendations,
        "Comparaison des modèles": page_models,
        "Analyse énergétique": page_energy,
        "Qualité des données": page_quality,
        "Rapports": page_reports,
        "Configuration": page_config,
    }

    pages[page](ctx)


main()