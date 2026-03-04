import os
import json
import ast
import re
import sys
import uuid
from typing import List
import shutil
import sqlite3

import pandas as pd
import numpy as np

# ----------------------------------------------------------------

def detect_axis_type(series):
    """Erkennt, ob eine Achse numerisch oder kategorisch ist."""
    try:
        pd.to_numeric(series.dropna().astype(str).str.replace(',', '.'), errors='raise')
        return "numeric"
    except Exception:
        return "categorical"
    

def sums_to_100_every_row(two_col_like) -> bool:
    """
    True, falls die *zwei Werte-Spalten* (letzte zwei Spalten im DF)
    zeilenweise auf 100 summieren (mit Toleranz).
    Ändert das Eingabe-DataFrame NICHT.
    """
    # Kopie erstellen, damit wir nichts mutieren
    df = two_col_like.copy() if isinstance(two_col_like, pd.DataFrame) else pd.DataFrame(two_col_like)

    # Wir brauchen mindestens 2 Spalten
    if df.shape[1] < 2:
        return False

    # Nehme die letzten zwei Spalten als Werte-Spalten
    vals = df.iloc[:, -2:]

    # Häufige Formate bereinigen: "45%", "45,6", Whitespaces, Dezimal-Komma
    cleaned = (
        vals.astype(str)
            .replace({r'%': '', r'\s+': '', ',': '.'}, regex=True)
    )

    numeric = cleaned.apply(pd.to_numeric, errors='coerce')

    # Wenn etwas nicht parsebar ist -> konservativ False
    if numeric.isna().any().any():
        return False

    # Toleranz für Rundungsfehler
    return np.isclose(numeric.sum(axis=1), 100.0, rtol=-1e-6, atol=1e-6).all()



def extract_data_from_json(json_path, folder_name, data_nzz_csv_path):
    """Extrahiert Daten aus einer JSON-Datei, speichert sie als CSV und bestimmt Achsentypen."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Fehler beim Lesen von {json_path}: {e}")
        return []

    data_entries = data.get("data", [])
    if not data_entries or len(data_entries) < 2:
        print(f"Warnung: Keine oder unzureichende Daten in {json_path}")
        return []

    column_names = data_entries[0]
    if not column_names or len(column_names) < 2:
        print(f"Warnung: Ungültige Spaltenüberschriften in {json_path}")
        return []

    column_names[0] = column_names[0] or "X"
    x_category = column_names[0]
    y_titles = column_names[1:]

    df_data = data_entries[1:]
    df = pd.DataFrame(df_data, columns=[x_category] + y_titles)

    # Chart-Type
    chart_type = data.get("options", {}).get("chartType", "")

    # Achsentyp-Erkennung
    x_axis_type = detect_axis_type(df[x_category])
    y_axis_types = {detect_axis_type(df[y]) for y in y_titles}
    y_axis_type = y_axis_types.pop() if len(y_axis_types) == 1 else "mixed"

    # Min/Max-Werte bestimmen
    if x_axis_type == "numeric":
        x_numeric = pd.to_numeric(df[x_category].astype(str).str.replace(',', '.'), errors='coerce')
        x_axis_min, x_axis_max = x_numeric.min(), x_numeric.max()
    elif chart_type.lower() == "line":
        x_values = df[x_category].dropna().values
        x_axis_min = x_values[0] if len(x_values) > 0 else np.nan
        x_axis_max = x_values[-1] if len(x_values) > 1 else np.nan
    else:
        x_axis_min = x_axis_max = np.nan

    if y_axis_type == "numeric":
        y_values = pd.concat([
            pd.to_numeric(df[y].astype(str).str.replace(',', '.'), errors='coerce')
            for y in y_titles
        ])
        y_axis_min, y_axis_max = y_values.min(), y_values.max()
    else:
        y_axis_min = y_axis_max = np.nan

    # For simple and complex definition, need to know if 2-column and sums to 100

    num_rows, num_columns = df.shape
    is_simple = ((num_columns == 3) and sums_to_100_every_row(df)) or (num_columns == 2)
    is_complex = ((num_columns > 3) and (num_rows > 1)) or ((num_columns == 3) and (not is_simple))

    # Metadaten
    chart_id = data.get("_id", str(uuid.uuid4()))
    title = data.get("title", np.nan)
    subtitle = data.get("subtitle", "")
    notes = data.get("notes", "")
    highlighted_col = data.get("options", {}).get("highlightDataSeries", [])
    highlighted_row = data.get("options", {}).get("highlightDataRows", [])
    prognosis = data.get("options", {}).get("dateSeriesOptions", {}).get("prognosisStart", "")
    events = data.get("events", "")

    # Speichern als CSV
    csv_path = os.path.join(data_nzz_csv_path, f"{folder_name}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # Zusammenfassung
    return [[
        chart_id, title, subtitle, notes, chart_type, x_category,
        df.shape[1], df.shape[0],
        x_axis_type, y_axis_type,
        x_axis_min, x_axis_max,
        y_axis_min, y_axis_max,
        highlighted_col, highlighted_row, prognosis, events,
        is_complex
    ]]

def generate_metadata_and_csv_per_plot(data_nzz_path, data_nzz_csv_path, meta_data_path):
    all_data = []

    for folder in os.listdir(data_nzz_path):
        folder_path = os.path.join(data_nzz_path, folder)
        json_file_path = os.path.join(folder_path, f"{folder}.json")

        if os.path.isdir(folder_path) and os.path.exists(json_file_path):
            extracted_info = extract_data_from_json(json_file_path, folder, data_nzz_csv_path)
            if extracted_info:
                all_data.extend(extracted_info)

    df_main = pd.DataFrame(
        all_data,
        columns=[
            "id", "title", "subtitle", "notes", "chart_type", "x_category",
            "num_columns", "num_rows", "x_axis_type", "y_axis_type",
            "x_axis_min", "x_axis_max", "y_axis_min", "y_axis_max",
            "highlighted_col", "highlighted_row", "prognosis", "events",
            "complex"
        ]
    )

    os.makedirs(os.path.dirname(meta_data_path), exist_ok=True)
    df_main.to_csv(meta_data_path, index=False, encoding="utf-8")

    print(f"✅ Einzel-CSV-Dateien gespeichert in: '{data_nzz_csv_path}'")
    print(f"✅ Übersichtstabelle gespeichert als: '{meta_data_path}'")

# ----------------------------------------------------------------

def flatten_csv(df_raw: pd.DataFrame, chart_id: str) -> pd.DataFrame:
    """
    Konvertiert ein DataFrame mit Zeitreihen-ähnlicher Struktur in ein flaches Format.
    Fügt Zeilen- und Spaltennummern hinzu. 'x_category' wird aus der Zelle oben links entnommen.
    """
    if df_raw.empty or df_raw.shape[1] < 2:
        return pd.DataFrame()

    # Extrahiere Spalten
    x_column = df_raw.columns[0]
    y_columns = df_raw.columns[1:]

    # Transformiere in Long-Format
    melted = df_raw.melt(id_vars=[x_column], var_name="y_category", value_name="y_value")
    melted.rename(columns={x_column: "x_value"}, inplace=True)

    # Metadaten-Spalten
    melted["chart_id"] = chart_id
    # Reihen- und Spaltennummern
    melted["row_index"] = melted.groupby("y_category").cumcount()
    col_number_map = {col: i for i, col in enumerate(y_columns)}
    melted["col_index"] = melted["y_category"].map(col_number_map)

    return melted[["chart_id", "x_value", "y_value", "y_category", "row_index", "col_index"]]


def format_csv_for_db(csv_path: str) -> pd.DataFrame:
    """Liest alle CSV-Dateien im angegebenen Ordner ein und gibt ein zusammengeführtes flaches DataFrame zurück.
    Output-Beispiel:
    chart_id	x_value	y_value	y_category	row_index	col_index
    0	0023e8fed9d111fed753bb3f6b0afe78	2012	5.011559	US-Exporte	0	0
    """
    all_data: List[pd.DataFrame] = []

    for file_name in os.listdir(csv_path):
        if not file_name.lower().endswith(".csv"):
            continue

        file_path = os.path.join(csv_path, file_name)
        try:
            df_raw = pd.read_csv(file_path)
        except Exception as e:
            print(f"Fehler beim Lesen der Datei {file_name}: {e}")
            continue

        chart_id = os.path.splitext(file_name)[0]
        df_flat = flatten_csv(df_raw, chart_id)

        if not df_flat.empty:
            all_data.append(df_flat)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# ----------------------------------------------------------------

def get_highlighted_col(df):

    for col in ["highlighted_col", "highlighted_row"]:
        df.loc[:, col] = df[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
        )

    df.loc[:, "highlighted"] = df.apply(
        lambda row: row["col_index"] in row["highlighted_col"],
        axis=1
    )

    df.loc[:, "highlighted"] = df.apply(
        lambda row: row["row_index"] in row["highlighted_row"],
        axis=1
    )

    return df


def mark_prognosis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Setzt die 'prognosis'-Spalte auf:
    - True, wenn row_index >= prognosis (und prognosis nicht NaN)
    - False, wenn prognosis NaN oder row_index < prognosis
    """
    df["prognosis"] = df.apply(
        lambda row: row["row_index"] >= row["prognosis"] if pd.notna(row["prognosis"]) else False,
        axis=1
    )
    return df

# ----------------------------------------------------------------
