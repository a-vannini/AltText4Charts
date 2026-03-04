import os
import re
import base64
import sqlite3
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util

from openai import OpenAI

import unicodedata



def load_api_key(path="../data/api_key.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_img_csv_path(chart_id, conn):
    img_path = os.path.join('..', 'data', 'NZZ_original', f'{chart_id}', f'{chart_id}.png')
    csv_path = os.path.join('..','data', 'data_nzz_csv', f'{chart_id}.csv')
    # print(img_path, csv_path)
    return img_path, csv_path


def get_prompt_data_from_chart(chart_id, conn):
    chart_query = """
    SELECT c.complex,
           c.title, c.subtitle, c.notes,
           c.x_axis_min, c.x_axis_max,
           c.y_axis_min, c.y_axis_max,
           ct.type AS chart_type
    FROM chart c
    JOIN chart_type ct ON c.chart_type_id = ct.id
    WHERE c.id = ?
    """
    data_query = "SELECT DISTINCT x_category, y_category FROM chart_data WHERE chart_id = ?"
    prognosis_query = "SELECT EXISTS (SELECT 1 FROM chart_data WHERE chart_id = ? AND prognosis = 1) AS has_prognosis;"
    highlighted_query = "SELECT EXISTS (SELECT 1 FROM chart_data WHERE chart_id = ? AND highlighted = 1) AS is_highlighted;"
    event_query = "SELECT type, date, label FROM data_event WHERE chart_id = ?"

    chart_df = pd.read_sql_query(chart_query, conn, params=(chart_id,))
    if chart_df.empty:
        raise ValueError(f"No chart data found for chart_id={chart_id}")

    data_df = pd.read_sql_query(data_query, conn, params=(chart_id,))
    prognosis_df = pd.read_sql_query(prognosis_query, conn, params=(chart_id,))
    highlighted_df = pd.read_sql_query(highlighted_query, conn, params=(chart_id,))
    events_df = pd.read_sql_query(event_query, conn, params=(chart_id,))

    def safe_val(val):
        return str(val) if pd.notna(val) else None

    row = chart_df.iloc[0]

    return {
        "complex": safe_val(row["complex"]),
        "title": safe_val(row["title"]),
        "subtitle": safe_val(row["subtitle"]),
        "notes": safe_val(row["notes"]),
        "chart_type": safe_val(row["chart_type"]),
        "x_category": safe_val(data_df.iloc[0]["x_category"]),
        "y_category": safe_val(data_df.iloc[0]["y_category"]),
        "x_axis_min": safe_val(row["x_axis_min"]),
        "x_axis_max": safe_val(row["x_axis_max"]),
        "y_axis_min": safe_val(row["y_axis_min"]),
        "y_axis_max": safe_val(row["y_axis_max"]),
        "has_prognosis": bool(prognosis_df.iloc[0]["has_prognosis"]),
        "is_highlighted": bool(highlighted_df.iloc[0]["is_highlighted"]),
        "data_events": events_df.to_dict(orient="records") if not events_df.empty else np.nan
    }


def encode_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def get_multimodal_model_output(
        prompt_text, 
        image_path, 
        api_key_path, 
        model,
        temperature):
    
    image_base64 = encode_image_base64(image_path)
    image_data_url = f"data:image/png;base64,{image_base64}"

    api_key = load_api_key(api_key_path)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    completion = client.chat.completions.create(
        # extra_headers={
        #     "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
        #     "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
        # },
        extra_body={},
        model=model,
        temperature=temperature,
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "image_url",
                "image_url": {"url": image_data_url}
                },
                {
                "type": "text",
                "text": prompt_text
                }
            ]
            }
        ]
    )
    response = completion.choices[0].message.content

    return response


def get_language_model_output(
        prompt_text, 
        api_key_path, 
        model
    ):

    api_key = load_api_key(api_key_path)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    completion = client.chat.completions.create(
        extra_body={},
        model=model,
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt_text
                }
            ]
            }
        ]
    )
    response = completion.choices[0].message.content

    return response


def load_prompt_text_function(prompt_text_function_name: str):
    import sqlite3

    conn = sqlite3.connect("../chart_database.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, content FROM prompt_text_function WHERE name = ?
    """, (prompt_text_function_name,))
    
    result = cursor.fetchone()
    conn.close()

    if result is None:
        raise ValueError(f"Funktion '{prompt_text_function_name}' nicht gefunden.")

    prompt_text_id, function_code = result
    return prompt_text_id, function_code


def compile_function_from_string(function_code: str, function_name: str):
    import re, os, pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    # Standard-Umgebung vorbereiten
    namespace = {
        "re": re,
        "os": os,
        "pd": pd,
        "plt": plt,
        "mpimg": mpimg
    }

    # Funktion kompilieren
    exec(function_code, namespace)

    # Funktion zurückgeben
    if function_name not in namespace:
        raise ValueError(f"Funktion '{function_name}' wurde im Code nicht definiert.")
    
    return namespace[function_name]


# Funktionen: Bereinigt alt_text und splitet sie in short und long auf

# def clean_text(text):
#     return re.sub(r"[^a-zA-ZäöüÄÖÜß0-9 .,;:!?()\"'\-\n]", "", text)

def clean_text(text):
    if text is None:
        return None

    text = text.strip()
    text = re.sub(r'^\*+\s*', '', text) # ** am Anfang entfernen
    text = re.sub(r'\s*\*+$', '', text) # ** am Ende entfernen
    return text.strip()


def extract_descriptions(text):
    """
    Extrahiert Kurzbeschreibung, Überblick und Lange Beschreibung
    aus einem Modell-Output.
    """
    pattern = re.compile(
        r"(?:\*\*)?\s*Kurzbeschreibung\s*:?\s*(.*?)\s*(?:\*\*)?\s*Überblick\s*:?\s*(.*?)\s*(?:\*\*)?\s*Lange\s*Beschreibung\s*:?\s*(.*)",

        re.IGNORECASE | re.DOTALL
    )
    
    match = pattern.search(text)
    
    if match:
        short_desc = clean_text(match.group(1).strip())
        overview = clean_text(match.group(2).strip())
        long_desc = clean_text(match.group(3).strip())
        return short_desc, overview, long_desc
    else:
        raise ValueError("Konnte 'Kurzbeschreibung', 'Überblick' und 'Lange Beschreibung' nicht zuverlässig extrahieren.")

    
def fill_generation_run_table(model_id, prompt_text_function_id, temperature, n_variants, conn=None):
    close_conn = False
    if conn is None:
        conn = sqlite3.connect("../chart_database.db")
        close_conn = True

    cursor = conn.cursor()

    # ---- 1) Letzte generation_run.id holen ----
    cursor.execute("SELECT MAX(id) FROM generation_run")
    row = cursor.fetchone()
    last_id = row[0] if row[0] is not None else 0

    new_id = last_id + 1   # neue ID

    created_at = datetime.now().isoformat()

    # ---- 2) Einfügen mit manueller ID ----
    cursor.execute("""
        INSERT INTO generation_run
            (id, model_id, prompt_text_function_id, created_at, temperature, n_variants)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (new_id, model_id, prompt_text_function_id, created_at, temperature, n_variants))

    conn.commit()

    if close_conn:
        conn.close()

    return new_id



def fill_alt_text_embedding_table(
        alt_text_id,
        model_id,
        embedding,         
        normalized=False,
        conn=None):

    close_conn = False
    if conn is None:
        conn = sqlite3.connect("../chart_database.db")
        close_conn = True

    cursor = conn.cursor()

    def to_blob(vec):
        if vec is None:
            return None
        if isinstance(vec, np.ndarray):
            return vec.tobytes()
        return bytes(vec)

    dim = None
    if isinstance(embedding, np.ndarray):
        dim = embedding.shape[0]

    cursor.execute("""
        INSERT INTO alt_text_embedding (
            alt_text_id, model_id, dim, normalized, embedding
        ) VALUES (?, ?, ?, ?, ?)
    """, (
        alt_text_id,
        model_id,
        dim,
        int(normalized),
        to_blob(embedding)
    ))

    alt_text_embedding_id = cursor.lastrowid
    conn.commit()
    if close_conn:
        conn.close()
    return alt_text_embedding_id




def fill_alt_text_table(
        chart_id,
        generation_run_id, 
        variant_no,
        short_description_metadata, 
        short_description_overview, 
        long_description, 
        conn=None):
    
    close_conn = False
    if conn is None:
        conn = sqlite3.connect("../chart_database.db")
        close_conn = True

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO alt_text (
            chart_id,
            generation_run_id,
            variant_no,
            short_description_metadata,
            short_description_overview,
            long_description
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        chart_id,
        generation_run_id,
        variant_no,
        short_description_metadata,
        short_description_overview,
        long_description
    ))

    alt_text_id = cursor.lastrowid
    conn.commit()

    if close_conn:
        conn.close()
    
    return alt_text_id

    
def run_chart_alt_text_generation_pipeline(
        chart_ids_lst, 
        prompt_alt_text_function_name,
        model_alt_text,
        model_embedding,
        api_key_path, 
        conn,
        verbose,
        temperature,
        n_variants,               # SO viele Varianten werden erzeugt & gespeichert
        compute_embeddings,     # Embeddings optional deaktivierbar
        normalize_embeddings    # L2-Normalisierung für Speicherung
    ):
    
    cursor = conn.cursor()
    variation_number=1  # Startnummer für Varianten

    # -- Helper: Model-ID aus Tabelle "model" holen (model_series == name)
    def get_model_id(model_name):
        cursor.execute("SELECT id FROM model WHERE model_series = ?", (model_name,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Model '{model_name}' not found in table 'model' (matching on model_series).")
        return row[0]

    # IDs für Text- und Embedding-Modelle
    model_alt_text_id = get_model_id(model_alt_text)
    embedding_model_id = get_model_id(model_embedding) if compute_embeddings else None

    # Prompt-Funktion laden & kompilieren
    prompt_alt_text_id, code_alt_text_str = load_prompt_text_function(prompt_alt_text_function_name)
    prompt_alt_text_function = compile_function_from_string(code_alt_text_str, prompt_alt_text_function_name)

    # -------- DB: EIN generation_run anlegen, der die Gesamtzahl der Varianten protokolliert --------
    generation_run_id = fill_generation_run_table(
        model_id=model_alt_text_id,
        prompt_text_function_id=prompt_alt_text_id,
        temperature=temperature,
        n_variants=n_variants,
        conn=conn
    )

    # Ergebnisse sammeln
    variant_results = []  # Liste mit Dicts je Variante und je chart_id

    # -------- n Varianten pro chart_id generieren & speichern --------

    for chart_id in chart_ids_lst:
        # Bild-/CSV-Pfade und Chart-Kontext laden
        img_path, csv_path = get_img_csv_path(chart_id, conn)
        chart_info = get_prompt_data_from_chart(chart_id, conn)

        # Prompt erzeugen (konstanter Prompt; die Variation kommt aus Stochastik/Temperature/Model)
        prompt_alt_text = prompt_alt_text_function(chart_info, csv_path, img_path)
       
        for i in range(n_variants):
            variant_no = variation_number + i

            # Multimodale Anfrage (Bild + Prompt)
            alt_text = get_multimodal_model_output(
                prompt_text=prompt_alt_text, 
                image_path=img_path, 
                api_key_path=api_key_path, 
                model=model_alt_text,
                temperature=temperature
            )
            # if verbose:
            #     print(f"\n=== Modell-Output für Variante {variant_no} ===")
            #     print(alt_text)
            #     print("=============================================")
            # Kurzbeschreibung, Überblick und Lange Beschreibung extrahieren
            short_desc_metadata, short_desc_overview, long_desc = extract_descriptions(alt_text)

            # if verbose:
            #     print(f"\n— Variante {variant_no} —")
            #     print(f"**Kurzbeschreibung**:\n{short_desc_metadata}")
            #     print(f"\n**Übersicht**:\n{short_desc_overview}")
            #     print(f"\n**Lange Beschreibung**:\n{long_desc}")

            # 1) alt_text zunächst OHNE Embedding-Referenz anlegen (FK-Zyklus vermeiden)
            alt_text_id = fill_alt_text_table(
                chart_id=chart_id,
                generation_run_id=generation_run_id,
                variant_no=variant_no,                    
                short_description_metadata=short_desc_metadata, 
                short_description_overview=short_desc_overview,
                long_description=long_desc,
                conn=conn
            )


            # 2) Optional: Embeddings berechnen und in alt_text_embedding speichern
            alt_text_embedding_id = None
            if compute_embeddings:
            
                st_model = SentenceTransformer(model_embedding)

                def enc(text):
                    if text is None or str(text).strip() == "":
                        return None
                    vec = st_model.encode(text, normalize_embeddings=normalize_embeddings)
                    return np.asarray(vec, dtype=np.float32)
                
                alt_text_total = f"{short_desc_metadata} {short_desc_overview} {long_desc}"

                alt_text_emb = enc(alt_text_total)

                alt_text_embedding_id = fill_alt_text_embedding_table(
                    alt_text_id=alt_text_id,
                    model_id=embedding_model_id,  # darf None sein, falls Modell nicht in DB
                    embedding=alt_text_emb,
                    normalized=bool(normalize_embeddings),
                    conn=conn
                )


            # Variante protokollieren
            variant_results.append({
                "variant_no": variant_no,
                "alt_text_id": alt_text_id,
                "alt_text_embedding_id": alt_text_embedding_id
            })

    # if verbose:
    #     print("\nPipeline erfolgreich abgeschlossen.")

    return {
        "generation_run_id": generation_run_id,
        "variants": variant_results
    }
