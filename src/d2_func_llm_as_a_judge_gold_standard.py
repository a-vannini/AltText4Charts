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
import pickle
import time
from openai import APIStatusError, RateLimitError, InternalServerError


nltk.download('punkt', quiet=True)


# ------------------------------------
# LLM as a Judge Functions
# ------------------------------------
def load_prompt_text_function(prompt_text_function_name: str, conn):
    """
    output: prompt_text_id, function_code
    """
    import sqlite3

    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, content FROM prompt_text_function WHERE name = ?
    """, (prompt_text_function_name,))
    
    result = cursor.fetchone()

    if result is None:
        raise ValueError(f"Funktion '{prompt_text_function_name}' nicht gefunden.")

    prompt_text_id, function_code = result
    # print(prompt_text_id, function_code)

    cursor.close()

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


def get_img_csv_path(chart_id):
    """
    output: img_path, csv_path
    """
    img_path = os.path.join('..', 'data', 'NZZ_original', f'{chart_id}', f'{chart_id}.png')
    csv_path = os.path.join('..','data', 'data_nzz_csv', f'{chart_id}.csv')
    # print(img_path, csv_path)
    return img_path, csv_path


def encode_image_base64(image_path):
    """
    output: encoded image
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
    

def load_api_key(path="../data/api_key.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()
    

def construct_llm_as_a_judge_prompt(gold_standard_id, judge_fn, conn, verbose=True):
    """
    output: judge_prompt_text, chart_id
    """
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT chart_id,
                   short_description_metadata,
                   short_description_overview,
                   long_description
            FROM gold_standard_alt_text
            WHERE id = ?
        """, (gold_standard_id,))
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Gold-Standard Alt-Text mit id={gold_standard_id} nicht gefunden.")

        chart_id, short_meta, short_overview, long_desc = row

        combined_text = (
            f"Kurzbeschreibung (Metadaten): {short_meta}\n\n"
            f"Kurzbeschreibung (Überblick): {short_overview}\n\n"
            f"Lange Beschreibung:\n{long_desc}"
        )

        csv_path = os.path.join("..", "data", "data_nzz_csv", f"{chart_id}.csv")
        df = pd.read_csv(csv_path)

    finally:
        cursor.close()

    chart_data = df.to_string(index=False)
    judge_prompt_text = judge_fn(alt_text=combined_text, csv_text=chart_data)
    return judge_prompt_text, chart_id



def get_multimodal_model_output(
        prompt_text, 
        chart_id, 
        api_key_path, 
        model,
        timeout_seconds=60,
        max_retries=5,
        base_delay=1.0,
        max_delay=10.0):
    """
    Erste Version (Minimal-Request) + robuste Fehlerbehandlung.
    Keine extra_headers.
    """

    # frühe Validierung
    if isinstance(model, str) and model.startswith("http"):
        raise ValueError(f"Invalid model ID (looks like URL): {model}")

    # Bild laden und base64 encodieren
    img_path = os.path.join('..', 'data', 'NZZ_original', f'{chart_id}', f'{chart_id}.png')
    image_base64 = encode_image_base64(img_path)
    image_data_url = f"data:image/png;base64,{image_base64}"

    # API-Key & Client
    api_key = load_api_key(api_key_path)
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Retry-Schleife
    attempt = 0
    while True:
        try:
            completion = client.chat.completions.create(
                extra_body={},   # bleibt aus der ersten Version erhalten
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ],
                timeout=timeout_seconds,
            )

            return completion.choices[0].message.content

        except RateLimitError:
            # 429 -> exponential backoff
            delay = min(max_delay, base_delay * (2 ** attempt)) + (0.2 * (attempt + 1))
            time.sleep(delay)
            attempt += 1
            if attempt > max_retries:
                raise

        except (InternalServerError, APIStatusError) as e:
            status = getattr(e, "status_code", None)
            if status and status >= 500:
                # Serverfehler -> retry
                delay = min(max_delay, base_delay * (2 ** attempt)) + (0.2 * (attempt + 1))
                time.sleep(delay)
                attempt += 1
                if attempt > max_retries:
                    raise
            else:
                # andere Fehler sofort weitergeben
                raise




def extract_judge_response(output_text: str):
    """
    Erwartetes Format:
    Reason: <1-2 Sätze>
    Score: <1-5>
    """
    score_match = re.search(r"Score:\s*([1-5])\b", output_text, re.IGNORECASE)
    reason_match = re.search(r"Reason:\s*(.+)", output_text, re.IGNORECASE)

    score = int(score_match.group(1)) if score_match else None
    reason = reason_match.group(1).strip() if reason_match else None

    return score, reason



#---------------------------------------------------
# Metrics Funktionen
#---------------------------------------------------

def character_size(alt_text):
    return len(alt_text)

#---------------------------------------------------
# Fill golden_standar_alt_table
#---------------------------------------------------

def update_gold_standard_alt_text_row(
    gold_standard_id,
    tokens_short_description_metadata,
    tokens_short_description_overview,
    tokens_long_description,
    judge_model_id,
    func_clarity_id,
    func_completeness_id,
    func_conciseness_id,
    func_preceived_completeness_id,
    func_neutrality_id,
    func_correctness_id,
    score_clarity,
    reason_clarity,
    score_completeness,
    reason_completeness,
    score_conciseness,
    reason_conciseness,
    score_preceived_completeness,
    reason_preceived_completeness,
    score_neutrality,
    reason_neutrality,
    score_correctness,
    reason_correctness,
    conn
):
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE gold_standard_alt_text
        SET
            tokens_short_description_metadata = ?,
            tokens_short_description_overview = ?,
            tokens_long_description = ?,

            judge_model_id = ?,
            func_clarity_id = ?,
            func_completeness_id = ?,
            func_conciseness_id = ?,
            func_preceived_completeness_id = ?,
            func_neutrality_id = ?,
            func_correctness_id = ?,

            score_clarity = ?,
            reason_clarity = ?,
            score_completeness = ?,
            reason_completeness = ?,
            score_conciseness = ?,
            reason_conciseness = ?,
            score_preceived_completeness = ?,
            reason_preceived_completeness = ?,
            score_neutrality = ?,
            reason_neutrality = ?,
            score_correctness = ?,
            reason_correctness = ?
        WHERE id = ?
    """, (
        tokens_short_description_metadata,
        tokens_short_description_overview,
        tokens_long_description,

        judge_model_id,
        func_clarity_id,
        func_completeness_id,
        func_conciseness_id,
        func_preceived_completeness_id,
        func_neutrality_id,
        func_correctness_id,

        score_clarity,
        reason_clarity,
        score_completeness,
        reason_completeness,
        score_conciseness,
        reason_conciseness,
        score_preceived_completeness,
        reason_preceived_completeness,
        score_neutrality,
        reason_neutrality,
        score_correctness,
        reason_correctness,

        gold_standard_id
    ))
    conn.commit()
    cursor.close()


#---------------------------------------------------
# Funktionen zusammenstellen
#---------------------------------------------------

def get_gold_standard_ids(conn):
    cursor = conn.cursor()
    rows = cursor.execute("SELECT id FROM gold_standard_alt_text").fetchall()
    cursor.close()
    return [r[0] for r in rows]


def get_judge_model(model_name, conn):
    cursor = conn.cursor()
    try:
        # Return the DB id and the *slug* (model_series), not the URL
        cursor.execute("SELECT id, model_series FROM model WHERE model_series = ?", (model_name,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Model '{model_name}' nicht gefunden.")
        return row[0], row[1]  # (model_id, model_series)
    finally:
        cursor.close()


def evaluation_pipeline(
    func_name_judge_clarity,
    func_name_completeness,
    func_name_conciseness,
    func_name_preceived_completeness,
    func_name_neutrality,
    func_name_correctness,
    api_key_path,
    judge_model_name,
    conn
):
    judge_model_id, judge_model_series = get_judge_model(model_name=judge_model_name, conn=conn)
    gold_standard_id_lst = get_gold_standard_ids(conn)

    def load_and_compile_prompt_text_function(prompt_text_function_name, conn):
        judge_function_id, function_code = load_prompt_text_function(
            prompt_text_function_name=prompt_text_function_name,
            conn=conn
        )
        judge_fn = compile_function_from_string(
            function_code=function_code,
            function_name=prompt_text_function_name
        )
        return judge_function_id, judge_fn

    func_clarity_id, clarity_fn = load_and_compile_prompt_text_function(func_name_judge_clarity, conn)
    func_completeness_id, completeness_fn = load_and_compile_prompt_text_function(func_name_completeness, conn)
    func_conciseness_id, conciseness_fn = load_and_compile_prompt_text_function(func_name_conciseness, conn)
    func_preceived_completeness_id, preceived_completeness_fn = load_and_compile_prompt_text_function(func_name_preceived_completeness, conn)
    func_neutrality_id, neutrality_fn = load_and_compile_prompt_text_function(func_name_neutrality, conn)
    func_correctness_id, correctness_fn = load_and_compile_prompt_text_function(func_name_correctness, conn)

    for gold_standard_id in gold_standard_id_lst:

        row = conn.execute("""
            SELECT short_description_metadata, short_description_overview, long_description
            FROM gold_standard_alt_text
            WHERE id = ?
        """, (gold_standard_id,)).fetchone()

        if row is None:
            raise ValueError(f"No text with this gold_standard_id: {gold_standard_id}")

        meta, overview, long_desc = row
        meta = meta or ""
        overview = overview or ""
        long_desc = long_desc or ""

        tokens_meta = len(meta)
        tokens_overview = len(overview)
        tokens_long = len(long_desc)

        # --- clarity ---
        prompt, chart_id = construct_llm_as_a_judge_prompt(gold_standard_id, clarity_fn, conn)
        resp = get_multimodal_model_output(prompt, chart_id, api_key_path, judge_model_series)
        score_clarity, reason_clarity = extract_judge_response(resp)

        # --- completeness ---
        prompt, chart_id = construct_llm_as_a_judge_prompt(gold_standard_id, completeness_fn, conn)
        resp = get_multimodal_model_output(prompt, chart_id, api_key_path, judge_model_series)
        score_completeness, reason_completeness = extract_judge_response(resp)

        # --- conciseness ---
        prompt, chart_id = construct_llm_as_a_judge_prompt(gold_standard_id, conciseness_fn, conn)
        resp = get_multimodal_model_output(prompt, chart_id, api_key_path, judge_model_series)
        score_conciseness, reason_conciseness = extract_judge_response(resp)

        # --- perceived completeness ---
        prompt, chart_id = construct_llm_as_a_judge_prompt(gold_standard_id, preceived_completeness_fn, conn)
        resp = get_multimodal_model_output(prompt, chart_id, api_key_path, judge_model_series)
        score_preceived_completeness, reason_pc = extract_judge_response(resp)

        # --- neutrality ---
        prompt, chart_id = construct_llm_as_a_judge_prompt(gold_standard_id, neutrality_fn, conn)
        resp = get_multimodal_model_output(prompt, chart_id, api_key_path, judge_model_series)
        score_neutrality, reason_neu = extract_judge_response(resp)

        # --- correctness ---
        prompt, chart_id = construct_llm_as_a_judge_prompt(gold_standard_id, correctness_fn, conn)
        resp = get_multimodal_model_output(prompt, chart_id, api_key_path, judge_model_series)
        score_correctness, reason_corr = extract_judge_response(resp)

        # --- UPDATE in gold_standard_alt_text ---
        update_gold_standard_alt_text_row(
        gold_standard_id=gold_standard_id,
        tokens_short_description_metadata=tokens_meta,
        tokens_short_description_overview=tokens_overview,
        tokens_long_description=tokens_long,

        judge_model_id=judge_model_id,
        func_clarity_id=func_clarity_id,
        func_completeness_id=func_completeness_id,
        func_conciseness_id=func_conciseness_id,
        func_preceived_completeness_id=func_preceived_completeness_id,
        func_neutrality_id=func_neutrality_id,
        func_correctness_id=func_correctness_id,

        score_clarity=score_clarity,
        reason_clarity=reason_clarity,
        score_completeness=score_completeness,
        reason_completeness=reason_completeness,
        score_conciseness=score_conciseness,
        reason_conciseness=reason_conciseness,
        score_preceived_completeness=score_preceived_completeness,
        reason_preceived_completeness=reason_pc,
        score_neutrality=score_neutrality,
        reason_neutrality=reason_neu,
        score_correctness=score_correctness,
        reason_correctness=reason_corr,

        conn=conn
    )







