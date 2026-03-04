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
    

def construct_llm_as_a_judge_prompt(alt_text_id, judge_fn, conn, verbose=True):
    """
    output: judge_prompt_text, chart_id
    """
    cursor = conn.cursor()
    # --- DB lesen ---
    try:
        try:
            cursor.execute("""
                SELECT chart_id, short_description_metadata,
                       short_description_overview, long_description
                FROM alt_text
                WHERE id = ?
            """, (alt_text_id,))
            alt_row = cursor.fetchone()
            if alt_row is None:
                raise ValueError(f"Alt-Text mit id={alt_text_id} nicht gefunden.")
            chart_id, short_meta, short_overview, long_desc = alt_row
            combined_text = (
                f"Kurzbeschreibung (Metadaten): {short_meta}\n\n"
                f"Kurzbeschreibung (Überblick): {short_overview}\n\n"
                f"Lange Beschreibung:\n{long_desc}"
            )
        except Exception as e:
            raise ValueError("Alt-Text konnte nicht aus DB geholt werden.") from e

        # --- CSV lesen ---
        try:
            csv_path = os.path.join("..", "data", "data_nzz_csv", f"{chart_id}.csv")
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError("chart_id existiert nicht oder CSV konnte nicht gelesen werden.") from e

    finally:
        cursor.close()

    chart_data = df.to_string(index=False)
    judge_prompt_text = judge_fn(alt_text=combined_text, csv_text=chart_data)

    # if verbose:
    #     print(judge_prompt_text)

    return judge_prompt_text, chart_id



# def get_multimodal_model_output(
#         prompt_text, 
#         chart_id, 
#         api_key_path, 
#         model,
#         temperature):
    
#     img_path = os.path.join('..', 'data', 'NZZ_original', f'{chart_id}', f'{chart_id}.png')
#     image_base64 = encode_image_base64(img_path)
#     image_data_url = f"data:image/png;base64,{image_base64}"

#     api_key = load_api_key(api_key_path)

#     client = OpenAI(
#         base_url="https://openrouter.ai/api/v1",
#         api_key=api_key
#     )

#     completion = client.chat.completions.create(
#         extra_body={},
#         model=model,
#         temperature=temperature,
#         messages=[
#             {
#             "role": "user",
#             "content": [
#                 {
#                 "type": "image_url",
#                 "image_url": {"url": image_data_url}
#                 },
#                 {
#                 "type": "text",
#                 "text": prompt_text
#                 }
#             ]
#             }
#         ]
#     )
#     response = completion.choices[0].message.content

#     return response





def get_multimodal_model_output(
        prompt_text, 
        chart_id, 
        api_key_path, 
        model,
        temperature,
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
                temperature=temperature,
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


def _load_emb(blob):
    if blob is None:
        return None
    # 1) erst versuchen: pickle
    try:
        arr = pickle.loads(blob)
        return np.asarray(arr, dtype=float).ravel()
    except Exception:
        pass
    # 2) fallback: roher float32-Puffer
    try:
        return np.frombuffer(blob, dtype=np.float32).astype(float).ravel()
    except Exception:
        return None

def _cosine(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def get_sbert_score_to_gold_standard(alt_text_id, conn):
    cursor = conn.cursor()
    try:
        # 1) chart_id zu diesem alt_text_id holen
        cursor.execute("SELECT chart_id FROM alt_text WHERE id = ?", (alt_text_id,))
        row = cursor.fetchone()
        if not row or row[0] is None:
            raise ValueError(f"alt_text.id={alt_text_id} nicht gefunden oder ohne chart_id.")
        chart_id = row[0]

        # 2) Gold-Embedding mit chart_id laden
        cursor.execute("""
            SELECT embedding
            FROM gold_standard_alt_text
            WHERE chart_id = ?
            ORDER BY id DESC
            LIMIT 1
        """, (chart_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Kein Gold-Standard für chart_id='{chart_id}' gefunden.")
        gold_emb = _load_emb(row[0])
        if gold_emb is None:
            raise ValueError("Gold-Standard-Embedding konnte nicht geladen/parset werden.")

        # 3) Alt-Embedding mit alt_text_id laden (bei dir: genau eins vorhanden)
        cursor.execute("""
            SELECT embedding
            FROM alt_text_embedding
            WHERE alt_text_id = ?
            ORDER BY id DESC
            LIMIT 1
        """, (alt_text_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Kein Alt-Text-Embedding für alt_text_id={alt_text_id} gefunden.")
        alt_emb = _load_emb(row[0])
        if alt_emb is None:
            raise ValueError("Alt-Text-Embedding konnte nicht geladen/parset werden.")

        # 4) Cosine Similarity berechnen
        return _cosine(alt_emb, gold_emb)

    finally:
        cursor.close()




#---------------------------------------------------
# Fill tabels (metrics, evaluation_run, llm_evaluation)
#---------------------------------------------------

def fill_llm_evaluation(
        evaluation_run_id,
        alt_text_id,
        no_eval,
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
        conn):

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO llm_evaluation (
        evaluation_run_id,
        alt_text_id,
        no_eval,
        score_clarity,
        reasoning_clarity,
        score_completeness,
        reasoning_completeness,
        score_conciseness,
        reasoning_conciseness,
        score_preceived_completeness,
        reasoning_preceived_completeness,
        score_neutrality,
        reasoning_neutrality,
        score_correctness,
        reasoning_correctness)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            evaluation_run_id,
            alt_text_id,
            no_eval,
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
            reason_correctness
    ))

    conn.commit()
    cursor.close()


def fill_metric_table(
    alt_text_id,
    tokens_short_description_metadata,
    tokens_short_description_overview,
    tokens_long_description,
    conn
):
    """
    Achtung: Deine Tabelle metric hat die Spalten:
      - tokens_short_description_metadata
      - tokens_short_description_overview
      - tokens_long_description
    """
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO metric (
            alt_text_id,
            tokens_short_description_metadata,
            tokens_short_description_overview,
            tokens_long_description
        )
        VALUES (?, ?, ?, ?)
    """, (
        alt_text_id,
        tokens_short_description_metadata,
        tokens_short_description_overview,
        tokens_long_description
    ))
    metric_id = cursor.lastrowid
    conn.commit()
    cursor.close()

    #return metric_id


def fill_evaluation_run_table(
        generation_run_id,
        model_id,
        num_evals,
        func_clarity_id, 
        func_completeness_id, 
        func_conciseness_id, 
        func_preceived_completeness_id, 
        func_neutrality_id, 
        func_correctness_id,
        conn):

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO evaluation_run (
            generation_run_id,
            model_id,
            num_evals,
            func_clarity_id, 
            func_completeness_id, 
            func_conciseness_id, 
            func_preceived_completeness_id, 
            func_neutrality_id, 
            func_correctness_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        generation_run_id,
        model_id,
        num_evals,
        func_clarity_id, 
        func_completeness_id, 
        func_conciseness_id, 
        func_preceived_completeness_id, 
        func_neutrality_id, 
        func_correctness_id
    ))

    # ID des neuen Datensatzes
    new_id = cursor.lastrowid

    conn.commit()
    cursor.close()
    
    return new_id

def fill_similarity_to_gold_standard_table(
        alt_text_id,
        similarity_score,
        conn):
    
    cursor = conn.cursor() 

    cursor.execute("""
        INSERT INTO alt_text_similarity_to_gold_standard (
            alt_text_id,
            similarity_score
        )
        VALUES (?, ?)
    """, (
        alt_text_id,
        similarity_score
    ))

    conn.commit()
    cursor.close() 

#---------------------------------------------------
# Funktionen zusammenstellen
#---------------------------------------------------

def get_alt_text_ids_with_generation_run_id(generation_run_id, conn):

    cursor = conn.cursor()
    cursor.execute("""
        SELECT id FROM alt_text WHERE generation_run_id = ?
    """, (generation_run_id,))
    
    rows = cursor.fetchall()

    # Tupel entpacken → flache Liste
    chart_ids_lst = [row[0] for row in rows]

    # print(chart_ids_lst)

    cursor.close()

    return chart_ids_lst

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
        generation_run_id, 
        num_evals, 
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
    
    alt_text_ids_lst = get_alt_text_ids_with_generation_run_id(generation_run_id, conn)

    #Load judge functions

    def load_and_compile_prompt_text_function(prompt_text_function_name, conn):
        try:
            judge_function_id, function_code = load_prompt_text_function(
                prompt_text_function_name=prompt_text_function_name, 
                conn=conn)
            judge_fn = compile_function_from_string(
                function_code=function_code, 
                function_name=prompt_text_function_name)
            return judge_function_id, judge_fn
            
        except Exception as e:
            raise ValueError("Judge function konnte nicht geladen werden")

    judge_clarity_function_id, clarity_fn = load_and_compile_prompt_text_function(
        prompt_text_function_name=func_name_judge_clarity, 
        conn=conn)
    
    judge_completeness_function_id, completeness_fn = load_and_compile_prompt_text_function(
        prompt_text_function_name=func_name_completeness, 
        conn=conn)
    
    judge_conciseness_function_id, conciseness_fn = load_and_compile_prompt_text_function(
        prompt_text_function_name=func_name_conciseness, 
        conn=conn)
    
    judge_preceived_completeness_function_id, preceived_completeness_fn = load_and_compile_prompt_text_function(
        prompt_text_function_name=func_name_preceived_completeness, 
        conn=conn)
    
    judge_neutrality_function_id, neutrality_fn = load_and_compile_prompt_text_function(
        prompt_text_function_name=func_name_neutrality, 
        conn=conn)
    
    judge_correctness_function_id, correctness_fn = load_and_compile_prompt_text_function(
        prompt_text_function_name=func_name_correctness, 
        conn=conn)
    
    # fill evaluation_run table

    evaluation_run_id = fill_evaluation_run_table(
        generation_run_id=generation_run_id,
        model_id=judge_model_id,
        num_evals=num_evals,
        func_clarity_id=judge_clarity_function_id, 
        func_completeness_id=judge_completeness_function_id, 
        func_conciseness_id=judge_conciseness_function_id, 
        func_preceived_completeness_id=judge_preceived_completeness_function_id, 
        func_neutrality_id=judge_neutrality_function_id, 
        func_correctness_id=judge_correctness_function_id,
        conn=conn)


    for alt_text_id in alt_text_ids_lst:

        # Prüfen, ob alt_text_id in metric existiert
        exists = conn.execute("""
            SELECT 1 FROM metric WHERE alt_text_id = ? LIMIT 1
        """, (alt_text_id,)).fetchone()

        if not exists:
            # Alt-Text aus DB holen
            row = conn.execute("""
                SELECT short_description_metadata, short_description_overview, long_description
                FROM alt_text
                WHERE id = ?
            """, (alt_text_id,)).fetchone()

            if row is None:
                ValueError(f"No text with this alt_text_id: {alt_text_id}")

            meta, overview, long_desc = row

            # Länge berechnen
            short_description_meta_size = len(meta)
            short_description_uberblick_size = len(overview)
            long_description_size = len(long_desc)

            # In metric Tabelle einfügen
            fill_metric_table(
                alt_text_id=alt_text_id,
                tokens_short_description_metadata=short_description_meta_size,
                tokens_short_description_overview=short_description_uberblick_size,
                tokens_long_description=long_description_size,
                conn=conn
            )

        for no_eval in range(num_evals):

            # calculate sbert_score
            similarity_score_ = get_sbert_score_to_gold_standard(
                alt_text_id=alt_text_id,
                conn=conn)
            
            # fill alt_text_similarity_to_gold_standard            
            fill_similarity_to_gold_standard_table(
                alt_text_id=alt_text_id,
                similarity_score=similarity_score_,
                conn=conn)

            # clarity
            judge_clarity_prompt_text, chart_id = construct_llm_as_a_judge_prompt(
                alt_text_id=alt_text_id, 
                judge_fn=clarity_fn, 
                conn=conn,
                verbose=True)

            clarity_response = get_multimodal_model_output(
                    prompt_text=judge_clarity_prompt_text, 
                    chart_id=chart_id, 
                    api_key_path=api_key_path, 
                    model=judge_model_series,
                    temperature=None)

            score_clarity, reason_clarity = extract_judge_response(output_text=clarity_response)


            # completeness
            judge_completeness_prompt_text, chart_id = construct_llm_as_a_judge_prompt(
                alt_text_id=alt_text_id, 
                judge_fn=completeness_fn, 
                conn=conn,
                verbose=True)

            completeness_response = get_multimodal_model_output(
                    prompt_text=judge_completeness_prompt_text, 
                    chart_id=chart_id, 
                    api_key_path=api_key_path, 
                    model=judge_model_series,
                    temperature=None)

            score_completeness, reason_completeness = extract_judge_response(output_text=completeness_response)


            # conciseness
            judge_conciseness_prompt_text, chart_id = construct_llm_as_a_judge_prompt(
                alt_text_id=alt_text_id, 
                judge_fn=conciseness_fn, 
                conn=conn,
                verbose=True)

            conciseness_response = get_multimodal_model_output(
                    prompt_text=judge_conciseness_prompt_text, 
                    chart_id=chart_id, 
                    api_key_path=api_key_path, 
                    model=judge_model_series,
                    temperature=None)

            score_conciseness, reason_conciseness = extract_judge_response(output_text=conciseness_response)


            # preceived_completeness
            judge_preceived_completeness_prompt_text, chart_id = construct_llm_as_a_judge_prompt(
                alt_text_id=alt_text_id, 
                judge_fn=preceived_completeness_fn,
                conn=conn,
                verbose=True)

            preceived_completeness_response = get_multimodal_model_output(
                    prompt_text=judge_preceived_completeness_prompt_text, 
                    chart_id=chart_id, 
                    api_key_path=api_key_path, 
                    model=judge_model_series,
                    temperature=None)

            score_preceived_completeness, reason_preceived_completeness = extract_judge_response(output_text=preceived_completeness_response)
 

            # neutrality
            judge_neutrality_prompt_text, chart_id = construct_llm_as_a_judge_prompt(
                alt_text_id=alt_text_id, 
                judge_fn=neutrality_fn, 
                conn=conn,
                verbose=True)

            neutrality_response = get_multimodal_model_output(
                    prompt_text=judge_neutrality_prompt_text, 
                    chart_id=chart_id, 
                    api_key_path=api_key_path, 
                    model=judge_model_series,
                    temperature=None)

            score_neutrality, reason_neutrality = extract_judge_response(output_text=neutrality_response)


            # correctness
            judge_correctness_prompt_text, chart_id = construct_llm_as_a_judge_prompt(
                alt_text_id=alt_text_id, 
                judge_fn=correctness_fn, 
                conn=conn,
                verbose=True)

            correctness_response = get_multimodal_model_output(
                    prompt_text=judge_correctness_prompt_text, 
                    chart_id=chart_id, 
                    api_key_path=api_key_path, 
                    model=judge_model_series,
                    temperature=None)

            score_correctness, reason_correctness = extract_judge_response(output_text=correctness_response)


            # fill tabels
            fill_llm_evaluation(
                evaluation_run_id=evaluation_run_id,
                alt_text_id=alt_text_id,
                no_eval=no_eval,
                score_clarity=score_clarity,
                reason_clarity=reason_clarity,
                score_completeness=score_completeness,
                reason_completeness=reason_completeness,
                score_conciseness=score_conciseness,
                reason_conciseness=reason_conciseness,
                score_preceived_completeness=score_preceived_completeness,
                reason_preceived_completeness=reason_preceived_completeness,
                score_neutrality=score_neutrality,
                reason_neutrality=reason_neutrality,
                score_correctness=score_correctness,
                reason_correctness=reason_correctness,
                conn=conn
            )

    conn.close()







