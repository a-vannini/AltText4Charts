import pandas as pd
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from matplotlib.patches import Patch
import re
from matplotlib.ticker import MaxNLocator



def get_csv_from_db(cursor, generation_run_ids_lst):
    """
    Pro Zeile ein alt_text.
    Liefert DataFrame mit:
      chart_id (TEXT), alt_text_id, generation_run_id, temp, variation_no,
      people_evaluation_ids, gold_standard_id, embedding_ids, metrics_ids
    """
    if not generation_run_ids_lst:
        return pd.DataFrame(columns=[
            "chart_id","alt_text_id","generation_run_id","temp","variation_no",
            "people_evaluation_ids","gold_standard_id","embedding_ids","metrics_ids"
        ]).convert_dtypes()

    placeholders = ",".join(["?"] * len(generation_run_ids_lst))

    query = f"""
        SELECT
            a.id                         AS alt_text_id,
            a.chart_id                   AS chart_id,
            a.generation_run_id          AS generation_run_id,
            gr.temperature               AS temp,
            a.variant_no                 AS variation_no,

            pe.pe_ids                    AS people_evaluation_ids,
            gsa.id                       AS gold_standard_id,
            emb.emb_ids                  AS embedding_ids,
            mt.mt_ids                    AS metrics_ids

        FROM alt_text a
        JOIN generation_run gr
              ON gr.id = a.generation_run_id

        LEFT JOIN (
            SELECT alt_text_id, GROUP_CONCAT(id) AS pe_ids
            FROM people_evaluation
            GROUP BY alt_text_id
        ) pe ON pe.alt_text_id = a.id

        LEFT JOIN gold_standard_alt_text gsa
               ON gsa.chart_id = a.chart_id

        LEFT JOIN (
            SELECT alt_text_id, GROUP_CONCAT(id) AS emb_ids
            FROM alt_text_embedding
            GROUP BY alt_text_id
        ) emb ON emb.alt_text_id = a.id

        LEFT JOIN (
            SELECT alt_text_id, GROUP_CONCAT(id) AS mt_ids
            FROM metric
            GROUP BY alt_text_id
        ) mt ON mt.alt_text_id = a.id

        WHERE a.generation_run_id IN ({placeholders})
        ORDER BY a.chart_id, a.generation_run_id, a.variant_no;
    """

    cursor.execute(query, generation_run_ids_lst)
    rows = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]

    df = pd.DataFrame(rows, columns=colnames).convert_dtypes()

    ordered_cols = [
        "chart_id","alt_text_id","generation_run_id","temp","variation_no",
        "people_evaluation_ids","gold_standard_id","embedding_ids","metrics_ids"
    ]
    return df[[c for c in ordered_cols if c in df.columns]]



def filter_csv_and_get_data_for_rq1_in_db(csv_data, db_path="../chart_database.db"):
    if csv_data is None or len(csv_data) == 0:
        raise ValueError("csv_data ist leer.")

    df_gen = csv_data.loc[csv_data.temp == 1.0]
    if df_gen.empty:
        raise ValueError("Keine Generated-Alt-Texte mit temperature==1 gefunden.")

    chart_ids = sorted(df_gen["chart_id"].dropna().unique().tolist())
    if not chart_ids:
        raise ValueError("Keine chart_id in den Generated Daten gefunden.")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # --- Kategorien ---
    q_cat = f"""
        SELECT c.id AS chart_id,
               ct.type AS chart_type,
               CASE WHEN c.complex = 1 THEN 'complex' ELSE 'simple' END AS complexity
        FROM chart c
        JOIN chart_type ct ON ct.id = c.chart_type_id
        WHERE c.id IN ({",".join(["?"]*len(chart_ids))})
    """
    cursor.execute(q_cat, chart_ids)
    cat_df = pd.DataFrame(cursor.fetchall(), columns=[d[0] for d in cursor.description])
    cat_df["cat6"] = cat_df["chart_type"] + "_" + cat_df["complexity"]

    # --- Gold ---
    # NOTE: embedding_metadata/overview/long_description müssen existieren (wie du vorher ergänzt hast)
    q_gold = f"""
        SELECT
            id AS gold_standard_id,
            chart_id,
            short_description_metadata,
            short_description_overview,
            long_description,
            embedding,  -- total (bestehend)
            embedding_metadata,
            embedding_overview,
            embedding_long_description,
            tokens_short_description_metadata,
            tokens_short_description_overview,
            tokens_long_description,
            score_clarity,
            score_completeness,
            score_conciseness,
            score_preceived_completeness,
            score_neutrality,
            score_correctness
        FROM gold_standard_alt_text
        WHERE chart_id IN ({",".join(["?"]*len(chart_ids))})
    """
    cursor.execute(q_gold, chart_ids)
    gold_df = pd.DataFrame(cursor.fetchall(), columns=[d[0] for d in cursor.description])
    gold_df["source"] = "gold"

    if gold_df.empty:
        raise ValueError("Keine Einträge in gold_standard_alt_text zu den Charts gefunden.")

    # --- Generated IDs ---
    gen_ids = df_gen[["alt_text_id", "chart_id", "generation_run_id", "temp", "variation_no"]].copy()
    gen_ids["source"] = "generated"
    alt_ids_all = gen_ids["alt_text_id"].dropna().unique().tolist()
    if not alt_ids_all:
        raise ValueError("Keine alt_text_id in Generated Daten gefunden.")

    # --- Generated Texte (Tokens) ---
    q_texts = f"""
        SELECT
            alt_text_id,
            tokens_short_description_metadata,
            tokens_short_description_overview,
            tokens_long_description
        FROM metric
        WHERE alt_text_id IN ({",".join(["?"]*len(alt_ids_all))})
    """
    cursor.execute(q_texts, alt_ids_all)
    texts_df = pd.DataFrame(cursor.fetchall(), columns=[d[0] for d in cursor.description]).convert_dtypes()

    # --- LLM Scores (Generated) ---
    q_scores = f"""
        SELECT
            alt_text_id,
            score_clarity,
            score_conciseness,
            score_preceived_completeness,
            score_neutrality,
            score_completeness,
            score_correctness
        FROM llm_evaluation
        WHERE alt_text_id IN ({",".join(["?"]*len(alt_ids_all))})
    """
    cursor.execute(q_scores, alt_ids_all)
    scores_gen = pd.DataFrame(cursor.fetchall(), columns=[d[0] for d in cursor.description]).convert_dtypes()

    # --- Embeddings (Generated): total + parts aus alt_text_embedding ---
    # Wähle pro alt_text_id genau 1 Zeile: bevorzugt normalized=1, sonst höchstes id
    # und lies zusätzlich metadata_embedding/overview_embedding/long_description_embedding aus derselben Zeile.
    q_emb = f"""
        SELECT t.alt_text_id,
               t.embedding AS total_embedding,
               t.metadata_embedding,
               t.overview_embedding,
               t.long_description_embedding
        FROM alt_text_embedding t
        JOIN (
            SELECT alt_text_id,
                   COALESCE(MAX(CASE WHEN normalized = 1 THEN id END), MAX(id)) AS chosen_id
            FROM alt_text_embedding
            WHERE alt_text_id IN ({",".join(["?"]*len(alt_ids_all))})
            GROUP BY alt_text_id
        ) pick
        ON pick.alt_text_id = t.alt_text_id AND pick.chosen_id = t.id
    """
    cursor.execute(q_emb, alt_ids_all)
    emb_gen = pd.DataFrame(cursor.fetchall(), columns=[d[0] for d in cursor.description]).convert_dtypes()

    def _to_vec(blob):
        if blob is None:
            return None
        return np.frombuffer(blob, dtype=np.float32)

    # Generated vecs
    emb_gen["vec_total"] = emb_gen["total_embedding"].apply(_to_vec)
    emb_gen["vec_meta"] = emb_gen["metadata_embedding"].apply(_to_vec)
    emb_gen["vec_overview"] = emb_gen["overview_embedding"].apply(_to_vec)
    emb_gen["vec_long"] = emb_gen["long_description_embedding"].apply(_to_vec)
    emb_gen = emb_gen.drop(columns=["total_embedding", "metadata_embedding", "overview_embedding", "long_description_embedding"])

    # Gold vecs (by chart)
    gold_vec_df = gold_df[[
        "chart_id",
        "embedding",
        "embedding_metadata",
        "embedding_overview",
        "embedding_long_description",
    ]].copy()

    gold_vec_df["vec_total"] = gold_vec_df["embedding"].apply(_to_vec)
    gold_vec_df["vec_meta"] = gold_vec_df["embedding_metadata"].apply(_to_vec)
    gold_vec_df["vec_overview"] = gold_vec_df["embedding_overview"].apply(_to_vec)
    gold_vec_df["vec_long"] = gold_vec_df["embedding_long_description"].apply(_to_vec)
    gold_vec_df = gold_vec_df.drop(columns=["embedding", "embedding_metadata", "embedding_overview", "embedding_long_description"])

    gold_vec_by_chart = {
        r.chart_id: {
            "total": r.vec_total,
            "meta": r.vec_meta,
            "overview": r.vec_overview,
            "long": r.vec_long,
        }
        for r in gold_vec_df.itertuples(index=False)
    }

    def _cosine(a, b):
        if a is None or b is None:
            return np.nan
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return np.nan
        return float(np.dot(a, b) / (na * nb))


    # --- Similarity DF (NUR Generated, gegen Gold) ---
    sim_records = []

    # Generated rows: cosine against corresponding gold vectors
    gen_with_vec = gen_ids.merge(emb_gen, on="alt_text_id", how="left")
    for r in gen_with_vec.itertuples(index=False):
        g = gold_vec_by_chart.get(r.chart_id, {})
        sim_records.append({
            "chart_id": r.chart_id,
            "alt_text_id": r.alt_text_id,
            "source": "generated",
            "similarity_meta": _cosine(getattr(r, "vec_meta", None), g.get("meta")),
            "similarity_overview": _cosine(getattr(r, "vec_overview", None), g.get("overview")),
            "similarity_long": _cosine(getattr(r, "vec_long", None), g.get("long")),
            "similarity_total": _cosine(getattr(r, "vec_total", None), g.get("total")),
        })

    similarity_df = pd.DataFrame(sim_records).merge(cat_df, on="chart_id", how="left").convert_dtypes()
    similarity_df["cat6"] = similarity_df["chart_type"] + "_" + similarity_df["complexity"]


    # --- Best/Worst (Generated) ---
    # Für best/worst nehmen wir TOTAL similarity (wie zuvor similarity_to_gold)
    best_worst = (
        similarity_df.query("source=='generated'")
        .sort_values(["chart_id", "similarity_total"], ascending=[True, False])
        .groupby("chart_id", as_index=False)
        .agg(
            best_alt_text_id=("alt_text_id", "first"),
            best_similarity=("similarity_total", "max"),
            worst_alt_text_id=("alt_text_id", "last"),
            worst_similarity=("similarity_total", "min"),
        )
        .merge(
            df_gen[["alt_text_id", "variation_no"]].rename(columns={"variation_no": "best_variant_no"}),
            left_on="best_alt_text_id", right_on="alt_text_id", how="left"
        )
        .drop(columns=["alt_text_id"])
        .merge(
            df_gen[["alt_text_id", "variation_no"]].rename(columns={"variation_no": "worst_variant_no"}),
            left_on="worst_alt_text_id", right_on="alt_text_id", how="left"
        )
        .drop(columns=["alt_text_id"])
        .convert_dtypes()
    )

    # --- LENGTHS/TOKENS (Gold + Gen) ---
    lengths_gen = (
        gen_ids.merge(texts_df, on="alt_text_id", how="left")
              .merge(cat_df, on="chart_id", how="left")
    )
    lengths_gen["cat6"] = lengths_gen["chart_type"] + "_" + lengths_gen["complexity"]
    lengths_gen["source"] = "generated"

    lengths_gold = gold_df[[
        "chart_id",
        "tokens_short_description_metadata",
        "tokens_short_description_overview",
        "tokens_long_description"
    ]].copy()
    lengths_gold["source"] = "gold"
    lengths_gold = lengths_gold.merge(cat_df, on="chart_id", how="left")
    lengths_gold["cat6"] = lengths_gold["chart_type"] + "_" + lengths_gold["complexity"]
    for col in ["alt_text_id", "generation_run_id", "temp", "variation_no"]:
        lengths_gold[col] = pd.NA

    lengths_df = pd.concat([
        lengths_gold[[
            "chart_id","source",
            "tokens_short_description_metadata","tokens_short_description_overview","tokens_long_description",
            "chart_type","complexity","cat6",
            "alt_text_id","generation_run_id","temp","variation_no"
        ]],
        lengths_gen[[
            "chart_id","source",
            "tokens_short_description_metadata","tokens_short_description_overview","tokens_long_description",
            "chart_type","complexity","cat6",
            "alt_text_id","generation_run_id","temp","variation_no"
        ]],
    ], ignore_index=True).convert_dtypes()

    # --- SCORES (Gold + Generated) ---
    score_cols = [
        "score_clarity","score_conciseness","score_preceived_completeness",
        "score_neutrality","score_completeness","score_correctness"
    ]

    # Generated
    scores_gen_df = (
        gen_ids[["alt_text_id","chart_id"]]
        .merge(scores_gen, on="alt_text_id", how="left")
        .merge(cat_df, on="chart_id", how="left")
    )
    scores_gen_df["cat6"] = scores_gen_df["chart_type"] + "_" + scores_gen_df["complexity"]
    scores_gen_df["source"] = "generated"

    # Gold
    for c in score_cols:
        if c not in gold_df.columns:
            gold_df[c] = pd.NA

    scores_gold = gold_df[["chart_id"] + score_cols].copy()
    scores_gold["source"] = "gold"
    scores_gold = scores_gold.merge(cat_df, on="chart_id", how="left")
    scores_gold["cat6"] = scores_gold["chart_type"] + "_" + scores_gold["complexity"]
    scores_gold["alt_text_id"] = pd.NA

    scores_df = pd.concat(
        [
            scores_gold[["alt_text_id","chart_id","source","cat6","chart_type","complexity"] + score_cols],
            scores_gen_df[["alt_text_id","chart_id","source","cat6","chart_type","complexity"] + score_cols],
        ],
        ignore_index=True
    ).convert_dtypes()

    # --- UMAP (Gold + Generated, nur mit vec_total) ---
    umap_gen = (
        gen_ids.merge(emb_gen[["alt_text_id","vec_total"]], on="alt_text_id", how="left")
              .merge(cat_df, on="chart_id", how="left")
    )
    umap_gen["source"] = "generated"
    umap_gen["cat6"] = umap_gen["chart_type"] + "_" + umap_gen["complexity"]

    umap_gold = gold_vec_df.merge(cat_df, on="chart_id", how="left")
    umap_gold["alt_text_id"] = pd.NA
    umap_gold["source"] = "gold"
    umap_gold["cat6"] = umap_gold["chart_type"] + "_" + umap_gold["complexity"]

    # Für UMAP verwenden wir total embedding
    umap_gold = umap_gold.rename(columns={"vec_total": "vec"})
    umap_gen = umap_gen.rename(columns={"vec_total": "vec"})

    umap_meta = pd.concat([
        umap_gold[["alt_text_id","chart_id","source","chart_type","complexity","cat6","vec"]],
        umap_gen[["alt_text_id","chart_id","source","chart_type","complexity","cat6","vec"]],
    ], ignore_index=True)

    has_vec = umap_meta["vec"].notna()
    X = np.stack(umap_meta.loc[has_vec, "vec"].to_list()) if has_vec.any() else np.zeros((0, 0), dtype=np.float32)
    umap_meta = umap_meta.loc[has_vec, ["alt_text_id","chart_id","source","chart_type","complexity","cat6"]].reset_index(drop=True)

    conn.close()

    return {
        "lengths_df": lengths_df,
        "scores_df": scores_df,
        "similarity_df": similarity_df,
        "similarity_best_worst": best_worst,
        "umap": {"X": X, "meta": umap_meta.convert_dtypes()}
    }


def filter_csv_and_get_data_for_rq2_in_db(csv_df: pd.DataFrame, db_path: str = "../chart_database.db") -> pd.DataFrame:
    score_cols = ["score_clarity", "score_conciseness", "score_neutrality", "score_preceived_completeness"]
    out_cols = ["alt_text_id", "evaluator", "chart_category"] + score_cols

    if csv_df is None or csv_df.empty:
        return pd.DataFrame(columns=out_cols)

    f = csv_df.loc[csv_df["generation_run_id"] == 1].copy()
    if f.empty:
        return pd.DataFrame(columns=out_cols)

    alt_ids = f["alt_text_id"].dropna().astype(int).unique().tolist()
    if not alt_ids:
        return pd.DataFrame(columns=out_cols)

    in_clause = lambda n: "(" + ",".join(["?"] * n) + ")"

    # people_evaluation_ids (optional): global + pro alt_text_id (Reihenfolge)
    pe_ids_all, pe_ids_by_alt = [], {}
    if "people_evaluation_ids" in f.columns:
        for _, r in f[["alt_text_id", "people_evaluation_ids"]].dropna().iterrows():
            aid = int(r["alt_text_id"])
            ids = [int(x) for x in str(r["people_evaluation_ids"]).split(",") if x.strip().isdigit()]
            if ids:
                pe_ids_by_alt[aid] = ids
                pe_ids_all.extend(ids)
        pe_ids_all = sorted(set(pe_ids_all))

    with sqlite3.connect(db_path) as conn:
        # chart_category meta
        chart_meta = pd.read_sql_query(
            f"""
            SELECT at.id AS alt_text_id, ct.type AS chart_type, c.complex AS complex_flag
            FROM alt_text at
            JOIN chart c ON c.id = at.chart_id
            JOIN chart_type ct ON ct.id = c.chart_type_id
            WHERE at.id IN {in_clause(len(alt_ids))}
            """,
            conn,
            params=alt_ids,
        )

        # human scores
        if pe_ids_all:
            human = pd.read_sql_query(
                f"""
                SELECT id AS people_evaluation_id, alt_text_id,
                       score_neutrality, score_clarity, score_conciseness, score_preceived_completeness
                FROM people_evaluation
                WHERE id IN {in_clause(len(pe_ids_all))}
                """,
                conn,
                params=pe_ids_all,
            )
        else:
            human = pd.read_sql_query(
                f"""
                SELECT id AS people_evaluation_id, alt_text_id,
                       score_neutrality, score_clarity, score_conciseness, score_preceived_completeness
                FROM people_evaluation
                WHERE alt_text_id IN {in_clause(len(alt_ids))}
                """,
                conn,
                params=alt_ids,
            )

        # llm scores (nur die 4 Spalten, die du später mittelst)
        llm = pd.read_sql_query(
            f"""
            SELECT alt_text_id, score_clarity, score_conciseness, score_neutrality, score_preceived_completeness
            FROM llm_evaluation
            WHERE alt_text_id IN {in_clause(len(alt_ids))}
            """,
            conn,
            params=alt_ids,
        )

    if human.empty:
        raise ValueError("Keine Human-Scores gefunden (people_evaluation).")
    if llm.empty:
        raise ValueError("Keine LLM-Scores gefunden (llm_evaluation).")

    # chart_category bauen
    def type_prefix(t: str) -> str:
        t = (t or "").lower()
        if "stack" in t:
            return "stacked_bar"
        if "bar" in t:
            return "bar"
        if "line" in t:
            return "line"
        return t.replace(" ", "_") if t else "unknown"

    def to_category(row) -> str:
        is_complex = bool(int(row["complex_flag"])) if pd.notna(row["complex_flag"]) else False
        return f"{type_prefix(row['chart_type'])}_{'complex' if is_complex else 'simple'}"

    chart_meta["chart_category"] = chart_meta.apply(to_category, axis=1)
    cat = dict(zip(chart_meta["alt_text_id"].astype(int), chart_meta["chart_category"]))

    llm_mean = llm.groupby("alt_text_id")[score_cols].mean(numeric_only=True)

    rows = []
    for aid in alt_ids:
        chart_category = cat.get(aid, "unknown")

        # humans: CSV-Reihenfolge wenn vorhanden, sonst first 5 by id
        if aid in pe_ids_by_alt:
            wanted = pe_ids_by_alt[aid]
            h = human[(human["alt_text_id"] == aid) & (human["people_evaluation_id"].isin(wanted))].copy()
            h["__ord"] = h["people_evaluation_id"].map({pid: i for i, pid in enumerate(wanted)})
            h = h.sort_values("__ord").drop(columns="__ord")
        else:
            h = human[human["alt_text_id"] == aid].sort_values("people_evaluation_id").head(5)

        h = h.reset_index(drop=True)

        for i in range(5):
            base = {"alt_text_id": aid, "evaluator": f"person{i+1}", "chart_category": chart_category}
            if i < len(h):
                rows.append({**base, **{c: h.loc[i, c] for c in score_cols}})
            else:
                rows.append({**base, **{c: np.nan for c in score_cols}})

        base = {"alt_text_id": aid, "evaluator": "llm_judge", "chart_category": chart_category}
        if aid in llm_mean.index:
            rows.append({**base, **llm_mean.loc[aid, score_cols].to_dict()})
        else:
            rows.append({**base, **{c: np.nan for c in score_cols}})

    df = pd.DataFrame(rows, columns=out_cols)

    eord = {f"person{i}": i for i in range(1, 6)} | {"llm_judge": 99}
    df["__eord"] = df["evaluator"].map(eord).fillna(999).astype(int)
    return df.sort_values(["chart_category", "alt_text_id", "__eord"]).drop(columns="__eord").reset_index(drop=True)


def filter_csv_and_get_data_for_rq3_in_db(
    csv_df: pd.DataFrame,
    db_path: str = "../chart_database.db",
) -> pd.DataFrame:
    """
    RQ3:
    - CSV filtern: generation_run_id == 4, temp == 1 (falls vorhanden)
    - embeddings pro alt_text_id aus alt_text_embedding holen
      (pro alt_text_id genau 1 Zeile: bevorzugt normalized=1, sonst höchste id)
    - BLOB -> numpy arrays (float32)

    Rückgabe: DataFrame mit mindestens:
      chart_id, alt_text_id, generation_run_id, temp, variation_no,
      chart_type, complexity, cat6,
      vec_total, vec_meta, vec_overview, vec_long
    """

    if csv_df is None or csv_df.empty:
        raise ValueError("csv_df ist leer.")

    # --- Filter: generation_run_id = 4
    if "generation_run_id" not in csv_df.columns:
        raise ValueError("csv_df muss die Spalte 'generation_run_id' enthalten.")

    df_gen = csv_df.loc[csv_df["generation_run_id"] == 4].copy()
    if df_gen.empty:
        raise ValueError("Keine Rows mit generation_run_id == 4 gefunden.")

    # --- Filter: temp = 1 (falls vorhanden)
    if "temp" in df_gen.columns:
        df_gen = df_gen.loc[pd.to_numeric(df_gen["temp"], errors="coerce") == 1].copy()
        if df_gen.empty:
            raise ValueError("Keine Rows mit temp == 1 nach dem Filtern gefunden.")

    # --- IDs
    if "alt_text_id" not in df_gen.columns:
        raise ValueError("csv_df muss die Spalte 'alt_text_id' enthalten.")
    if "chart_id" not in df_gen.columns:
        raise ValueError("csv_df muss die Spalte 'chart_id' enthalten.")

    df_gen["alt_text_id"] = pd.to_numeric(df_gen["alt_text_id"], errors="coerce").astype("Int64")
    df_gen["chart_id"] = df_gen["chart_id"].astype(str)

    alt_ids_all = df_gen["alt_text_id"].dropna().astype(int).unique().tolist()
    if not alt_ids_all:
        raise ValueError("Keine gültigen alt_text_id nach dem Filtern gefunden.")

    chart_ids = df_gen["chart_id"].dropna().astype(str).unique().tolist()
    if not chart_ids:
        raise ValueError("Keine gültigen chart_id nach dem Filtern gefunden.")

    def in_clause(n: int) -> str:
        return "(" + ",".join(["?"] * n) + ")"

    # --- DB loads
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # chart_category / cat6 wie zuvor
    q_cat = f"""
        SELECT c.id AS chart_id,
               ct.type AS chart_type,
               CASE WHEN c.complex = 1 THEN 'complex' ELSE 'simple' END AS complexity
        FROM chart c
        JOIN chart_type ct ON ct.id = c.chart_type_id
        WHERE c.id IN {in_clause(len(chart_ids))}
    """
    cur.execute(q_cat, chart_ids)
    cat_df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    cat_df["chart_id"] = cat_df["chart_id"].astype(str)
    cat_df["cat6"] = cat_df["chart_type"] + "_" + cat_df["complexity"]

    # Embeddings (Generated) aus alt_text_embedding
    # Spalten laut deinem Schema:
    #   embedding, meta_embedding, overview_embedding, long_description_embedding
    q_emb = f"""
        SELECT t.alt_text_id,
               t.embedding AS total_embedding,
               t.metadata_embedding AS meta_embedding,
               t.overview_embedding,
               t.long_description_embedding
        FROM alt_text_embedding t
        JOIN (
            SELECT alt_text_id,
                   COALESCE(MAX(CASE WHEN normalized = 1 THEN id END), MAX(id)) AS chosen_id
            FROM alt_text_embedding
            WHERE alt_text_id IN {in_clause(len(alt_ids_all))}
            GROUP BY alt_text_id
        ) pick
        ON pick.alt_text_id = t.alt_text_id AND pick.chosen_id = t.id
    """
    cur.execute(q_emb, alt_ids_all)
    emb_df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description]).convert_dtypes()

    conn.close()

    if emb_df.empty:
        raise ValueError("Keine Embeddings in alt_text_embedding für die gefilterten alt_text_id gefunden.")

    # --- BLOB -> numpy float32
    def _to_vec(blob):
        if blob is None:
            return None
        return np.frombuffer(blob, dtype=np.float32)

    emb_df["vec_total"] = emb_df["total_embedding"].apply(_to_vec)
    emb_df["vec_meta"] = emb_df["meta_embedding"].apply(_to_vec)
    emb_df["vec_overview"] = emb_df["overview_embedding"].apply(_to_vec)
    emb_df["vec_long"] = emb_df["long_description_embedding"].apply(_to_vec)

    emb_df = emb_df.drop(
        columns=["total_embedding", "meta_embedding", "overview_embedding", "long_description_embedding"],
        errors="ignore",
    )

    # --- join: csv-filtered rows + embeddings + categories
    base_cols = [c for c in ["chart_id", "alt_text_id", "generation_run_id", "temp", "variation_no"] if c in df_gen.columns]
    out = (
        df_gen[base_cols].drop_duplicates()
        .merge(emb_df, on="alt_text_id", how="left")
        .merge(cat_df, on="chart_id", how="left")
        .convert_dtypes()
    )

    # Optional: falls du genau 10 Alt-Texte pro chart_id brauchst, hier filtern
    # (so bleibt nur das "vollständige" Setup für RQ3)
    n_alt = out.groupby("chart_id")["alt_text_id"].nunique(dropna=True)
    keep_charts = n_alt[n_alt == 10].index.astype(str).tolist()
    out = out.loc[out["chart_id"].isin(keep_charts)].reset_index(drop=True)

    # Optional: sanity check: fehlende vecs zeigen
    # (nicht raisen, aber häufig hilfreich)
    # missing = out["vec_total"].isna().sum()
    # if missing:
    #     print(f"Warnung: {missing} Rows ohne vec_total.")

    return out


def filter_csv_and_get_data_for_rq4_in_db(
    csv_df: pd.DataFrame,
    db_path: str = "../chart_database.db",
) -> dict:
    """
    CSV filter: generation_run_id IN [2,4,6], temp = alle

    Returns dict with:
      - lengths_df: per alt_text_id length (token sum) + temp
      - similarity_df: within same (chart_id, generation_run_id, temp) all pairwise cosine sims
                       -> includes column 'temp' and 'similarity_total'
      - llm_scores_df: wide table (alt_text_id + 6 criteria + temp)
      - llm_scores_long_df: long table (criterion, score, temp) for plot #3
      - embeddings_total: {"X": np.ndarray, "meta": pd.DataFrame} (optional utility)
    """

    # -------------------------
    # 0) Validate + Filter CSV
    # -------------------------
    if csv_df is None or csv_df.empty:
        raise ValueError("csv_df ist leer.")

    needed_cols = {"alt_text_id", "chart_id", "generation_run_id"}
    missing = needed_cols - set(csv_df.columns)
    if missing:
        raise ValueError(f"csv_df fehlt Spalten: {sorted(missing)}")

    f = csv_df.loc[csv_df["generation_run_id"].isin([2, 4, 6])].copy()
    if f.empty:
        raise ValueError("Keine Zeilen mit generation_run_id in {2,4,6} gefunden.")

    # Ensure columns exist
    if "temp" not in f.columns:
        f["temp"] = pd.NA
    if "variation_no" not in f.columns:
        f["variation_no"] = pd.NA

    # temp numeric
    f["temp"] = pd.to_numeric(f["temp"], errors="coerce")

    # limit to temps used in your plots
    f = f[f["temp"].isin(TEMP_ORDER)].copy()
    if f.empty:
        raise ValueError(f"Keine Zeilen mit temp in {TEMP_ORDER} gefunden (nach Run-Filter).")

    # alt_text ids
    alt_ids = f["alt_text_id"].dropna().astype(int).unique().tolist()
    if not alt_ids:
        raise ValueError("Keine alt_text_id in den gefilterten Daten gefunden.")

    # helper
    def in_clause(n: int) -> str:
        return "(" + ",".join(["?"] * n) + ")"

    # -------------------------
    # 1) DB: Embeddings + Tokens + LLM Scores
    # -------------------------
    with sqlite3.connect(db_path) as conn:
        # Embedding (total) from alt_text_embedding, one chosen row per alt_text_id
        emb = pd.read_sql_query(
            f"""
            SELECT t.alt_text_id,
                   t.embedding AS total_embedding
            FROM alt_text_embedding t
            JOIN (
                SELECT alt_text_id,
                       COALESCE(MAX(CASE WHEN normalized = 1 THEN id END), MAX(id)) AS chosen_id
                FROM alt_text_embedding
                WHERE alt_text_id IN {in_clause(len(alt_ids))}
                GROUP BY alt_text_id
            ) pick
            ON pick.alt_text_id = t.alt_text_id AND pick.chosen_id = t.id
            """,
            conn,
            params=alt_ids,
        ).convert_dtypes()

        # Tokens/lengths from metric (same as rq1 logic)
        metric = pd.read_sql_query(
            f"""
            SELECT
                alt_text_id,
                tokens_short_description_metadata,
                tokens_short_description_overview,
                tokens_long_description
            FROM metric
            WHERE alt_text_id IN {in_clause(len(alt_ids))}
            """,
            conn,
            params=alt_ids,
        ).convert_dtypes()

        # LLM scores (6 criteria)
        llm = pd.read_sql_query(
            f"""
            SELECT
                alt_text_id,
                score_clarity,
                score_conciseness,
                score_preceived_completeness,
                score_neutrality,
                score_completeness,
                score_correctness
            FROM llm_evaluation
            WHERE alt_text_id IN {in_clause(len(alt_ids))}
            """,
            conn,
            params=alt_ids,
        ).convert_dtypes()

    if emb.empty:
        raise ValueError("Keine Embeddings in alt_text_embedding für die gefilterten alt_text_ids gefunden.")
    if metric.empty:
        raise ValueError("Keine Token-Daten in metric für die gefilterten alt_text_ids gefunden.")
    if llm.empty:
        raise ValueError("Keine LLM-Scores in llm_evaluation für die gefilterten alt_text_ids gefunden.")

    # -------------------------
    # 2) Bytes -> numpy vectors
    # -------------------------
    def _to_vec(blob):
        if blob is None or (isinstance(blob, float) and np.isnan(blob)):
            return None
        return np.frombuffer(blob, dtype=np.float32)

    emb["vec_total"] = emb["total_embedding"].apply(_to_vec)
    emb = emb.drop(columns=["total_embedding"])

    # -------------------------
    # 3) Meta join (CSV + vec + metric + llm)
    # -------------------------
    meta = f[["alt_text_id", "chart_id", "generation_run_id", "temp", "variation_no"]].copy()
    meta["alt_text_id"] = meta["alt_text_id"].astype(int)

    # metric
    for c in [
        "tokens_short_description_metadata",
        "tokens_short_description_overview",
        "tokens_long_description",
    ]:
        metric[c] = pd.to_numeric(metric[c], errors="coerce")

    metric["length_tokens_total"] = metric[
        ["tokens_short_description_metadata", "tokens_short_description_overview", "tokens_long_description"]
    ].sum(axis=1, min_count=1)

    # merge
    meta = (
        meta.merge(emb, on="alt_text_id", how="left")
            .merge(metric[["alt_text_id", "length_tokens_total"]], on="alt_text_id", how="left")
            .merge(llm, on="alt_text_id", how="left")
            .convert_dtypes()
    )

    # -------------------------
    # 4) Embeddings matrix X (optional utility)
    # -------------------------
    has_vec = meta["vec_total"].notna()
    meta_vec = meta.loc[has_vec].reset_index(drop=True)

    if len(meta_vec) == 0:
        X = np.zeros((0, 0), dtype=np.float32)
    else:
        X = np.stack(meta_vec["vec_total"].to_list()).astype(np.float32, copy=False)

    # -------------------------
    # 5) Similarities: within same (chart_id, generation_run_id, temp), all pairs
    # -------------------------
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return np.nan
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return np.nan
        return float(np.dot(a, b) / (na * nb))

    sim_rows = []
    for (cid, grid, t), g in meta_vec.groupby(["chart_id", "generation_run_id", "temp"], dropna=False):
        g = g.sort_values(["alt_text_id"]).reset_index(drop=True)
        aids = g["alt_text_id"].astype(int).to_list()
        vars_ = g["variation_no"].to_list()
        vecs = g["vec_total"].to_list()
        n = len(aids)
        if n < 2:
            continue

        for i in range(n):
            for j in range(i + 1, n):
                sim_rows.append(
                    {
                        "chart_id": cid,
                        "generation_run_id": grid,
                        "temp": t,
                        "alt_text_id_a": aids[i],
                        "alt_text_id_b": aids[j],
                        "variation_no_a": vars_[i],
                        "variation_no_b": vars_[j],
                        "similarity_total": _cosine(vecs[i], vecs[j]),
                    }
                )

    similarity_df = pd.DataFrame(sim_rows).convert_dtypes()
    if similarity_df.empty:
        similarity_df = pd.DataFrame(
            columns=[
                "chart_id",
                "generation_run_id",
                "temp",
                "alt_text_id_a",
                "alt_text_id_b",
                "variation_no_a",
                "variation_no_b",
                "similarity_total",
            ]
        ).convert_dtypes()

    # -------------------------
    # 6) lengths_df + llm scores dfs for plotting
    # -------------------------
    lengths_df = meta[["alt_text_id", "chart_id", "generation_run_id", "temp", "variation_no", "length_tokens_total"]].copy()

    llm_scores_df = meta[["alt_text_id", "chart_id", "generation_run_id", "temp", "variation_no"] + SCORE_COLS_6].copy()

    llm_scores_long_df = llm_scores_df.melt(
        id_vars=["alt_text_id", "chart_id", "generation_run_id", "temp", "variation_no"],
        value_vars=SCORE_COLS_6,
        var_name="criterion",
        value_name="score",
    )

    return {
        "lengths_df": lengths_df,
        "similarity_df": similarity_df,
        "llm_scores_df": llm_scores_df,
        "llm_scores_long_df": llm_scores_long_df,
        "embeddings_total": {
            "X": X,
            "meta": meta_vec.drop(columns=["vec_total"]).convert_dtypes(),
        },
    }

# =========================
# Helfer
# =========================

def _ensure_outdir(outdir):
    os.makedirs(outdir, exist_ok=True)

def _safe_name(s):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))

def _save(fig, outdir, name):
    _ensure_outdir(outdir)
    path = os.path.join(outdir, _safe_name(name) + ".png")
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return path

# =========================
# RQ1 Visualisierungen
# input_dict = output von filter_csv_and_get_data_for_rq1_in_db
# =========================


def visualize_rq1_similarity_boxplot(input_dict, outdir="../outputs/eval_figures/rq1"):
    """
    4 Boxplots:
    - Cosine Similarity für: metadata, overview, long_description, total
    - Kategorien: Chart Type (Line, Bar, StackedBar)
    - Hue: Complexity (simple / complex)
    """
    _ensure_outdir(outdir)

    similarity_df = input_dict.get("similarity_df")
    if similarity_df is None:
        raise ValueError("input_dict enthält keinen Key 'similarity_df'.")

    df = similarity_df.copy()

    # ---- HIER ggf. an deine echten Spaltennamen anpassen ----
    SIM_COLS = {
        "meta": "similarity_meta",
        "overview": "similarity_overview",
        "long": "similarity_long",
        "total": "similarity_total",
    }

    required = {"chart_type", "complexity"} | set(SIM_COLS.values())
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"similarity_df fehlt Spalten: {missing}")

    desired_types = ["Line", "Bar", "StackedBar"]

    # -------- Normalisierung --------
    def _norm_type(x):
        if x is None:
            return None
        s = str(x).strip().lower()
        if s == "line":
            return "Line"
        if s == "bar":
            return "Bar"
        if s in ("stackedbar", "stacked_bar", "stacked bar", "stacked-bar"):
            return "StackedBar"
        return str(x)

    def _norm_cplx(x):
        if x is None:
            return None
        s = str(x).strip().lower()
        if s in ("simple", "0", "false"):
            return "simple"
        if s in ("complex", "1", "true"):
            return "complex"
        return str(x)

    df["chart_type_norm"] = df["chart_type"].map(_norm_type)
    df["complexity_norm"] = df["complexity"].map(_norm_cplx)

    df = df[df["chart_type_norm"].isin(desired_types)].copy()
    df = df[df["complexity_norm"].isin(["simple", "complex"])].copy()
    if df.empty:
        raise ValueError("Keine Daten nach Filterung auf Chart-Type und Complexity.")

    types_present = [t for t in desired_types if (df["chart_type_norm"] == t).any()]
    if not types_present:
        raise ValueError("Keine Daten für chart_type in {Line, Bar, StackedBar} gefunden.")

    face = {"simple": "tab:blue", "complex": "tab:orange"}

    # -------- Plot-Helper (3 Kategorien, Hue = Complexity) --------
    def _plot_3cats_hue_complexity(value_series: str, title: str, filename: str):
        groups, meta = [], []  # meta: (chart_type, complexity)

        for t in types_present:
            for cplx in ["simple", "complex"]:
                vals = (
                    df.loc[
                        (df["chart_type_norm"] == t)
                        & (df["complexity_norm"] == cplx),
                        value_series,
                    ]
                    .dropna()
                    .astype(float)
                    .values
                )
                if len(vals) > 0:
                    groups.append(vals)
                    meta.append((t, cplx))

        if not groups:
            raise ValueError(f"Keine Werte für Plot '{title}' in Spalte '{value_series}'.")

        fig, ax = plt.subplots(figsize=(8.5, 7))

        base_pos = {t: i + 1 for i, t in enumerate(types_present)}  # 1..3
        offset = {"simple": -0.18, "complex": +0.18}
        positions = [base_pos[t] + offset[c] for (t, c) in meta]

        bp = ax.boxplot(
            groups,
            positions=positions,
            vert=False,
            widths=0.28,
            patch_artist=True,
            zorder=3,
            showfliers=True,  # <— Ausreißer anzeigen
            flierprops=dict(  # <— Ausreißer-Stil
                marker="o",
                markersize=3,     # kleine Punkte
                markerfacecolor="black",
                markeredgecolor="black",
                alpha=0.5,
                linestyle="none",
            ),
        )

        # Boxen einfärben
        for box, (t, cplx) in zip(bp["boxes"], meta):
            box.set_facecolor(face[cplx])
            box.set_alpha(0.6)
            box.set_linewidth(1.2)

        # Median-Linien
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(2.0)

        # y-Achse: nur 3 Kategorien
        ax.set_yticks([base_pos[t] for t in types_present])
        ax.set_yticklabels(types_present)

        ax.set_xlabel("SBERT similarity score")
        ax.set_ylabel("")
        # ax.set_title(title)
        
        ax.grid(axis="x", color='lightgrey', alpha=0.5, zorder=0)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelsize=12)
        ax.tick_params(axis="y", which="both", left=False, right=False, labelsize=12)

        # ---- Dynamischer x-Rand links, damit Boxen nicht über n-Labels laufen ----
        all_vals = np.concatenate([v for v in groups if len(v) > 0])
        xmin_data = float(np.nanmin(all_vals))
        xmax_data = float(np.nanmax(all_vals))

        span = max(1e-6, xmax_data - xmin_data)

        left_pad = 0.08 * span   # ~8% Platz links
        right_pad = 0.03 * span

        ax.set_xlim(xmin_data - left_pad, xmax_data + right_pad)


        ax.invert_yaxis()

        # n links
        x_axes = 0.02
        for y_pos, vals in zip(positions, groups):
            ax.text(
                x_axes, y_pos, f"n={len(vals)}",
                transform=ax.get_yaxis_transform(),
                ha="left", va="center",
                fontsize=8, color="0.5"
            )

        # Legende unten
        legend_handles = [
            Patch(facecolor=face["simple"], edgecolor="black", alpha=0.6, label="simple"),
            Patch(facecolor=face["complex"], edgecolor="black", alpha=0.6, label="complex"),
        ]
        ax.legend(
            handles=legend_handles,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.8, -0.2),
            ncol=2,
            fontsize=11,
        )

        fig.subplots_adjust(bottom=0.25)
        _save(fig, outdir, filename)

    # -------- 4 Plots --------
    _plot_3cats_hue_complexity(
        value_series=SIM_COLS["meta"],
        title="Cosine Similarity – Metadata by Chart Type × Complexity",
        filename="similarity_metadata_boxplot",
    )

    _plot_3cats_hue_complexity(
        value_series=SIM_COLS["overview"],
        title="Cosine Similarity – Overview by Chart Type × Complexity",
        filename="similarity_overview_boxplot",
    )

    _plot_3cats_hue_complexity(
        value_series=SIM_COLS["long"],
        title="Cosine Similarity – Long Description by Chart Type × Complexity",
        filename="similarity_long_boxplot",
    )

    _plot_3cats_hue_complexity(
        value_series=SIM_COLS["total"],
        title="Cosine Similarity – Total (Meta + Overview + Long) by Chart Type × Complexity",
        filename="similarity_total_boxplot",
    )


def visualize_rq1_length(rq1, outdir="../outputs/eval_figures/rq1"):
    _ensure_outdir(outdir)

    lengths_df = rq1.get("lengths_df")
    if lengths_df is None:
        raise ValueError("rq1 enthält keinen Key 'lengths_df'.")

    df = lengths_df.copy()

    required = {
        "source", "chart_type", "complexity",
        "tokens_short_description_metadata",
        "tokens_short_description_overview",
        "tokens_long_description",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"lengths_df fehlt Spalten: {missing}")

    desired_types = ["Line", "Bar", "StackedBar"]

    def _norm_type(x):
        if x is None:
            return None
        s = str(x).strip().lower()
        if s == "line":
            return "Line"
        if s == "bar":
            return "Bar"
        if s in ("stackedbar", "stacked_bar", "stacked bar", "stacked-bar"):
            return "StackedBar"
        return str(x)

    def _norm_cplx(x):
        if x is None:
            return None
        s = str(x).strip().lower()
        if s in ("simple", "0", "false"):
            return "simple"
        if s in ("complex", "1", "true"):
            return "complex"
        return str(x)

    def _norm_source(x):
        if x is None:
            return None
        s = str(x).strip().lower()
        if s in ("gold", "golden", "golden standards", "golden_standard", "golden-standards"):
            return "gold"
        if s in ("generated", "gen"):
            return "generated"
        return str(x)

    df["chart_type_norm"] = df["chart_type"].map(_norm_type)
    df["complexity_norm"] = df["complexity"].map(_norm_cplx)
    df["source_norm"] = df["source"].map(_norm_source)

    df = df[df["chart_type_norm"].isin(desired_types)].copy()
    df = df[df["source_norm"].isin(["gold", "generated"])].copy()
    if df.empty:
        raise ValueError("Keine Daten nach Filterung auf Chart-Type und source (gold/generated).")

    types_present = [t for t in desired_types if (df["chart_type_norm"] == t).any()]
    combo_order = [(t, c) for t in types_present for c in ["simple", "complex"]]
    sources_order = ["gold", "generated"]

    face_src = {"gold": "#F1A605", "generated": "#018032",}

    def _plot_6cats_hue_source(value_series: str, title: str, filename: str):
        """
        value_series: Spaltenname in df, der geplottet wird (z.B. tokens_long_description)
        """
        groups, meta = [], []  # meta: (type, complexity, source)
        for (t, c) in combo_order:
            for src in sources_order:
                vals = (
                    df.loc[
                        (df["chart_type_norm"] == t)
                        & (df["complexity_norm"] == c)
                        & (df["source_norm"] == src),
                        value_series,
                    ]
                    .dropna()
                    .astype(float)
                    .values
                )
                if len(vals) > 0:
                    groups.append(vals)
                    meta.append((t, c, src))

        if not groups:
            raise ValueError(f"Keine Werte für Plot '{title}' in Spalte '{value_series}' (nach DropNA).")

        fig, ax = plt.subplots(figsize=(8.5, max(4.8, len(combo_order) * 1.5)))

        base_pos = {tc: i + 1 for i, tc in enumerate(combo_order)}  # 1..6
        offset = {"gold": -0.18, "generated": +0.18}
        positions = [base_pos[(t, c)] + offset[src] for (t, c, src) in meta]

        bp = ax.boxplot(
            groups,
            positions=positions,
            vert=False,
            widths=0.28,
            zorder=3,
            patch_artist=True,
            showfliers=True,  # <— Ausreißer anzeigen
            flierprops=dict(  # <— Ausreißer-Stil
                marker="o",
                markersize=3,     # kleine Punkte
                markerfacecolor="black",
                markeredgecolor="black",
                alpha=0.5,
                linestyle="none",
            ),
        )


        # Boxen einfärben
        for box, (t, c, src) in zip(bp["boxes"], meta):
            box.set_facecolor(face_src[src])
            box.set_alpha(0.6)
            box.set_linewidth(1.2)

        # Median-Linien schwarz + dicker
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(2.0)

        # y ticks/labels: immer alle 6 Kategorien anzeigen
        ax.set_yticks([base_pos[tc] for tc in combo_order])
        ax.set_yticklabels([f"{t} ({c})" for (t, c) in combo_order])

        ax.set_xlabel("Character count")
        ax.set_ylabel("")
        # ax.set_title(title)

        ax.grid(axis="x", color='lightgrey', alpha=0.5, zorder=0)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelsize=12)
        ax.tick_params(axis="y", which="both", left=False, right=False, labelsize=12)

        # --- Mehr Platz links schaffen, damit Boxen nicht über die n= Labels laufen ---
        # Mindest-/Maximalwert aus allen Gruppen
        all_vals = np.concatenate([v for v in groups if len(v) > 0])
        xmin_data = float(np.nanmin(all_vals))
        xmax_data = float(np.nanmax(all_vals))

        # Breite des Wertebereichs (Fallback wenn alles gleich ist)
        span = max(1.0, xmax_data - xmin_data)

        # Linken Rand um z.B. 25% der Spannweite erweitern
        left_pad = 0.08 * span
        ax.set_xlim(xmin_data - left_pad, xmax_data + 0.05 * span)

        ax.invert_yaxis()

        # n links, konstanter Abstand
        x_axes = 0.02
        for y_pos, vals in zip(positions, groups):
            ax.text(
                x_axes, y_pos, f"n={len(vals)}",
                transform=ax.get_yaxis_transform(),
                ha="left", va="center",
                fontsize=8, color="0.5"
            )

        # Legende unten
        legend_handles = [
            Patch(facecolor=face_src["gold"], edgecolor="black", alpha=0.6, label="Gold-standard"),
            Patch(facecolor=face_src["generated"], edgecolor="black", alpha=0.6, label="LLM-generated"),
        ]
        ax.legend(
            handles=legend_handles,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.8, -0.2),
            ncol=2,
            fontsize=11,
        )

        fig.subplots_adjust(bottom=0.25)
        _save(fig, outdir, filename)

    # ---- 4 Plots erzeugen ----
    _plot_6cats_hue_source(
        value_series="tokens_short_description_metadata",
        title="Alt-Text Length (Tokens) – Metadata by Chart Type × Complexity and Source",
        filename="length_tokens_metadata_boxplot_6cats_hue_source",
    )

    _plot_6cats_hue_source(
        value_series="tokens_short_description_overview",
        title="Alt-Text Length (Tokens) – Overview by Chart Type × Complexity and Source",
        filename="length_tokens_overview_boxplot_6cats_hue_source",
    )

    _plot_6cats_hue_source(
        value_series="tokens_long_description",
        title="Alt-Text Length (Tokens) – Long Description by Chart Type × Complexity and Source",
        filename="length_tokens_long_boxplot_6cats_hue_source",
    )

    # Total = Summe aus allen drei Texten
    df["tokens_total_all"] = (
        df["tokens_short_description_metadata"].fillna(0)
        + df["tokens_short_description_overview"].fillna(0)
        + df["tokens_long_description"].fillna(0)
    )

    # Wenn du Zeilen willst, wo wirklich alles fehlt -> NaN statt 0:
    # mask_all_nan = df[["tokens_short_description_metadata","tokens_short_description_overview","tokens_long_description"]].isna().all(axis=1)
    # df.loc[mask_all_nan, "tokens_total_all"] = np.nan

    _plot_6cats_hue_source(
        value_series="tokens_total_all",
        title="Total Alt-Text Length (Tokens) – Meta + Overview + Long by Chart Type × Complexity and Source",
        filename="length_tokens_total_boxplot_6cats_hue_source",
    )


def visualize_rq1_llm_as_judge(input_dict, outdir="../outputs/eval_figures/rq1"):
    """
    Titel: Scores of LLM-as-Judge between Golden Standards & Generated Alt-Texts

    Plot:
    - y-Achse: Clarity, Conciseness, Completeness, Correctness
    - Hue: source ∈ {gold, generated}
    - x-Achse: Scores
    - Legende unterhalb (eine Zeile)
    - n= links (grau) mit zusätzlichem linken Abstand (xlim beginnt < 1)
    """
    _ensure_outdir(outdir)

    scores_df = input_dict["scores_df"].copy()

    score_cols = [
        ("score_clarity", "Clarity"),
        ("score_conciseness", "Conciseness"),
        ("score_completeness", "Completeness"),
        ("score_correctness", "Correctness"),
    ]

    # Validierung
    if "source" not in scores_df.columns:
        raise ValueError("scores_df muss die Spalte 'source' enthalten (gold/generated).")
    missing = [c for c, _ in score_cols if c not in scores_df.columns]
    if missing:
        raise ValueError(f"scores_df fehlt Spalten: {missing}")

    df = scores_df[scores_df["source"].isin(["gold", "generated"])].copy()
    if df.empty:
        raise ValueError("Keine Daten in scores_df für source ∈ {gold, generated} gefunden.")

    metrics = [lbl for _, lbl in score_cols]
    sources_order = ["gold", "generated"]
    face = {"gold": "#F1A605", "generated": "#018032",}

    # Daten sammeln (pro Metrik zwei Gruppen: gold/generated)
    groups, meta = [], []  # meta: (metric_label, source)
    for col, metric_label in score_cols:
        for src in sources_order:
            vals = df.loc[df["source"] == src, col].dropna().astype(float).values
            if len(vals) > 0:
                groups.append(vals)
                meta.append((metric_label, src))

    if not groups:
        raise ValueError("Keine Score-Werte vorhanden (nach DropNA).")

    # Plot
    fig, ax = plt.subplots(figsize=(8.5, 7))

    base_pos = {m: i + 1 for i, m in enumerate(metrics)}  # 1..n
    offset = {"gold": -0.18, "generated": +0.18}
    positions = [base_pos[m] + offset[src] for (m, src) in meta]

    bp = ax.boxplot(
        groups,
        positions=positions,
        vert=False,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        zorder=3
    )

    # Boxen einfärben
    for box, (m, src) in zip(bp["boxes"], meta):
        box.set_facecolor(face[src])
        box.set_alpha(0.6)
        box.set_linewidth(1.2)

    # Median-Linien schwarz + dicker
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(2.0)

    # Achsenbeschriftung (y)
    ax.set_yticks([base_pos[m] for m in metrics])
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Scores")
    ax.set_ylabel("")
    # ax.set_title("Scores of LLM-as-Judge between Golden Standards & Generated Alt-Texts")

    # Links Platz schaffen für n=, aber Ticks bei 1..5 lassen
    ax.set_xlim(0.5, 5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(axis="x", color='lightgrey', alpha=0.5, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelsize=12)
    ax.tick_params(axis="y", which="both", left=False, right=False, labelsize=12)

    # Reihenfolge: Clarity oben etc.
    ax.invert_yaxis()

    # n links (grau)
    x_text = 0.52
    for y_pos, vals in zip(positions, groups):
        ax.text(
            x_text, y_pos, f"n={len(vals)}",
            ha="left", va="center",
            fontsize=8, color="0.5"
        )

    # Legende unten, eine Zeile
    legend_handles = [
        Patch(facecolor=face["gold"], edgecolor="black", alpha=0.6, label="Gold-standard"),
        Patch(facecolor=face["generated"], edgecolor="black", alpha=0.6, label="LLM-generated"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.8, -0.2),
        ncol=2,
        fontsize=11,
    )

    fig.subplots_adjust(bottom=0.25)

    _save(fig, outdir, "llm_as_judge_scores_boxplot")


def visualize_rq1_length_by_conciseness_hue_source(rq1, outdir="../outputs/eval_figures/rq1"):
    """
    1 Boxplot:
    - x: Total Alt-Text Length (Tokens)
    - y: Conciseness Score (1/2/3)
    - Hue: Source (gold vs generated) in einem Plot
    - Style analog zu visualize_rq1_length / visualize_rq1_similarity_boxplot
    """
    _ensure_outdir(outdir)

    lengths_df = rq1.get("lengths_df")
    scores_df  = rq1.get("scores_df")
    if lengths_df is None:
        raise ValueError("rq1 enthält keinen Key 'lengths_df'.")
    if scores_df is None:
        raise ValueError("rq1 enthält keinen Key 'scores_df'.")

    len_df = lengths_df.copy()
    sc_df  = scores_df.copy()

    # ---- Required columns ----
    req_len = {
        "chart_id", "source",
        "tokens_short_description_metadata",
        "tokens_short_description_overview",
        "tokens_long_description",
        "alt_text_id",
    }
    req_sc = {"chart_id", "source", "score_conciseness", "alt_text_id"}

    miss_len = req_len - set(len_df.columns)
    miss_sc  = req_sc  - set(sc_df.columns)
    if miss_len:
        raise ValueError(f"lengths_df fehlt Spalten: {miss_len}")
    if miss_sc:
        raise ValueError(f"scores_df fehlt Spalten: {miss_sc}")

    # -------- Normalisierung --------
    def _norm_source(x):
        if x is None:
            return None
        s = str(x).strip().lower()
        if s in ("gold", "golden", "golden standards", "golden_standard", "golden-standards"):
            return "gold"
        if s in ("generated", "gen"):
            return "generated"
        return str(x)

    len_df["source_norm"] = len_df["source"].map(_norm_source)
    sc_df["source_norm"]  = sc_df["source"].map(_norm_source)

    # -------- Total length tokens --------
    for c in [
        "tokens_short_description_metadata",
        "tokens_short_description_overview",
        "tokens_long_description",
    ]:
        len_df[c] = pd.to_numeric(len_df[c], errors="coerce")

    len_df["tokens_total_all"] = (
        len_df["tokens_short_description_metadata"].fillna(0)
        + len_df["tokens_short_description_overview"].fillna(0)
        + len_df["tokens_long_description"].fillna(0)
    )

    # -------- Conciseness (LLM-as-judge) --------
    sc_df["score_conciseness"] = pd.to_numeric(sc_df["score_conciseness"], errors="coerce")

    # -------- Join (gold by chart_id, generated by alt_text_id) --------
    gold_len = len_df[len_df["source_norm"] == "gold"][
        ["chart_id", "source_norm", "tokens_total_all"]
    ]
    gold_sc = sc_df[sc_df["source_norm"] == "gold"][
        ["chart_id", "source_norm", "score_conciseness"]
    ]
    gold = gold_len.merge(gold_sc, on=["chart_id", "source_norm"], how="inner")

    gen_len = len_df[len_df["source_norm"] == "generated"][
        ["alt_text_id", "source_norm", "tokens_total_all"]
    ]
    gen_sc = sc_df[sc_df["source_norm"] == "generated"][
        ["alt_text_id", "score_conciseness"]
    ]
    gen = gen_len.merge(gen_sc, on="alt_text_id", how="inner")

    df = pd.concat([gold, gen], ignore_index=True)

    df = df.dropna(subset=["tokens_total_all", "score_conciseness"]).copy()

    # Conciseness as categories 1/2/3
    df["conciseness_cat"] = (
        pd.to_numeric(df["score_conciseness"], errors="coerce")
        .round()
        .astype("Int64")
    )
    df = df[df["conciseness_cat"].isin([1, 2, 3])].copy()
    if df.empty:
        raise ValueError("Keine Daten für Conciseness ∈ {1,2,3} nach Merge/Filter.")

    # -------- Plot setup (ähnlicher Style) --------
    conc_order = [3, 2, 1]
    sources_order = ["gold", "generated"]
    face_src = {"gold": "#F1A605", "generated": "#018032"}  

    groups, meta = [], []  # meta: (conc, source)
    for conc in conc_order:
        for src in sources_order:
            vals = (
                df.loc[
                    (df["conciseness_cat"] == conc) & (df["source_norm"] == src),
                    "tokens_total_all",
                ]
                .dropna()
                .astype(float)
                .values
            )
            if len(vals) > 0:
                groups.append(vals)
                meta.append((conc, src))

    if not groups:
        raise ValueError("Keine Werte für den Boxplot (nach DropNA).")

    fig, ax = plt.subplots(figsize=(8.5, 6))

    base_pos = {c: i + 1 for i, c in enumerate(conc_order)}  # 1..3
    offset = {"gold": -0.18, "generated": +0.18}
    positions = [base_pos[c] + offset[src] for (c, src) in meta]

    bp = ax.boxplot(
        groups,
        positions=positions,
        vert=False,
        widths=0.28,
        patch_artist=True,
        showfliers=True,
        zorder=3,
        flierprops=dict(
            marker="o",
            markersize=3,
            markerfacecolor="black",
            markeredgecolor="black",
            alpha=0.5,
            linestyle="none",
        ),
    )

    # Boxen einfärben
    for box, (conc, src) in zip(bp["boxes"], meta):
        box.set_facecolor(face_src[src])
        box.set_alpha(0.6)
        box.set_linewidth(1.2)

    # Median-Linien
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(2.0)

    # y-Achse: 1/2/3
    ax.set_yticks([base_pos[c] for c in conc_order])
    ax.set_yticklabels([str(c) for c in conc_order])

    ax.set_xlabel("Total alt text length (character count)")
    ax.set_ylabel("Conciseness Score")
    # ax.set_title("Total Length (Tokens) by Conciseness and Source")

    ax.grid(axis="x", color='lightgrey', alpha=0.5, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelsize=12)
    ax.tick_params(axis="y", which="both", left=False, right=False, labelsize=12)

    # --- Dynamischer x-Rand links, damit Boxen nicht über n-Labels laufen ---
    all_vals = np.concatenate([v for v in groups if len(v) > 0])
    xmin_data = float(np.nanmin(all_vals))
    xmax_data = float(np.nanmax(all_vals))
    span = max(1.0, xmax_data - xmin_data)
    left_pad = 0.1 * span
    right_pad = 0.02 * span
    ax.set_xlim(xmin_data - left_pad, xmax_data + right_pad)

    ax.invert_yaxis()

    # n links
    x_axes = 0.015
    for y_pos, vals in zip(positions, groups):
        ax.text(
            x_axes, y_pos, f"n={len(vals)}",
            transform=ax.get_yaxis_transform(),
            ha="left", va="center",
            fontsize=8, color="0.5"
        )

    # Legende unten (eine Zeile)
    legend_handles = [
        Patch(facecolor=face_src["gold"], edgecolor="black", alpha=0.6, label="Gold-standard"),
        Patch(facecolor=face_src["generated"], edgecolor="black", alpha=0.6, label="LLM-generated"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.75, -0.25),
        ncol=2,
        fontsize=11,
    )

    fig.subplots_adjust(bottom=0.25)
    _save(fig, outdir, "length_tokens_total_by_conciseness_hue_source_boxplot")




# =========================
# RQ2 Visualisierungen
# input_dict = output von filter_csv_and_get_data_for_rq2_in_db
# =========================


def visualize_rq2_summary(
    df: pd.DataFrame,
    figsize=(8, 8),
    outdir="../outputs/eval_figures/rq2",
    filename="summary.png",
):
    """
    Titel: Scores of Humans vs LLM-as-Judge (Summary)

    Plot (Style wie visualize_rq1_llm_as_judge):
    - y-Achse: 4 Kriterien
    - Hue: group ∈ {human(person1-5), llm_judge}
    - x-Achse: Scores
    - Legende unterhalb (eine Zeile)
    - n= links (grau) mit zusätzlichem linken Abstand (xlim beginnt < 1)
    - showfliers=False, median schwarz & dick, Boxen halbtransparent
    - Reihenfolge: erstes Kriterium oben (invert_yaxis)
    """
    os.makedirs(outdir, exist_ok=True)

    score_cols = [
        ("score_clarity", "Clarity"),
        ("score_conciseness", "Conciseness"),
        ("score_neutrality", "Neutrality"),
        ("score_preceived_completeness", "Perceived completeness"),
    ]

    # Validierung
    required = {"evaluator"}
    if not required.issubset(df.columns):
        raise ValueError(f"df muss Spalten enthalten: {sorted(required)}")
    missing = [c for c, _ in score_cols if c not in df.columns]
    if missing:
        raise ValueError(f"df fehlt Spalten: {missing}")

    d = df.copy()
    d["evaluator"] = d["evaluator"].astype(str)

    groups_order = ["human", "llm"]
    face = {"human": "#955803", "llm": "#027FD8"}
    offset = {"human": -0.18, "llm": +0.18}

    # Daten sammeln (pro Metrik zwei Gruppen: human/llm)
    groups, meta = [], []  # meta: (metric_label, group)
    for col, metric_label in score_cols:
        # humans = person1..5
        human_vals = d.loc[d["evaluator"].str.startswith("person"), col].dropna().astype(float).values
        llm_vals = d.loc[d["evaluator"] == "llm_judge", col].dropna().astype(float).values

        if len(human_vals) > 0:
            groups.append(human_vals)
            meta.append((metric_label, "human"))
        if len(llm_vals) > 0:
            groups.append(llm_vals)
            meta.append((metric_label, "llm"))

    if not groups:
        raise ValueError("Keine Score-Werte vorhanden (nach DropNA).")

    metrics = [lbl for _, lbl in score_cols]
    base_pos = {m: i + 1 for i, m in enumerate(metrics)}  # 1..n
    positions = [base_pos[m] + offset[g] for (m, g) in meta]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        groups,
        positions=positions,
        vert=False,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        zorder=3
    )

    # Boxen einfärben
    for box, (m, g) in zip(bp["boxes"], meta):
        box.set_facecolor(face[g])
        box.set_alpha(0.6)
        box.set_linewidth(1.2)

    # Median-Linien schwarz + dicker
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(2.0)

    # Achsenbeschriftung (y)
    ax.set_yticks([base_pos[m] for m in metrics])
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Scores")
    ax.set_ylabel("")
    # ax.set_title("Scores of Humans vs. LLM-as-Judge (Summary)")

    # Links Platz schaffen für n=, aber Ticks bei 1..5 lassen
    ax.set_xlim(0.55, 5.2)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(axis="x", color='lightgrey', alpha=0.5, zorder=0)
    ax.margins(y=-0.01)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelsize=12)
    ax.tick_params(axis="y", which="both", left=False, right=False, labelsize=12)

    # Reihenfolge: Clarity oben etc.
    ax.invert_yaxis()

    # n links (grau)
    x_text = 0.6
    for y_pos, vals in zip(positions, groups):
        ax.text(
            x_text, y_pos, f"n={len(vals)}",
            ha="left", va="center",
            fontsize=8, color="0.5"
        )

    # Legende unten, eine Zeile
    legend_handles = [
        Patch(facecolor=face["human"], edgecolor="black", alpha=0.6, label="Human"),
        Patch(facecolor=face["llm"], edgecolor="black", alpha=0.6, label="LLM"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.7, -0.1),
        ncol=2,
        frameon=False,
    )

    fig.subplots_adjust(bottom=0.25)

    save_path = os.path.join(outdir, filename)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def visualize_rq2_person_by_criterion(
    df: pd.DataFrame,
    figsize=(6, 14),
    outdir="../outputs/eval_figures/rq2",
    filename_prefix="criterion",
    y_spacing=1.6,
):
    """
    Pro Kriterium ein Plot.

    y-Achse: 12 Zeilen = (Basis-Kategorie + 1/2)
    x-Achse: Scores (1..5)

    Darstellung:
      - single humans: grau, alpha=0.2
      - mean human: orange, alpha=1.0
      - llm: blaues Quadrat, alpha=1.0

    Layout:
      - y-Achse reverse
      - keine horizontale Gridlines
      - feine Linie zwischen jedem Eintrag
      - dickere Linie nach jedem 2. Eintrag
      - Legende unter dem Plot
    """
    if df is None or df.empty:
        raise ValueError("df ist leer.")

    os.makedirs(outdir, exist_ok=True)

    criteria = [
        ("Clarity", "score_clarity"),
        ("Conciseness", "score_conciseness"),
        ("Neutrality", "score_neutrality"),
        ("Perceived completeness", "score_preceived_completeness"),
    ]

    base_order = [
        "line_simple",
        "line_complex",
        "bar_simple",
        "bar_complex",
        "stacked_bar_simple",
        "stacked_bar_complex",
    ]

    nice_label = {
        "line_simple": "Line Simple",
        "line_complex": "Line Complex",
        "bar_simple": "Bar Simple",
        "bar_complex": "Bar Complex",
        "stacked_bar_simple": "Stacked Bar Simple",
        "stacked_bar_complex": "Stacked Bar Complex",
    }

    # alt_text_id je Basis-Kategorie → 1 / 2
    y_rows = []
    for b in base_order:
        ids = (
            df.loc[df["chart_category"] == b, "alt_text_id"]
            .dropna()
            .astype(int)
            .unique()
        )
        for i, aid in enumerate(sorted(ids)[:2], start=1):
            y_rows.append((b, int(aid), f"{nice_label[b]} {i}"))

    if not y_rows:
        raise ValueError("Keine passenden chart_category / alt_text_id Kombinationen gefunden.")

    offsets = {
        "mean_human": +0.30,
        "person1": +0.20,
        "person2": +0.10,
        "person3":  0.00,
        "person4": -0.10,
        "person5": -0.20,
        "llm_judge": -0.35,
    }

    # y-Positionen (reverse Reihenfolge)
    y_base = np.arange(len(y_rows)) * y_spacing
    y_pos = y_base.max() - y_base

    saved = {}

    for crit_label, crit_col in criteria:
        fig, ax = plt.subplots(figsize=figsize)
        # ax.set_title(crit_label, fontsize=16, pad=40)

        # 🔹 Trennlinien NUR zwischen Einträgen
        for i in range(len(y_pos) - 1):
            y_mid = (y_pos[i] + y_pos[i + 1]) / 2

            # feine Linie
            ax.axhline(y_mid, color="#B8B7B7", linewidth=0.2, alpha=0.5, zorder=0)

            # dickere Linie nach jedem 2. Eintrag
            if i % 2 == 1:
                ax.axhline(y_mid, color="#B8B7B7", linewidth=0.2, alpha=0.8, zorder=0)

        for i, (base_cat, aid, _) in enumerate(y_rows):
            sub = df[
                (df["chart_category"] == base_cat)
                & (df["alt_text_id"].astype(int) == aid)
            ]

            # single humans
            for p in ["person1", "person2", "person3", "person4", "person5"]:
                prow = sub[sub["evaluator"] == p]
                if not prow.empty:
                    val = prow.iloc[0][crit_col]
                    if pd.notna(val):
                        ax.scatter(
                            val,
                            y_pos[i] + offsets[p],
                            s=18,
                            color="#955803",
                            alpha=0.1,
                            marker="o",
                            label="Single Human" if (i == 0 and p == "person1") else None,
                            zorder=3,
                        )

            # mean human
            humans = sub[sub["evaluator"].str.startswith("person")]
            if not humans.empty:
                mean_val = humans[crit_col].mean(numeric_only=True)
                if pd.notna(mean_val):
                    ax.scatter(
                        mean_val,
                        y_pos[i] + offsets["mean_human"],
                        s=90,
                        color="#955803",
                        alpha=1.0,
                        marker="o",
                        label="Mean Human" if i == 0 else None,
                        zorder=4,
                    )

            # llm
            lrow = sub[sub["evaluator"] == "llm_judge"]
            if not lrow.empty:
                val = lrow.iloc[0][crit_col]
                if pd.notna(val):
                    ax.scatter(
                        val,
                        y_pos[i] + offsets["llm_judge"],
                        s=90,
                        color="#027FD8",
                        alpha=1.0,
                        marker="o",
                        label="LLM" if i == 0 else None,
                        zorder=4,
                    )

        ax.set_yticks(y_pos)
        ax.set_yticklabels([r[2] for r in y_rows])
        ax.set_ylabel("")

        ax.set_xlim(0.8, 5.3)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Score")

        ax.grid(axis="x", color='lightgrey', alpha=0.5, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelsize=12)
        ax.tick_params(axis="y", which="both", left=False, right=False, labelsize=12)
        ax.margins(y=0.02)

        # Legende oben
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.7, -0.1),
            ncol=3,
            fontsize=11,
            frameon=False,
        )

        fig.tight_layout(rect=[0, 0, 1, 1])

        fname = f"{filename_prefix}_{crit_col}.png"
        save_path = os.path.join(outdir, fname)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        saved[crit_label] = save_path

    return saved

# =========================
# RQ3 Visualisierungen
# input_dict = output von filter_csv_and_get_data_for_rq3_in_db
# =========================



def _visualize_rq3_for_vector(
    df: pd.DataFrame,
    vec_col: str,
    outdir: str,
    suffix: str,
):
    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    os.makedirs(outdir, exist_ok=True)

    required = {"chart_id", "alt_text_id", vec_col, "chart_type", "complexity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df fehlt Spalten: {sorted(missing)}")

    HIST_COLOR = "#027FD8"

    PRETTY_LABELS = {
        "line_simple": "Line Simple",
        "line_complex": "Line Complex",
        "bar_simple": "Bar Simple",
        "bar_complex": "Bar Complex",
        "stacked_bar_simple": "Stacked Bar Simple",
        "stacked_bar_complex": "Stacked Bar Complex",
    }

    def _pairwise_cosine(vecs):
        X = np.stack(vecs).astype(np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = np.nan
        Xn = X / norms
        S = Xn @ Xn.T
        iu = np.triu_indices(S.shape[0], k=1)
        sims = S[iu]
        return sims[~np.isnan(sims)]

    def _safe_vecs(sub):
        return [v for v in sub[vec_col].tolist() if isinstance(v, np.ndarray)]

    def _floor_to_step(x, step=0.05):
        return float(np.floor(x / step) * step)

    def _norm_key(chart_type, complexity):
        ct = str(chart_type).strip()
        cx = str(complexity).strip().lower()
        ct = ct.replace("StackedBar", "stacked_bar").replace("stackedbar", "stacked_bar")
        ct = ct.replace("-", "_").replace(" ", "_").lower()
        ct = re.sub(r"__+", "_", ct)
        return f"{ct}_{cx}"

    desired_order = [
        "line_simple", "line_complex",
        "bar_simple", "bar_complex",
        "stacked_bar_simple", "stacked_bar_complex",
    ]
    order_index = {k: i for i, k in enumerate(desired_order)}

    def _rank(k):
        return order_index.get(k, 10_000)

    # ---------- Compute similarities ----------
    chart_sims = {}
    chart_meta = {}
    bad_charts = []

    for chart_id, g in df.groupby("chart_id"):
        vecs = _safe_vecs(g)
        if len(vecs) != 10:
            bad_charts.append((chart_id, len(vecs)))
            continue

        sims = _pairwise_cosine(vecs)
        if sims.size == 0:
            bad_charts.append((chart_id, len(vecs)))
            continue

        r0 = g.iloc[0]
        key_norm = _norm_key(r0["chart_type"], r0["complexity"])
        chart_sims[chart_id] = sims
        chart_meta[chart_id] = {"key_norm": key_norm}

    if not chart_sims:
        raise ValueError("Keine Charts mit genau 10 gültigen Vektoren gefunden.")

    # ---------- Global scale ----------
    all_sims = np.concatenate(list(chart_sims.values()))
    min_sim = float(np.min(all_sims))

    xlim = (0.3, 1.0)

    bins = np.arange(0.3, 1.0001, 0.05)   # Histogram-Auflösung (feiner ok)
    xticks = np.arange(0.3, 1.01, 0.1)    # Achsenbeschriftung


    import math

    # ---------- Per chart ----------
    chart_ids_sorted = sorted(
        chart_sims.keys(),
        key=lambda cid: (_rank(chart_meta[cid]["key_norm"]), chart_meta[cid]["key_norm"], str(cid)),
    )

    n = len(chart_ids_sorted)
    ncols = 4
    nrows = math.ceil(n / ncols)

    # Kürzere Höhe pro Zeile (statt 24 bei 8 Zeilen)
    row_h = 2.2  # <- hier kannst du feinjustieren (z.B. 2.0..2.5)
    fig_w = 17
    fig_h = max(6, nrows * row_h)

    fig1, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        sharex=True,
        constrained_layout=True,  # besser als tight_layout bei vielen Achsen
    )
    axes = np.atleast_2d(axes)

    for idx, ax in enumerate(axes.flat):
        if idx >= n:
            ax.axis("off")
            continue

        cid = chart_ids_sorted[idx]
        sims = chart_sims[cid]
        key = chart_meta[cid]["key_norm"]

        ax.hist(sims, bins=bins, color=HIST_COLOR, alpha=0.8)
        ax.set_title(f"{PRETTY_LABELS.get(key, key)} | {cid}", fontsize=9)
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_axisbelow(True)  # grid behind artists
        ax.grid(axis="x", color="lightgrey", alpha=0.5, zorder=0)
        ax.grid(axis="y", color="lightgrey", alpha=0.5, zorder=0)

        ax.tick_params(axis="x", which="both", bottom=False, top=False)
        ax.tick_params(axis="y", which="both", left=False, right=False)


        for s in ax.spines.values():
            s.set_visible(False)

        # y-label nur links
        if idx % ncols == 0:
            ax.set_ylabel("Count")

    # --- wichtig: x-Ticks/Label nur in der letzten Zeile, aber sicher sichtbar ---
    last_row_axes = axes[-1, :]
    for ax in last_row_axes:
        if ax.has_data():
            ax.set_xlabel(f"Cosine similarity ({vec_col})")
            ax.tick_params(axis="x", labelbottom=True)  # <- erzwingt Tick-Labels unten

    # Für alle anderen Reihen Tick-Labels aus (optional, ist bei sharex eh Standard)
    for ax in axes[:-1, :].flat:
        ax.tick_params(axis="x", labelbottom=False)

    p1 = os.path.join(outdir, f"rq3_hist_per_chart_{nrows}x{ncols}_{suffix}.png")

    # bbox_inches="tight" weglassen oder padding geben, damit unten nichts abgeschnitten wird
    fig1.savefig(p1, dpi=300, pad_inches=0.2)  # <- kein bbox_inches="tight"
    plt.close(fig1)


    # ---------- Aggregated ----------
    group_sims = {}
    for cid, sims in chart_sims.items():
        k = chart_meta[cid]["key_norm"]
        group_sims.setdefault(k, []).append(sims)

    grid = [
        ["line_simple", "bar_simple", "stacked_bar_simple"],
        ["line_complex", "bar_complex", "stacked_bar_complex"],
    ]

    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 8), sharex=True)

    for r in range(2):
        for c in range(3):
            ax = axes2[r, c]
            key = grid[r][c]

            if key not in group_sims:
                ax.axis("off")
                continue

            sims_all = np.concatenate(group_sims[key])
            ax.hist(sims_all, bins=bins, color=HIST_COLOR, alpha=0.8)
            ax.set_title(PRETTY_LABELS.get(key, key), fontsize=11)
            ax.set_xlim(*xlim)
            ax.set_xticks(xticks)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            ax.set_axisbelow(True)
            ax.grid(axis="x", color="lightgrey", alpha=0.5, zorder=0)
            ax.grid(axis="y", color="lightgrey", alpha=0.5, zorder=0)

            ax.tick_params(axis="x", which="both", bottom=False, top=False)
            ax.tick_params(axis="y", which="both", left=False, right=False)


            for s in ax.spines.values():
                s.set_visible(False)

            if c == 0:
                ax.set_ylabel("Count")
            if r == 1:
                ax.set_xlabel(f"Cosine similarity ({vec_col})")

    fig2.tight_layout()
    p2 = os.path.join(outdir, f"rq3_hist_by_cat6_2x3_{suffix}.png")
    fig2.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    return {
        "skipped_charts": bad_charts,
        "paths": {"per_chart": p1, "by_cat6": p2},
        "min_similarity": min_sim,
    }


def visualize_rq3_short_meta(df, outdir="../outputs/eval_figures/rq3"):
    return _visualize_rq3_for_vector(
        df=df,
        vec_col="vec_meta",
        outdir=outdir,
        suffix="short_meta",
    )


def visualize_rq3_short_overview(df, outdir="../outputs/eval_figures/rq3"):
    return _visualize_rq3_for_vector(
        df=df,
        vec_col="vec_overview",
        outdir=outdir,
        suffix="short_overview",
    )


def visualize_rq3_long_description(df, outdir="../outputs/eval_figures/rq3"):
    return _visualize_rq3_for_vector(
        df=df,
        vec_col="vec_long",
        outdir=outdir,
        suffix="long_description",
    )


def visualize_rq3_total_text(df, outdir="../outputs/eval_figures/rq3"):
    """
    RQ3: Inter-generation consistency based on embeddings of the full alt text
    (meta + overview + long description combined).
    """
    return _visualize_rq3_for_vector(
        df=df,
        vec_col="vec_total",
        outdir=outdir,
        suffix="total_text",
    )




def _umap_rq3_for_vector(
    df,
    vec_col,
    outdir,
    suffix,
    n_neighbors=15,
    min_dist=0.15,
    random_state=42,
):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import umap.umap_ as umap

    os.makedirs(outdir, exist_ok=True)

    DOT_COLOR = "#027FD8"

    required = {"chart_id", "temp", vec_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df fehlt Spalten: {sorted(missing)}")

    # --- keep only valid vectors ---
    dfv = df[df[vec_col].apply(lambda x: isinstance(x, np.ndarray))].copy()

    X = np.stack(dfv[vec_col].to_list()).astype(np.float32)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    emb2d = reducer.fit_transform(X)

    dfv["u1"] = emb2d[:, 0]
    dfv["u2"] = emb2d[:, 1]

    temps = sorted(dfv["temp"].unique())
    charts = sorted(dfv["chart_id"].unique())

    # --- plot ---
    fig, axes = plt.subplots(1, len(temps), figsize=(6 * len(temps), 5), sharex=True, sharey=True)
    if len(temps) == 1:
        axes = [axes]

    cmap = plt.cm.tab20
    color_map = {cid: cmap(i % 20) for i, cid in enumerate(charts)}

    for ax, t in zip(axes, temps):
        sub = dfv[dfv["temp"] == t]

        for cid, g in sub.groupby("chart_id"):
            ax.scatter(
                g["u1"],
                g["u2"],
                s=30,
                alpha=0.75,
                color=DOT_COLOR
            )

        # ax.set_title(f"Temperature = {t}", fontsize=11)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(alpha=0.3)

    # legend (once)
    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(
    #     handles,
    #     labels,
    #     loc="center right",
    #     bbox_to_anchor=(1.15, 0.5),
    #     title="Chart ID",
    #     fontsize=9,
    # )

    fig.tight_layout()
    path = os.path.join(outdir, f"rq3_umap_{suffix}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "path": path,
        "n_points": len(dfv),
        "n_charts": len(charts),
        "temps": temps,
    }


def visualize_rq3_umap_meta(df, outdir="../outputs/eval_figures/rq3_umap"):
    """
    UMAP for RQ3 based on meta-level alt-text embeddings.
    """
    return _umap_rq3_for_vector(
        df=df,
        vec_col="vec_meta",
        outdir=outdir,
        suffix="meta",
    )

def visualize_rq3_umap_overview(df, outdir="../outputs/eval_figures/rq3_umap"):
    """
    UMAP for RQ3 based on overview-level alt-text embeddings.
    """
    return _umap_rq3_for_vector(
        df=df,
        vec_col="vec_overview",
        outdir=outdir,
        suffix="overview",
    )

def visualize_rq3_umap_long(df, outdir="../outputs/eval_figures/rq3_umap"):
    """
    UMAP for RQ3 based on long-description alt-text embeddings.
    """
    return _umap_rq3_for_vector(
        df=df,
        vec_col="vec_long",
        outdir=outdir,
        suffix="long_description",
    )

def visualize_rq3_umap_total(df, outdir="../outputs/eval_figures/rq3_umap"):
    """
    UMAP for RQ3 based on full alt-text embeddings (meta + overview + long).
    """
    return _umap_rq3_for_vector(
        df=df,
        vec_col="vec_total",
        outdir=outdir,
        suffix="total_text",
    )


# =========================
# RQ4 Visualisierungen
# input_dict = output von filter_csv_and_get_data_for_rq3_in_db
# =========================
from pathlib import Path


TEMP_ORDER = [1.6, 1.0, 0.4]
SCORE_COLS_6 = [
    "score_clarity",
    "score_conciseness",
    "score_preceived_completeness",
    "score_neutrality",
    "score_completeness",
    "score_correctness",
]



def _ensure_temp_order(df: pd.DataFrame, temp_col: str = "temp") -> pd.DataFrame:
    df = df.copy()
    df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
    df = df[df[temp_col].isin(TEMP_ORDER)]
    df[temp_col] = pd.Categorical(df[temp_col], categories=TEMP_ORDER, ordered=True)
    return df


def _boxplot_horizontal_styled(
    ax,
    groups,
    positions,
    meta,
    facecolor_by_key,
    key_index_in_meta: int,
    widths=0.28,
    showfliers=True,
    xlabel="",
    ylabel="",
    title="",
    left_pad_ratio=0.16,
    right_pad_ratio=0.03,
    legend=None,
):
    bp = ax.boxplot(
        groups,
        positions=positions,
        vert=False,
        widths=widths,
        patch_artist=True,
        showfliers=showfliers,
        flierprops=dict(
            marker="o",
            markersize=3,
            markerfacecolor="black",
            markeredgecolor="black",
            alpha=0.5,
            linestyle="none",
        ),
        manage_ticks=False,
        zorder=3,
    )

    # Boxen einfärben
    for box, m in zip(bp["boxes"], meta):
        key = m[key_index_in_meta]
        box.set_facecolor(facecolor_by_key[key])
        box.set_alpha(0.6)
        box.set_linewidth(1.2)

    # Median-Linien
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(2.0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_title(title)
    ax.grid(axis="x", color='lightgrey', alpha=0.5, zorder=0)

    # ---- Dynamischer x-Rand links, damit Boxen nicht über n-Labels laufen ----
    all_vals = np.concatenate([v for v in groups if len(v) > 0])
    xmin_data = float(np.nanmin(all_vals))
    xmax_data = float(np.nanmax(all_vals))
    span = max(1e-6, xmax_data - xmin_data)

    left_pad = left_pad_ratio * span
    right_pad = right_pad_ratio * span
    ax.set_xlim(xmin_data - left_pad, xmax_data + right_pad)

    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelsize=12)
    ax.tick_params(axis="y", which="both", left=False, right=False, labelsize=12)

    # n links
    x_axes = 0.02
    for y_pos, vals in zip(positions, groups):
        ax.text(
            x_axes, y_pos, f"n={len(vals)}",
            transform=ax.get_yaxis_transform(),
            ha="left", va="center",
            fontsize=8, color="0.5",
        )

    if legend is not None:
        ax.legend(
            handles=legend.get("handles", []),
            labels=legend.get("labels", []),
            title=legend.get("title", None),
            loc=legend.get("loc", "lower cente"),
            bbox_to_anchor=legend.get("bbox_to_anchor", (0.8, -0.2)),
            ncol=legend.get("ncol", 3),
            frameon=legend.get("frameon", False),
        )


def _safe_savefig(fig, path, **kwargs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **kwargs)



# 1) Length vs Temperature (ohne Hue -> keine Legende)
def plot_rq4_lengths_by_temperature(
    rq4_data: dict,
    path: str = "../outputs/eval_figures/rq4/length_vs_temp.png",
):
    lengths_df = rq4_data["lengths_df"].copy()
    lengths_df["length_tokens_total"] = pd.to_numeric(lengths_df["length_tokens_total"], errors="coerce")
    lengths_df = _ensure_temp_order(lengths_df, "temp").dropna(subset=["length_tokens_total", "temp"])

    face_temp = {
        TEMP_ORDER[0]: "#771D1D",
        TEMP_ORDER[1]: "#5D5C5C",
        TEMP_ORDER[2]: "#1F3FBE",
    }

    groups, meta, positions = [], [], []
    base_pos = {t: i + 1 for i, t in enumerate(TEMP_ORDER)}  # 1..3

    for t in TEMP_ORDER:
        vals = lengths_df.loc[lengths_df["temp"] == t, "length_tokens_total"].dropna().astype(float).values
        if len(vals) > 0:
            groups.append(vals)
            meta.append((t,))
            positions.append(base_pos[t])

    if not groups:
        raise ValueError("Keine Daten für plot_rq4_lengths_by_temperature (nach DropNA).")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    _boxplot_horizontal_styled(
        ax=ax,
        groups=groups,
        positions=positions,
        meta=meta,
        facecolor_by_key=face_temp,
        key_index_in_meta=0,
        widths=0.32,
        showfliers=True,
        xlabel="Character count",
        ylabel="Generation temperature (LLM)",
        title="Length of alt texts across generation temperatures",
        legend=None,  # keine Legende
    )

    ax.set_yticks([base_pos[t] for t in TEMP_ORDER])
    ax.set_yticklabels([str(t) for t in TEMP_ORDER])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelsize=12)
    ax.tick_params(axis="y", which="both", left=False, right=False, labelsize=12)

    plt.tight_layout()
    _safe_savefig(fig, path, dpi=300, bbox_inches="tight")
    plt.close(fig)



# 2) Similarity Score vs Temperature (ohne Hue -> keine Legende)
def plot_rq4_similarity_by_temperature(
    rq4_data: dict,
    path: str = "../outputs/eval_figures/rq4/similarity_scores_vs_temp.png",
):
    sim = rq4_data["similarity_df"].copy()
    sim["similarity_total"] = pd.to_numeric(sim["similarity_total"], errors="coerce")
    sim = _ensure_temp_order(sim, "temp").dropna(subset=["similarity_total", "temp"])

    face_temp = {
        TEMP_ORDER[0]: "#771D1D",
        TEMP_ORDER[1]: "#5D5C5C",
        TEMP_ORDER[2]: "#1F3FBE",
    }

    groups, meta, positions = [], [], []
    base_pos = {t: i + 1 for i, t in enumerate(TEMP_ORDER)}  # 1..3

    for t in TEMP_ORDER:
        vals = sim.loc[sim["temp"] == t, "similarity_total"].dropna().astype(float).values
        if len(vals) > 0:
            groups.append(vals)
            meta.append((t,))
            positions.append(base_pos[t])

    if not groups:
        raise ValueError("Keine Daten für plot_rq4_similarity_by_temperature (nach DropNA).")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    _boxplot_horizontal_styled(
        ax=ax,
        groups=groups,
        positions=positions,
        meta=meta,
        facecolor_by_key=face_temp,
        key_index_in_meta=0,
        widths=0.32,
        showfliers=True,
        xlabel="SBERT similarity score",
        ylabel="Generation temperature (LLM)",
        title="Similarity scores across generation temperatures",
        legend=None,  # keine Legende
    )

    ax.set_yticks([base_pos[t] for t in TEMP_ORDER])
    ax.set_yticklabels([str(t) for t in TEMP_ORDER])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelsize=12)
    ax.tick_params(axis="y", which="both", left=False, right=False, labelsize=12)

    plt.tight_layout()
    _safe_savefig(fig, path, dpi=300, bbox_inches="tight")
    plt.close(fig)



# 3) LLM judge scores (Hue = temp -> Legende bleibt)
def plot_rq4_llm_scores_by_temperature(
    rq4_data: dict,
    path: str = "../outputs/eval_figures/rq4/llm_scores_vs_temp.png",
):
    long_df = rq4_data["llm_scores_long_df"].copy()
    long_df["score"] = pd.to_numeric(long_df["score"], errors="coerce")
    long_df = _ensure_temp_order(long_df, "temp").dropna(subset=["score", "temp", "criterion"])

    # Pretty labels for criteria (nur Anzeige; Spaltennamen bleiben unverändert)
    CRIT_LABELS = {
        "score_clarity": "Clarity",
        "score_conciseness": "Conciseness",
        "score_preceived_completeness": "Perceived Completeness",  # Spaltenname bleibt wie in deinen Daten
        "score_neutrality": "Neutrality",
        "score_completeness": "Completeness",
        "score_correctness": "Correctness",
    }

    # Reihenfolge beibehalten wie SCORE_COLS_6
    criteria = [c for c in SCORE_COLS_6 if c in CRIT_LABELS]

    face_temp = {
        TEMP_ORDER[0]: "#771D1D",
        TEMP_ORDER[1]: "#5D5C5C",
        TEMP_ORDER[2]: "#1F3FBE",
    }

    # Abstand zwischen Kriterien erhöhen
    crit_step = 1.5
    base_pos = {crit: 1 + i * crit_step for i, crit in enumerate(criteria)}

    # Abstand innerhalb eines Kriteriums (Temps)
    offsets = {TEMP_ORDER[0]: -0.32, TEMP_ORDER[1]: 0.0, TEMP_ORDER[2]: +0.32}

    groups, meta, positions = [], [], []
    for crit in criteria:
        for t in TEMP_ORDER:
            vals = (
                long_df.loc[(long_df["criterion"] == crit) & (long_df["temp"] == t), "score"]
                .dropna()
                .astype(float)
                .values
            )
            if len(vals) > 0:
                groups.append(vals)
                meta.append((crit, t))
                positions.append(base_pos[crit] + offsets[t])

    if not groups:
        raise ValueError("Keine Daten für plot_rq4_llm_scores_by_temperature (nach DropNA).")

    fig, ax = plt.subplots(figsize=(8.5, max(4.8, len(criteria) * 2.0)))

    _boxplot_horizontal_styled(
        ax=ax,
        groups=groups,
        positions=positions,
        meta=meta,
        facecolor_by_key=face_temp,
        key_index_in_meta=1,
        widths=0.22,
        showfliers=False,
        xlabel="Scores",
        ylabel="",  # keine y-Achsenbeschriftung
        title="LLM-as-a-Judge scores across generation temperatures",
        legend=dict(
            handles=[Patch(facecolor=face_temp[t], edgecolor="black", alpha=0.6, label=str(t)) for t in TEMP_ORDER],
            labels=[str(t) for t in TEMP_ORDER],
            bbox_to_anchor=(0.8, -0.122),
            ncol=3,
            frameon=False,
            loc="lower center",
        ),
    )

    ax.text(
        0.57, -0.1,                
        "Temperature:",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=11,
    )

    ax.set_yticks([base_pos[c] for c in criteria])
    ax.set_yticklabels([CRIT_LABELS[c] for c in criteria])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelsize=12)
    ax.tick_params(axis="y", which="both", left=False, right=False, labelsize=12)

    fig.subplots_adjust(bottom=0.15)
    _safe_savefig(fig, path, dpi=300, bbox_inches="tight")
    plt.close(fig)

