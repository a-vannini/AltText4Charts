# Viz4VisuallyImpaired

This project focuses on the automatic generation, evaluation, and analysis of alternative texts (short: alt texts) for NZZ charts, with the goal of improving accessibility for visually impaired people (PIVs).

It combines data preprocessing, LLM-based alt text generation, different evaluation approaches as LLM-based juding, SBERT, qualitative feedback through interviews, character and visualization analysis.

### Project goals

- Prepare and normalize chart datasets (NZZ data)
- Generate alt texts for charts using Large Language Models (LLMs) in this case Google’s Gemini 2.5 Flash
- Store charts, metadata, and generated alt texts in a SQLite database
- Automatically evaluate alt texts using multiple evaluation methods
- Compare generated texts against a gold standard that has been created on our own through interviews with PIVs and a linguistic expert
- Analyze and report evaluation results

## Synthetisised alt text structure
snthetisized structure see ./visuals/alt_text_structure.png
in case of an NZZ example this looks like this: ./visuals/visual_example.png


## Data
include graphic ./visuals/chart_type_distribution.png
scope: bar, line and stacked bar charts


## Enviroment
Every time the environment is started:
pipenv shell
pipenv sync

## Folder Structure

```text
VIZ4VISUALLYIMPAIRED/
├── data/
│   ├── NZZ_original/              # Original raw NZZ files
│   ├── nzz_metadata.csv           # Chart metadata
│
├── notebooks/
│   ├── a_generate_dfs_for_db.ipynb
│   ├── b_create_db_for_chart_data.ipynb
│   ├── c_alt_text_generation_pipeline.ipynb
│   ├── c1_check_insert_db.ipynb
│   ├── d1_llm_as_a_judge_evaluation_pipeline.ipynb
│   ├── d2_llm_as_a_judge_golden_standard.ipynb
│   ├── e_viz_analysis.ipynb
│   ├── f_best_text_all_texts_per_chart_id.ipynb
│   └── gold_standard_alt_texts_raw.txt
│
├── outputs/
│   ├── eval_figures/                  # Evaluation plots
│   ├── report_out_gold_standard/      # Gold standard evaluation reports
│   └── report_out_run/                # Model run reports
│
├── src/
│   ├── a_func_generate_dfs_for_db.py
│   ├── b_func_prompt_texts.py
│   ├── c_func_alt_text_generation_pipeline.py
│   ├── d1_func_llm_as_a_judge_generated.py
│   ├── d2_func_llm_as_a_judge_gold_standard.py
│   └── e_func_viz_pipeline.py
│
├── report.pdf
├── gold_standards.pdf
├── chart_database.db                  # Main SQLite database
├── Pipfile
├── Pipfile.lock
└── README.md
```


## Pipeline Overview

### Data Preparation
Raw chart data and metadata are cleaned and transformed into structured DataFrames

Notebooks: 'a_generate_dfs_for_db.ipynb'

### Database Creation
A SQLite database is created to store: Chart metadata, Data values, Generated alt texts, Evaluation results

Notebooks: 'b_create_db_for_chart_data.ipynb'

### Alt Text Generation
Alt texts are generated using prompt templates and LLMs. Multiple candidate texts can be generated per chart.

Notebooks: 'c_alt_text_generation_pipeline.ipynb'

### Evaluation (LLM as a Judge)
Generated alt texts are evaluated using an LLM acting as a judge. Evaluation criteria include clarity, completeness, perceived completeness, conciseness, neutrality, and factual correctness.
Notebooks: d1_llm_as_a_judge_evaluation_pipeline.ipynb


### Gold Standard Comparison
Generated texts are compared against manually written gold standard alt texts.

Notebooks: d2_llm_as_a_judge_golden_standard.ipynb


### Visualization Analysis & Reporting
Results are aggregated and visualized. Best-performing alt texts are selected per chart. Reports and figures are exported.

Notebooks: 'e_viz_analysis.ipynb', 'f_best_text_all_texts_per_chart_id.ipynb'

### Outputs
Evaluation plots: outputs/eval_figures/
Generated Alt-Texts plus evaluations: 
- Model run reports: outputs/generated_alt_texts
- LLM-judged Gold standards: outputs/LLMjudged_gold_standard_alt_texts


