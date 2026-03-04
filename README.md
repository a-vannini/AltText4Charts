# Viz4VisuallyImpaired

This project focuses on the automatic generation, evaluation, and analysis of alternative texts (short: alt texts) for NZZ charts, with the goal of improving accessibility for visually impaired people (PIVs).

It combines data preprocessing, LLM-based alt text generation, LLM-as-a-Judge evaluation, and visualization analysis, using real-world chart data.

### Project goals

- Prepare and normalize chart datasets (NZZ data)

- Generate high-quality alt texts for charts using Large Language Models

- Store charts, metadata, and generated texts in a SQLite database

- Automatically evaluate alt texts using LLM-as-a-Judge methods

- Compare generated texts against a gold standard

- Analyze and report evaluation results

## Data
OneDrive Alessia: https://1drv.ms/f/c/bf8fbf60ddff38a6/EsdHLsKeOaNClnKux02P8gABKLn5kh5wQ3rfwLsFJS41vw?e=fmPeEg


## Enviroment
Every time the environment is started:
pipenv shell
pipenv sync

## Folder Structure

```text
VIZ4VISUALLYIMPAIRED/
├── data/
│   ├── data_nzz_csv/              # Generated CSV for each chart
│   ├── db_tables/                 # Tables for databse
│   ├── NZZ_original/              # Original raw NZZ files
│   ├── api_key.txt                # API key (not committed)
│   ├── nzz_metadata.csv           # Chart metadata
│   └── Used_charts.xlsx           # Table of charts used where
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
│   ├── print_all_charts.ipynb
│   ├── print_charts_after_filter.ipynb
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
├── Report_CX_NZZ.pdf
├── chart_database.db                  # Main SQLite database
├── chart_database_backup.db           # Backup database
├── Pipfile
├── Pipfile.lock
└── README.md
```




## Pipeline Overview

### Data Preparation
Raw chart data and metadata are cleaned and transformed into structured DataFrames

**Notebooks**: 'a_generate_dfs_for_db.ipynb'

### Database Creation
A SQLite database is created to store: Chart metadata, Data values, Generated alt texts, Evaluation results

**Notebooks**: 'b_create_db_for_chart_data.ipynb'

### Alt Text Generation
Alt texts are generated using prompt templates and LLMs. Multiple candidate texts can be generated per chart.

**Notebooks**: 'c_alt_text_generation_pipeline.ipynb'

### Evaluation (LLM as a Judge)
Generated alt texts are evaluated using an LLM acting as a judge. Evaluation criteria include clarity, completeness, perceived completeness, conciseness, neutrality, and factual correctness.

**Notebook**: d1_llm_as_a_judge_evaluation_pipeline.ipynb


### Gold Standard Comparison
Generated texts are compared against manually written gold standard alt texts.

**Notebook**: d2_llm_as_a_judge_golden_standard.ipynb


### Visualization Analysis & Reporting
Results are aggregated and visualized. Best-performing alt texts are selected per chart. Reports and figures are exported.

**Notebook**: 'e_viz_analysis.ipynb', 'f_best_text_all_texts_per_chart_id.ipynb'

### Outputs
Evaluation plots: outputs/eval_figures/

Generated/Golden-Standard Alt-Texts plus evaluations: outputs/report_out_*

- Model run reports: outputs/report_out_run/

- Gold standard comparison reports: outputs/report_out_gold_standard/


