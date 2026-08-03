# EUR-Lex Retrieval Dataset pipeline and experiment scripts

This folder contains the cleaned, IDE-oriented, archival versions of the EUR-Lex Retrieval Dataset construction and experiment scripts.

Open a script your Python IDE, review the configuration constants near the top of the file, and run the complete file.

The Python source code and scripts included in this repository are licensed under the MIT License (see /00_LICENSE_CODE.txt).

## Repository placement

Place `pipeline_common.py` and the numbered scripts together under either:

```text
<repository>/scripts/
```

or:

```text
<repository>/code/
```

The scripts locate the repository root automatically when it contains `data/` and either `scripts/` or `code/`.

When automatic detection is not possible, set the following constant in the scripts:

```python
from pathlib import Path

PROJECT_ROOT_OVERRIDE: Path | None = Path(
    r"C:\path\to\EUR-LexRD"
)
```

Keep `PROJECT_ROOT_OVERRIDE = None` in the archival version unless a local override is required.

## Installation

Python 3.10 or newer is recommended.

Install the dependencies with:

```text
python -m pip install -r requirements.txt
```

For CUDA or another hardware-specific configuration, install the relevant PyTorch build first, then install the remaining requirements.

The full requirements file covers:

- EUR-Lex collection and HTML parsing
- Gemini-based enrichment and embedding generation
- corpus statistics and graph analysis
- zero-shot dense retrieval and reranking
- holdout and five-fold Sentence Transformers training

## Credentials

The following environment variables are used where applicable:

```text
EURLEX_USERNAME
EURLEX_PASSWORD
GEMINI_API_KEY
GOOGLE_API_KEY
```

`GEMINI_API_KEY` and `GOOGLE_API_KEY` are alternatives. Do not commit credentials or a populated `.env` file.

## Pipeline overview

The scripts fall into four stages:

```text
Dataset construction:
01 -> 02 -> 03 -> 04 -> 05

Annotation and qrels:
05 -> 06 -> 07 -> 08 -> 09 -> 10 -> 11

Evaluation preparation:
11 -> 12 -> 13 -> 14

Fixed-model experiments:
15
15 -> 16

Supervised holdout experiments:
17

Supervised five-fold experiments:
18

```

Scripts 13 and 14 are alternative supervised evaluation protocols for runs that need train, development, and test partitions.

Scripts 15 and 16 are fixed-model zero-shot experiments.

## Script descriptions

### Dataset construction

1. `01_collect_data.py`
   - Collects EUR-Lex case metadata.
   - Fetches Cellar HTML or XHTML.
   - Extracts cleaned raw judgment text.
   - Reads `EURLEX_USERNAME` and `EURLEX_PASSWORD` from the environment.
   - Writes the canonical pickle files under `data/raw/`.

2. `02_build_dataset.py`
   - Generates structured case JSON.
   - Produces summaries, keywords, metadata, and segmentation anchors.
   - Restores complete case segments from the source judgment text.
   - Reads `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
   - Writes intermediate results under `data/intermediate/`.

3. `03_extract_citations.py`
   - Extracts CJEU-style ECLI citations.
   - Adds citation fields to the restored case JSON.
   - Produces the canonical corpus under `data/enriched_json/`.
   - Produces the citation graph under `data/citations/`.

4. `04_generate_facts_and_figures.py`
   - Generates corpus statistics, citation summaries, keyword summaries, and figures.
   - Writes outputs under `facts_and_figures/`.

5. `05_build_query_and_candidate_pool.py`
   - Selects the evaluation query cases.
   - Creates citation, sparse, dense, and random candidate pools.
   - Produces:
     - `data/shared/query_pool.jsonl`
     - `data/shared/candidate_pool.tsv`
     - `data/qrels/qrels_citations.tsv`
     - individual candidate pools under `data/pooling/`
   - May use local model downloads, embedding caches, or Gemini access depending on the enabled retrievers.

### Annotation and adjudication

6. `06_create_annotation_sheets.py`
   - Creates the empty master annotation sheet.
   - Creates balanced annotator assignment files.
   - Writes anonymized assignments under `data/annotations/`.
   - Uses neutral IDs such as `annotator_01`.

7. `07_recompile_annotation_sheet.py`
   - Reads completed files from:
     - `data/annotations/annotator_assignments/labeled/`
   - Recompiles annotator grades into:
     - `data/annotations/annotation_sheet_recompiled.csv`

8. `08_inter_annotator_agreement.py`
   - Calculates agreement metrics on repeated-assignment rows.
   - Produces pairwise metrics, confusion matrices, distributions, disagreement diagnostics, and figures.

9. `09_create_author_annotation_sheets.py`
   - Creates adjudication files for unresolved disagreements.
   - Retains the historical `author_*` field names for schema compatibility.
   - These field names indicate adjudication and do not contain an individual's identity.

10. `10_recompile_annotation_sheet_with_author.py`
    - Recompiles annotator labels and completed adjudication files.
    - Preserves annotator-only and final post-adjudication audit fields.
    - Produces:
      - `data/annotations/annotation_sheet_final.csv`

11. `11_create_human_qrels.py`
    - Converts the final annotation sheet into:
      - `data/qrels/qrels_human_binary.tsv`
      - `data/qrels/qrels_human_graded.tsv`
    - Maps grades 0 and 1 to binary nonrelevant and grades 2 and 3 to binary relevant.

### Evaluation preparation

12. `12_create_eval_views.py`
    - Creates all 24 evaluation-view folders:
      - 12 under `data/eval_views/human/`
      - 12 under `data/eval_views/citations/`
    - Query views:
      - `summary`
      - `facts`
      - `full`
    - Corpus views:
      - `summary`
      - `facts`
      - `reasoning`
      - `full`
    - Human folders contain:
      - `corpus.jsonl`
      - `queries.jsonl`
      - `qrels_selected.tsv`
      - `qrels_selected_graded.tsv`
    - Citation folders contain:
      - `corpus.jsonl`
      - `queries.jsonl`
      - `qrels_selected.tsv`

13. `13_create_supervised_split.py`
    - Creates the fixed query-level supervised holdout:
      - 70 training queries
      - 10 development queries
      - 20 test queries
    - Writes the canonical assignment file:
      - `data/splits/holdout_70train_10dev_20test/query_assignments.tsv`
    - Optionally materializes partitioned queries and qrels for all 24 evaluation views.
    - Does not divide the retrieval corpus itself.

14. `14_create_five_fold_cross_validation.py`
    - Assigns every query to exactly one of five outer folds.
    - Each fold contains 20 outer-test queries.
    - Writes the canonical assignment file:
      - `data/splits/five_fold_cross_validation/fold_assignments.tsv`
    - Optionally materializes fold-specific query and qrels files for all 24 evaluation views.
    - The common retrieval corpus remains shared across folds.

The assignment TSV files are the canonical split artifacts. Materialized split-specific queries and qrels are deterministic convenience outputs and can be regenerated.

### Zero-shot experiments

15. `15_basic_runner_script_zero_shot_retrieval.py`
    - Runs fixed-model dense retrieval over one selected evaluation view.
    - Evaluates all 100 queries when no EUR-LexRD labels were used for training or model selection.
    - Produces a TREC run file, aggregate metrics, per-query metrics, judged-at-k diagnostics, and a run configuration record.
    - The default adapter uses a Sentence Transformers bi-encoder.

16. `16_basic_runner_script_zero_shot_reranker.py`
    - Reranks a first-stage TREC run produced by script 15.
    - Uses qrels only after scoring, for evaluation.
    - Produces reranked scores, a TREC run file, aggregate metrics, per-query metrics, and a run configuration record.
    - The default adapter uses a Sentence Transformers `CrossEncoder`.

### Supervised experiments

17. `17_basic_training_script_holdout_biencoder.py`
    - Fine-tunes a Sentence Transformers bi-encoder using the released 70/10/20 query-level split.
    - Uses training qrels to form positive query-document pairs.
    - Uses the development partition for checkpoint selection.
    - Keeps the test partition untouched until the final configuration is frozen.
    - Reads:
      - `data/splits/holdout_70train_10dev_20test/query_assignments.tsv`
    - Writes outputs under:
      - `results/holdout_biencoder/`

18. `18_basic_training_script__five_fold_biencoder.py`
    - Fine-tunes five independent bi-encoders using the released outer-fold assignments.
    - Uses 20 queries as the outer test set in each fold.
    - Creates a deterministic inner development set from the 80 non-test queries.
    - Evaluates every query exactly once as outer test data in the final five-fold run.
    - Reads:
      - `data/splits/five_fold_cross_validation/fold_assignments.tsv`
    - Writes outputs under:
      - `results/five_fold_biencoder/`

## Evaluation protocol

### Fixed or zero-shot evaluation

Use all 100 queries when:

- the model was trained outside EUR-LexRD
- no EUR-LexRD qrels were used for fitting
- no EUR-LexRD qrels were used for hyperparameter selection
- the experimental configuration was fixed independently of the benchmark labels

Scripts 15 and 16 follow this protocol.

### Supervised holdout evaluation

Use script 13 followed by script 17 when developing one model with:

- 70 training queries
- 10 development queries
- 20 test queries

Do not use the test partition for checkpoint selection, model selection, or hyperparameter tuning.

### Five-fold cross-validation

Use script 14 followed by script 18 when reporting out-of-fold performance over all 100 queries.

Do not use an outer test fold for model selection. Model selection must occur only within the 80 non-test queries for that fold.

## Reproducibility

- All project-controlled random operations use seed `123`.
- Training runners record their effective configuration and output paths.
- Model names, model revisions, view selections, cutoffs, and split versions should be retained with reported run.