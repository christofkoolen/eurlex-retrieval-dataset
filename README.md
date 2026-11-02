# EUR-Lex Retrieval Dataset (EUR-LexRD)

EUR-LexRD is a legal information retrieval dataset for long-form similar case retrieval over English judgments of the Court of Justice of the European Union (CJEU).

The dataset combines:

* a corpus of 20,697 structured CJEU judgments
* 100 query judgments
* 12,365 human-annotated query-candidate pairs
* binary and graded human relevance judgments
* citation-derived relevance judgments
* section-aware query and candidate representations
* fixed supervised splits and five-fold cross-validation assignments
* scripts that document the collection, enrichment, annotation, and evaluation pipeline

---

## Table of contents

* [Dataset overview](#dataset-overview)
* [Repository and archival release](#repository-and-archival-release)
* [Repository structure](#repository-structure)
* [Installation and use](#installation-and-use)
* [Core data files](#core-data-files)
* [Evaluation views](#evaluation-views)
* [Relevance judgments](#relevance-judgments)
* [Evaluation protocols](#evaluation-protocols)
* [Pipeline scripts](#pipeline-scripts)
* [Data formats](#data-formats)
* [Reproducibility](#reproducibility)
* [Limitations](#limitations)
* [Licensing and reuse](#licensing-and-reuse)
* [Citation](#citation)
* [Contact](#contact)

---

## Dataset overview

|Property|Value|
|-|-|
|Domain|CJEU case law|
|Task|Long-form similar case and legal relevance retrieval|
|Jurisdiction|European Union|
|Language|English|
|Temporal coverage|1954 to 2025|
|Corpus size|20,697 judgments|
|Evaluation queries|100|
|Human-annotated query-candidate pairs|12,365|
|Human relevance scale|Four grades: 0, 1, 2, 3|
|qrel sources|Human binary, human graded, and citations|
|query views|summary, facts, and full judgment|
|candidate views|summary, facts, reasoning, and full judgment|
|evaluation combinations|24 folders: 12 human and 12 citation|
|Supervised holdout protocol|70 train, 10 development, 20 test queries|
|Cross-validation protocol|Five folds of 20 test queries each|

Each enriched judgment contains four high-level components:

* `case_metadata`: identifiers and legal metadata
* `case_summary`: generated subject matter, keywords, and summary
* `case_segments`: section-aware judgment text
* `citations`: extracted outgoing case citations

The available judgment segments include:

* procedural introduction
* legal context
* facts and background
* reasoning
* decision on costs
* operative part

---

## Repository and archival release

This repository contains the source code, benchmark files, documentation, sample enriched judgments, and reproducibility utilities and is the authoritative source of the dataset.

* **Research paper:** Christof Koolen. (2026). The EUR-Lex Retrieval Dataset: A Section-Aware and Human-Annotated Dataset for Long-Form Similar Case Retrieval. In *Proceedings of the Natural Legal Language Processing Workshop 2026* (pp. 101–118). Budapest, Hungary: Association for Computational Linguistics. https://doi.org/10.18653/2026.nllp-1.11
* **Dataset:** Christof Koolen. (2026). The EUR-Lex Retrieval Dataset (Version 1.0) [Dataset]. KU Leuven RDR. https://doi.org/10.48804/TK4IER
* **GitHub repository:** `https://github.com/christofkoolen/eurlex-retrieval-dataset`

---

## Repository structure

```text
.
├── data/
│   ├── annotations/
│   │   ├── annotation guidelines
│   │   ├── annotation sheets
│   │   ├── adjudicated labels
│   │   └── agreement summaries
│   │
│   ├── citations/
│   │   └── citation_graph.tsv
│   │
│   ├── enriched_json/
│   │   └── enriched CJEU judgments in .json
│   │
│   ├── pooling/
│   │   ├── citation_pool.tsv
│   │   ├── tfidf_pool.tsv
│   │   ├── bm25_pool.tsv
│   │   ├── minilm_pool.tsv
│   │   ├── qwen_pool.tsv
│   │   ├── gemini_pool.tsv
│   │   ├── random_negatives.tsv
│   │   └── pool_config.json
│   │
│   ├── qrels/
│   │   ├── qrels_citations.tsv
│   │   ├── qrels_human_binary.tsv
│   │   └── qrels_human_graded.tsv
│   │
│   ├── shared/
│   │   ├── query_pool.jsonl
│   │   └── candidate_pool.tsv
│   │
│   └── splits/
│       ├── holdout_70train_10dev_20test/
│       │   ├── query_assignments.tsv
│       │   ├── split_config.json
│       │   └── split_summary.txt
│       └── five_fold_cross_validation/
│           ├── fold_assignments.tsv
│           ├── fold_config.json
│           └── cv_summary.txt
│
├── eval_views/
│   ├── human/
│   │   ├── q_summary__c_summary/
│   │   ├── q_summary__c_facts/
│   │   ├── q_summary__c_reasoning/
│   │   ├── q_summary__c_full/
│   │   ├── q_facts__c_summary/
│   │   ├── q_facts__c_facts/
│   │   ├── q_facts__c_reasoning/
│   │   ├── q_facts__c_full/
│   │   ├── q_full__c_summary/
│   │   ├── q_full__c_facts/
│   │   ├── q_full__c_reasoning/
│   │   └── q_full__c_full/
│   │
│   └── citations/
│       ├── q_summary__c_summary/
│       ├── q_summary__c_facts/
│       ├── q_summary__c_reasoning/
│       ├── q_summary__c_full/
│       ├── q_facts__c_summary/
│       ├── q_facts__c_facts/
│       ├── q_facts__c_reasoning/
│       ├── q_facts__c_full/
│       ├── q_full__c_summary/
│       ├── q_full__c_facts/
│       ├── q_full__c_reasoning/
│       └── q_full__c_full/
│
├── facts_and_figures/
│
├── code/
│   ├── pipeline_common.py
│   ├── requirements.txt
│   ├── .env.example
│   ├── 01_collect_data.py
│   ├── 02_build_dataset.py
│   ├── 03_extract_citations.py
│   ├── 04_generate_facts_and_figures.py
│   ├── 05_build_query_and_candidate_pool.py
│   ├── 06_create_annotation_sheets.py
│   ├── 07_recompile_annotation_sheet.py
│   ├── 08_inter_annotator_agreement.py
│   ├── 09_create_author_annotation_sheets.py
│   ├── 10_recompile_annotation_sheet_with_author.py
│   ├── 11_create_human_qrels.py
│   ├── 12_create_eval_views.py
│   ├── 13_create_supervised_split.py
│   ├── 14_create_five_fold_cross_validation.py
│   ├── 15_basic_runner_script_zero_shot_retrieval.py
│   ├── 16_basic_runner_script_zero_shot_reranker.py
│   ├── 17_basic_training_script_holdout_biencoder.py
│   ├── 18_basic_training_script_five_fold_biencoder.py
│
├── 00_CITATION.cff
├── 00_LICENSE_CODE.txt
├── 00_LICENSE_DATA.txt
└── 00_README.md
```

---

## Installation and use

The archival scripts are designed to be opened and run directly in your Python IDE.

Place `pipeline_common.py` and all numbered scripts together in `code/`. Each script exposes editable configuration constants near the top of the file.

Python 3.10 or newer is recommended. Install the shared environment with:

```text
python -m pip install -r code/requirements.txt
```

For CUDA or another hardware-specific setup, install the PyTorch build for the target system before installing the remaining requirements.

The current requirements cover:

* EUR-Lex collection and HTML parsing
* Gemini-based generation and embedding
* statistics and graph analysis
* zero-shot dense retrieval and cross-encoder reranking
* supervised Sentence Transformers training

Credentials are read from environment variables where required:

```text
EURLEX_USERNAME
EURLEX_PASSWORD
GEMINI_API_KEY
GOOGLE_API_KEY
```

`GEMINI_API_KEY` and `GOOGLE_API_KEY` are alternatives. Do not commit a populated `.env` file, API key, password, or access token.

When automatic repository-root detection is not possible, set the following in the relevant script:

```python
from pathlib import Path

PROJECT_ROOT_OVERRIDE: Path | None = Path(
    r"C:\path\to\EUR-LexRD"
)
```

Leave the override as `None` in the archival version.

---

## Core data files

### Query pool

`data/shared/query_pool.jsonl` contains the 100 query judgments and their available textual representations.

Each line is one JSON object. Query identifiers are stable across qrels, candidate-pool files, split assignments, and evaluation views.

### Candidate pool

`data/shared/candidate_pool.tsv` contains the pooled query-candidate pairs submitted for human annotation.

Candidates were obtained from several seeding methods:

* outgoing citations
* TF-IDF
* BM25
* all-MiniLM-L6-v2
* Qwen3-Embedding-8B
* gemini-embedding-001
* random negatives

The candidate pool records the source or sources through which a candidate entered the pool. It should be retained when interpreting the human judgments because the judgments are incomplete outside the pooled candidates.

### Enriched judgments

`data/enriched_json/` contains the enriched judgments in JSON format.

### Citation graph

`data/citations/citation_graph.tsv` contains directed citation edges:

```text
source	target
ECLI:EU:C:2020:001	ECLI:EU:C:2015:100
```

---

## Evaluation views

EUR-LexRD supports section-aware retrieval by separating query and candidate representations.

### Query representations

* `summary`
* `facts`
* `full`

### Candidate representations

* `summary`
* `facts`
* `reasoning`
* `full`

This produces 12 query-candidate view combinations for each qrel source.

### Human evaluation views

A human evaluation-view folder contains:

```text
eval_views/human/q_summary__c_full/
├── corpus.jsonl
├── queries.jsonl
├── qrels_selected.tsv
└── qrels_selected_graded.tsv
```

* `corpus.jsonl`: the candidate documents represented by means of the chosen candidate view
* `queries.jsonl`: the 100 query documents represented by means of the chosen query view
* `qrels_selected.tsv`: binary human relevance judgments
* `qrels_selected_graded.tsv`: graded human relevance judgments

### Citation evaluation views

A citation evaluation-view folder contains:

```text
eval_views/citations/q_summary__c_full/
├── corpus.jsonl
├── queries.jsonl
└── qrels_selected.tsv
```

* `corpus.jsonl`: the candidate documents represented by means of the chosen candidate view
* `queries.jsonl`: the 100 query documents represented by means of the chosen query view
* `qrels_selected.tsv`: citation-based relevance judgments

### Naming convention

```text
q_<query-view>__c_<candidate-view>
```

For example:

```text
q_facts__c_full
```

means that query cases are represented by their factual sections and candidate cases by their full judgment text.

---

## Relevance judgments

### Human graded qrels

`data/qrels/qrels_human_graded.tsv` uses the following ordinal scale:

|Grade|Interpretation|
|-:|-|
|0|Not relevant|
|1|Marginally relevant|
|2|Substantially relevant|
|3|Directly relevant|

### Human binary qrels

`data/qrels/qrels_human_binary.tsv` maps:

* grades 0 and 1 to non-relevant (`0`)
* grades 2 and 3 to relevant (`1`)

### Citation qrels

`data/qrels/qrels_citations.tsv` treats an outgoing citation from a query judgment to a candidate judgment as a citation-derived relevance signal.

Citation relevance and human legal relevance represent different constructs and should be reported separately.

### Qrel format

All qrel files use the TREC-style four-column format:

```text
query_id	iteration	document_id	relevance
```

Example:

```text
ECLI:EU:C:2023:100	0	ECLI:EU:C:2019:200	3
```

---

## Evaluation protocols

EUR-LexRD distinguishes between fixed-model evaluation and supervised model development.

### Protocol A: evaluation of fixed models

Use all 100 queries when the method is fixed independently of EUR-LexRD and the dataset's qrels are not used for training, tuning, checkpoint selection, prompt selection, or model selection.

Examples include:

* BM25 with predetermined parameters
* TF-IDF
* an off-the-shelf embedding model
* a pretrained reranker used without dataset-specific adaptation
* a fixed chunking strategy selected independently of the benchmark

For this protocol, the complete 100 query collection functions as the evaluation set.

### Protocol B: supervised 70/10/20 split

Use the released supervised split when EUR-LexRD relevance judgments influence training, model selection, checkpoint selection, or hyperparameter tuning.

Canonical assignment file:

```text
data/splits/holdout_70train_10dev_20test/query_assignments.tsv
```

The split contains:

|Partition|Number of queries|Purpose|
|-|-:|-|
|Train|70|Model fitting|
|Development|10|Model and checkpoint selection|
|Test|20|Final evaluation|

The split is performed at query level. Every qrel and candidate-pool row associated with a query remains in the same partition.

Script 13 creates the assignments and script 17 consumes them for supervised bi-encoder training.

### Protocol C: five-fold cross-validation

Canonical fold-assignment file:

```text
data/splits/five_fold_cross_validation/fold_assignments.tsv
```

Each query belongs to exactly one outer fold. For each run:

* 80 queries form the outer non-test partition;
* 20 queries form the outer test partition;
* an inner development subset is created only from the 80 non-test queries when model selection is required;
* each query is evaluated as an outer-test query exactly once.

The outer test fold must not be used for hyperparameter or checkpoint selection. Script 14 creates the outer folds and script 18 creates a deterministic inner development subset for each supervised training run.

### How splits affect evaluation views

Split assignments apply to these files inside every evaluation-view folder:

```text
queries.jsonl
qrels_selected.tsv
qrels_selected_graded.tsv
```

The candidate `corpus.jsonl` is not partitioned. All train, development, and test queries retrieve from the same candidate corpus for the selected candidate representation.

A query assigned to the test partition remains a test query in every human and citation view.

### Canonical versus derived split files

The canonical split artifacts are:

```text
data/splits/holdout_70train_10dev_20test/query_assignments.tsv
data/splits/five_fold_cross_validation/fold_assignments.tsv
```

Scripts 13 and 14 can additionally materialize filtered copies of the query, qrels, query-pool, and candidate-pool files. These are deterministic derivatives.

For the holdout protocol, optional materialized evaluation files are written under:

```text
data/splits/holdout_70train_10dev_20test/eval_views/
```

For five-fold cross-validation, optional materialized evaluation files are written under:

```text
data/splits/five_fold_cross_validation/eval_views/
```

By default, the candidate corpora are not duplicated. A relative `corpus_source.txt` points back to the corresponding canonical corpus under `data/eval_views/`.

---

## Pipeline scripts

The scripts are numbered in pipeline order. Configuration is controlled through constants near the top of each file.

|Step|Stage|Script|Purpose|
|-:|-|-|-|
|01|Collection|`01_collect_data.py`|Collect CJEU metadata, Cellar HTML or XHTML, and cleaned source text from EUR-Lex services|
|02|Enrichment|`02_build_dataset.py`|Generate structured JSON with metadata, summaries, keywords, and section anchors, then restore complete judgment segments|
|03|Citations|`03_extract_citations.py`|Extract ECLI citations, enrich the JSON corpus, and construct the citation graph|
|04|Descriptive analysis|`04_generate_facts_and_figures.py`|Generate corpus statistics, citation summaries, keyword summaries, reports, and figures|
|05|Pool construction|`05_build_query_and_candidate_pool.py`|Select queries and construct citation, lexical, dense, Gemini, Qwen, and random candidate pools|
|06|Annotation preparation|`06_create_annotation_sheets.py`|Create the master annotation sheet and anonymized, balanced annotator assignments|
|07|Annotation recompilation|`07_recompile_annotation_sheet.py`|Recompile completed annotator files and resolve annotator-only labels where possible|
|08|Agreement analysis|`08_inter_annotator_agreement.py`|Calculate overall and pairwise agreement statistics and diagnostic outputs|
|09|Adjudication preparation|`09_create_author_annotation_sheets.py`|Create adjudication sheets for unresolved disagreements|
|10|Final adjudication|`10_recompile_annotation_sheet_with_author.py`|Merge annotator labels and adjudication decisions into the final annotation sheet|
|11|Qrel generation|`11_create_human_qrels.py`|Create binary and graded human qrels from the final annotation sheet|
|12|Evaluation views|`12_create_eval_views.py`|Materialize all 24 unsplit evaluation-view folders under `data/eval_views/`|
|13|Holdout preparation|`13_create_supervised_split.py`|Create the fixed 70/10/20 query-level assignment and optional materialized split files|
|14|Cross-validation preparation|`14_create_five_fold_cross_validation.py`|Create five balanced outer query folds and optional materialized fold files|
|15|Zero-shot retrieval|`15_basic_runner_script_zero_shot_retrieval.py`|Run fixed-model dense retrieval over all 100 queries and report standard metrics|
|16|Zero-shot reranking|`16_basic_runner_script_zero_shot_reranker.py`|Rerank a first-stage TREC run with a fixed pretrained cross-encoder and evaluate the reranked output|
|17|Holdout training|`17_basic_training_script_holdout_biencoder.py`|Fine-tune and select a bi-encoder using the 70/10/20 holdout protocol|
|18|Five-fold training|`18_basic_training_script_five_fold_biencoder.py`|Fine-tune five bi-encoders with outer-fold isolation and deterministic inner development subsets|

### Workflow dependencies

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

Scripts 13 and 14 are alternative supervised evaluation protocols. Both assignment files may be generated and released. Scripts 15 and 16 do not require split files because fixed models are evaluated over all 100 queries.



---

## Data formats

### Corpus and query JSONL

Each line contains one JSON object.

A retrieval-oriented record generally uses:

```json
{
  "doc_id": "ECLI:EU:C:2023:100",
  "text": "Document or query text"
}
```

### Candidate pool TSV

The merged candidate-pool file includes fields such as:

```text
query_id
candidate_id
source_primary
source_all
score_bm25
score_gemini
score_minilm
score_qwen
score_tfidf
```

### Split assignment TSV

```text
query_id	split
ECLI:EU:C:2023:100	train
```

### Fold assignment TSV

```text
query_id	fold
ECLI:EU:C:2023:100	1
```

---

## Reproducibility

For reproducible experiments, report:

* EUR-LexRD dataset DOI
* query view
* candidate view
* candidate universe
* evaluation protocol
* retrieval or reranking model used
* truncation or chunking settings
* similarity or scoring function
* hyperparameters (where applicable)
* random seed
* evaluation metrics and cutoffs
* treatment of unjudged documents

A run description should be sufficiently precise to identify a single evaluation configuration, for example:

```text
Dataset: EUR-LexRD v1.0
Qrels: human graded
Query view: summary
Candidate view: full
Protocol: fixed-model evaluation over all 100 queries
Method: all-MiniLM-L6-v2
Top k: 100
Metrics: P@5, P@10, MAP@10, nDCG@10, nDCG@20, nDCG@30
```

The experiment runners write TREC run files, per-query results, aggregate metrics, and effective run-configuration records under `results/`.

Because the human judgments arise from a pooled candidate set, every result should explain how unjudged retrieved documents are handled.

---

## Limitations

Users should be mindful of the following limitations:

* The benchmark contains only 100 query judgments
* Human relevance judgments cover pooled candidates; there are no relevance labels available for all query-candidate pairs
* Candidate pooling likely favors systems that have contributed to the pool
* Although the inter-annotator agreement metrics hint at a fairly good degree of agreement among annotators, most query-candidate pairs have been judged by one annotator
* The dataset contains English CJEU judgments and, thus, might not generalize to other languages or jurisdictions.
* Section boundaries can contain errors
* Unrestricted candidate retrieval includes judgments issued after the query judgment (unless a temporal filter is applied)

See the accompanying paper for a more extensive discussion.

---

## Licensing and reuse

The release contains several kinds of material that are governed by different terms:

|Component|Applicable terms|
|-|-|
|CJEU judgment text|EUR-Lex reuse terms and applicable legal notices|
|EUR-Lex metadata|EUR-Lex reuse terms and applicable legal notices|
|Human relevance annotations|CC BY 4.0|
|Generated summaries and derived segments|CC BY 4.0|
|Citation graph and benchmark metadata|CC BY 4.0|
|Source code|MIT License|
|Documentation|CC BY 4.0|

---

## Citation

If you use EUR-LexRD in research, please cite **both** the archived dataset and the conference paper. The two publications have separate DOIs.

**Dataset (version 1.0, published July 28, 2026):**

Christof Koolen. (2026). The EUR-Lex Retrieval Dataset (Version 1.0) [Dataset]. KU Leuven RDR. https://doi.org/10.48804/TK4IER

```bibtex
@dataset{koolen-2026-eurlexrd-dataset,
  author    = {Koolen, Christof},
  title     = {The EUR-Lex Retrieval Dataset},
  year      = {2026},
  version   = {1.0},
  publisher = {KU Leuven RDR},
  doi       = {10.48804/TK4IER},
  url       = {https://doi.org/10.48804/TK4IER}
}
```

**Conference paper (NLLP 2026):**

Christof Koolen. (2026). The EUR-Lex Retrieval Dataset: A Section-Aware and Human-Annotated Dataset for Long-Form Similar Case Retrieval. In *Proceedings of the Natural Legal Language Processing Workshop 2026* (pp. 101–118). Budapest, Hungary: Association for Computational Linguistics. https://doi.org/10.18653/2026.nllp-1.11

```bibtex
@inproceedings{koolen-2026-eur,
  title     = {The {EUR}-Lex Retrieval Dataset: A Section-Aware and Human-Annotated Dataset for Long-Form Similar Case Retrieval},
  author    = {Koolen, Christof},
  booktitle = {Proceedings of the Natural Legal Language Processing Workshop 2026},
  month     = oct,
  year      = {2026},
  address   = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics},
  pages     = {101--118},
  doi       = {10.18653/2026.nllp-1.11},
  url       = {https://aclanthology.org/2026.nllp-1.11/}
}
```

The machine-readable references are provided in [`CITATION.cff`](CITATION.cff).

---

## Contact

**Christof Koolen**  
Centre for IT & IP Law, KU Leuven  
Email: `christof.koolen@kuleuven.be`

