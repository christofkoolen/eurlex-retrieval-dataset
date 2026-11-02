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

\---

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

\---

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

* `case\_metadata`: identifiers and legal metadata
* `case\_summary`: generated subject matter, keywords, and summary
* `case\_segments`: section-aware judgment text
* `citations`: extracted outgoing case citations

The available judgment segments include:

* procedural introduction
* legal context
* facts and background
* reasoning
* decision on costs
* operative part

\---

## Repository and archival release

This repository contains the source code, benchmark files, documentation, sample enriched judgments, and reproducibility utilities and is the authoritative source of the dataset.

* **Research paper:** Christof Koolen. (2026). The EUR-Lex Retrieval Dataset: A Section-Aware and Human-Annotated Dataset for Long-Form Similar Case Retrieval. In *Proceedings of the Natural Legal Language Processing Workshop 2026* (pp. 101–118). Budapest, Hungary: Association for Computational Linguistics. https://doi.org/10.18653/2026.nllp-1.11
* **Dataset:** Christof Koolen. (2026). The EUR-Lex Retrieval Dataset (Version 1.0) \[Dataset]. KU Leuven RDR. https://doi.org/10.48804/TK4IER
* **GitHub repository:** `https://github.com/christofkoolen/eurlex-retrieval-dataset`

\---

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
│   │   └── citation\_graph.tsv
│   │
│   ├── enriched\_json/
│   │   └── enriched CJEU judgments in .json
│   │
│   ├── pooling/
│   │   ├── citation\_pool.tsv
│   │   ├── tfidf\_pool.tsv
│   │   ├── bm25\_pool.tsv
│   │   ├── minilm\_pool.tsv
│   │   ├── qwen\_pool.tsv
│   │   ├── gemini\_pool.tsv
│   │   ├── random\_negatives.tsv
│   │   └── pool\_config.json
│   │
│   ├── qrels/
│   │   ├── qrels\_citations.tsv
│   │   ├── qrels\_human\_binary.tsv
│   │   └── qrels\_human\_graded.tsv
│   │
│   ├── shared/
│   │   ├── query\_pool.jsonl
│   │   └── candidate\_pool.tsv
│   │
│   └── splits/
│       ├── holdout\_70train\_10dev\_20test/
│       │   ├── query\_assignments.tsv
│       │   ├── split\_config.json
│       │   └── split\_summary.txt
│       └── five\_fold\_cross\_validation/
│           ├── fold\_assignments.tsv
│           ├── fold\_config.json
│           └── cv\_summary.txt
│
├── eval\_views/
│   ├── human/
│   │   ├── q\_summary\_\_c\_summary/
│   │   ├── q\_summary\_\_c\_facts/
│   │   ├── q\_summary\_\_c\_reasoning/
│   │   ├── q\_summary\_\_c\_full/
│   │   ├── q\_facts\_\_c\_summary/
│   │   ├── q\_facts\_\_c\_facts/
│   │   ├── q\_facts\_\_c\_reasoning/
│   │   ├── q\_facts\_\_c\_full/
│   │   ├── q\_full\_\_c\_summary/
│   │   ├── q\_full\_\_c\_facts/
│   │   ├── q\_full\_\_c\_reasoning/
│   │   └── q\_full\_\_c\_full/
│   │
│   └── citations/
│       ├── q\_summary\_\_c\_summary/
│       ├── q\_summary\_\_c\_facts/
│       ├── q\_summary\_\_c\_reasoning/
│       ├── q\_summary\_\_c\_full/
│       ├── q\_facts\_\_c\_summary/
│       ├── q\_facts\_\_c\_facts/
│       ├── q\_facts\_\_c\_reasoning/
│       ├── q\_facts\_\_c\_full/
│       ├── q\_full\_\_c\_summary/
│       ├── q\_full\_\_c\_facts/
│       ├── q\_full\_\_c\_reasoning/
│       └── q\_full\_\_c\_full/
│
├── facts\_and\_figures/
│
├── code/
│   ├── pipeline\_common.py
│   ├── requirements.txt
│   ├── .env.example
│   ├── 01\_collect\_data.py
│   ├── 02\_build\_dataset.py
│   ├── 03\_extract\_citations.py
│   ├── 04\_generate\_facts\_and\_figures.py
│   ├── 05\_build\_query\_and\_candidate\_pool.py
│   ├── 06\_create\_annotation\_sheets.py
│   ├── 07\_recompile\_annotation\_sheet.py
│   ├── 08\_inter\_annotator\_agreement.py
│   ├── 09\_create\_author\_annotation\_sheets.py
│   ├── 10\_recompile\_annotation\_sheet\_with\_author.py
│   ├── 11\_create\_human\_qrels.py
│   ├── 12\_create\_eval\_views.py
│   ├── 13\_create\_supervised\_split.py
│   ├── 14\_create\_five\_fold\_cross\_validation.py
│   ├── 15\_basic\_runner\_script\_zero\_shot\_retrieval.py
│   ├── 16\_basic\_runner\_script\_zero\_shot\_reranker.py
│   ├── 17\_basic\_training\_script\_holdout\_biencoder.py
│   ├── 18\_basic\_training\_script\_five\_fold\_biencoder.py
│
├── 00\_CITATION.cff
├── 00\_LICENSE\_CODE.txt
├── 00\_LICENSE\_DATA.txt
└── 00\_README.md
```

\---

## Installation and use

The archival scripts are designed to be opened and run directly in your Python IDE.

Place `pipeline\_common.py` and all numbered scripts together in `code/`. Each script exposes editable configuration constants near the top of the file.

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
EURLEX\_USERNAME
EURLEX\_PASSWORD
GEMINI\_API\_KEY
GOOGLE\_API\_KEY
```

`GEMINI\_API\_KEY` and `GOOGLE\_API\_KEY` are alternatives. Do not commit a populated `.env` file, API key, password, or access token.

When automatic repository-root detection is not possible, set the following in the relevant script:

```python
from pathlib import Path

PROJECT\_ROOT\_OVERRIDE: Path | None = Path(
    r"C:\\path\\to\\EUR-LexRD"
)
```

Leave the override as `None` in the archival version.

\---

## Core data files

### Query pool

`data/shared/query\_pool.jsonl` contains the 100 query judgments and their available textual representations.

Each line is one JSON object. Query identifiers are stable across qrels, candidate-pool files, split assignments, and evaluation views.

### Candidate pool

`data/shared/candidate\_pool.tsv` contains the pooled query-candidate pairs submitted for human annotation.

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

`data/enriched\_json/` contains the enriched judgments in JSON format.

### Citation graph

`data/citations/citation\_graph.tsv` contains directed citation edges:

```text
source	target
ECLI:EU:C:2020:001	ECLI:EU:C:2015:100
```

\---

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
eval\_views/human/q\_summary\_\_c\_full/
├── corpus.jsonl
├── queries.jsonl
├── qrels\_selected.tsv
└── qrels\_selected\_graded.tsv
```

* `corpus.jsonl`: the candidate documents represented by means of the chosen candidate view
* `queries.jsonl`: the 100 query documents represented by means of the chosen query view
* `qrels\_selected.tsv`: binary human relevance judgments
* `qrels\_selected\_graded.tsv`: graded human relevance judgments

### Citation evaluation views

A citation evaluation-view folder contains:

```text
eval\_views/citations/q\_summary\_\_c\_full/
├── corpus.jsonl
├── queries.jsonl
└── qrels\_selected.tsv
```

* `corpus.jsonl`: the candidate documents represented by means of the chosen candidate view
* `queries.jsonl`: the 100 query documents represented by means of the chosen query view
* `qrels\_selected.tsv`: citation-based relevance judgments

### Naming convention

```text
q\_<query-view>\_\_c\_<candidate-view>
```

For example:

```text
q\_facts\_\_c\_full
```

means that query cases are represented by their factual sections and candidate cases by their full judgment text.

\---

## Relevance judgments

### Human graded qrels

`data/qrels/qrels\_human\_graded.tsv` uses the following ordinal scale:

|Grade|Interpretation|
|-:|-|
|0|Not relevant|
|1|Marginally relevant|
|2|Substantially relevant|
|3|Directly relevant|

### Human binary qrels

`data/qrels/qrels\_human\_binary.tsv` maps:

* grades 0 and 1 to non-relevant (`0`)
* grades 2 and 3 to relevant (`1`)

### Citation qrels

`data/qrels/qrels\_citations.tsv` treats an outgoing citation from a query judgment to a candidate judgment as a citation-derived relevance signal.

Citation relevance and human legal relevance represent different constructs and should be reported separately.

### Qrel format

All qrel files use the TREC-style four-column format:

```text
query\_id	iteration	document\_id	relevance
```

Example:

```text
ECLI:EU:C:2023:100	0	ECLI:EU:C:2019:200	3
```

\---

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
data/splits/holdout\_70train\_10dev\_20test/query\_assignments.tsv
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
data/splits/five\_fold\_cross\_validation/fold\_assignments.tsv
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
qrels\_selected.tsv
qrels\_selected\_graded.tsv
```

The candidate `corpus.jsonl` is not partitioned. All train, development, and test queries retrieve from the same candidate corpus for the selected candidate representation.

A query assigned to the test partition remains a test query in every human and citation view.

### Canonical versus derived split files

The canonical split artifacts are:

```text
data/splits/holdout\_70train\_10dev\_20test/query\_assignments.tsv
data/splits/five\_fold\_cross\_validation/fold\_assignments.tsv
```

Scripts 13 and 14 can additionally materialize filtered copies of the query, qrels, query-pool, and candidate-pool files. These are deterministic derivatives.

For the holdout protocol, optional materialized evaluation files are written under:

```text
data/splits/holdout\_70train\_10dev\_20test/eval\_views/
```

For five-fold cross-validation, optional materialized evaluation files are written under:

```text
data/splits/five\_fold\_cross\_validation/eval\_views/
```

By default, the candidate corpora are not duplicated. A relative `corpus\_source.txt` points back to the corresponding canonical corpus under `data/eval\_views/`.

\---

## Pipeline scripts

The scripts are numbered in pipeline order. Configuration is controlled through constants near the top of each file.

|Step|Stage|Script|Purpose|
|-:|-|-|-|
|01|Collection|`01\_collect\_data.py`|Collect CJEU metadata, Cellar HTML or XHTML, and cleaned source text from EUR-Lex services|
|02|Enrichment|`02\_build\_dataset.py`|Generate structured JSON with metadata, summaries, keywords, and section anchors, then restore complete judgment segments|
|03|Citations|`03\_extract\_citations.py`|Extract ECLI citations, enrich the JSON corpus, and construct the citation graph|
|04|Descriptive analysis|`04\_generate\_facts\_and\_figures.py`|Generate corpus statistics, citation summaries, keyword summaries, reports, and figures|
|05|Pool construction|`05\_build\_query\_and\_candidate\_pool.py`|Select queries and construct citation, lexical, dense, Gemini, Qwen, and random candidate pools|
|06|Annotation preparation|`06\_create\_annotation\_sheets.py`|Create the master annotation sheet and anonymized, balanced annotator assignments|
|07|Annotation recompilation|`07\_recompile\_annotation\_sheet.py`|Recompile completed annotator files and resolve annotator-only labels where possible|
|08|Agreement analysis|`08\_inter\_annotator\_agreement.py`|Calculate overall and pairwise agreement statistics and diagnostic outputs|
|09|Adjudication preparation|`09\_create\_author\_annotation\_sheets.py`|Create adjudication sheets for unresolved disagreements|
|10|Final adjudication|`10\_recompile\_annotation\_sheet\_with\_author.py`|Merge annotator labels and adjudication decisions into the final annotation sheet|
|11|Qrel generation|`11\_create\_human\_qrels.py`|Create binary and graded human qrels from the final annotation sheet|
|12|Evaluation views|`12\_create\_eval\_views.py`|Materialize all 24 unsplit evaluation-view folders under `data/eval\_views/`|
|13|Holdout preparation|`13\_create\_supervised\_split.py`|Create the fixed 70/10/20 query-level assignment and optional materialized split files|
|14|Cross-validation preparation|`14\_create\_five\_fold\_cross\_validation.py`|Create five balanced outer query folds and optional materialized fold files|
|15|Zero-shot retrieval|`15\_basic\_runner\_script\_zero\_shot\_retrieval.py`|Run fixed-model dense retrieval over all 100 queries and report standard metrics|
|16|Zero-shot reranking|`16\_basic\_runner\_script\_zero\_shot\_reranker.py`|Rerank a first-stage TREC run with a fixed pretrained cross-encoder and evaluate the reranked output|
|17|Holdout training|`17\_basic\_training\_script\_holdout\_biencoder.py`|Fine-tune and select a bi-encoder using the 70/10/20 holdout protocol|
|18|Five-fold training|`18\_basic\_training\_script\_five\_fold\_biencoder.py`|Fine-tune five bi-encoders with outer-fold isolation and deterministic inner development subsets|

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



\---

## Data formats

### Corpus and query JSONL

Each line contains one JSON object.

A retrieval-oriented record generally uses:

```json
{
  "doc\_id": "ECLI:EU:C:2023:100",
  "text": "Document or query text"
}
```

### Candidate pool TSV

The merged candidate-pool file includes fields such as:

```text
query\_id
candidate\_id
source\_primary
source\_all
score\_bm25
score\_gemini
score\_minilm
score\_qwen
score\_tfidf
```

### Split assignment TSV

```text
query\_id	split
ECLI:EU:C:2023:100	train
```

### Fold assignment TSV

```text
query\_id	fold
ECLI:EU:C:2023:100	1
```

\---

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

\---

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

\---

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

\---

## Citation

If you use EUR-LexRD in research, please cite **both** the archived dataset and the conference paper. The two publications have separate DOIs.

**Dataset (version 1.0, published July 28, 2026):**

Christof Koolen. (2026). The EUR-Lex Retrieval Dataset (Version 1.0) \[Dataset]. KU Leuven RDR. https://doi.org/10.48804/TK4IER

```bibtex
@dataset{koolen\_eurlexrd\_dataset\_2026,
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

\---

## Contact

**Christof Koolen**  
Centre for IT \& IP Law, KU Leuven  
Email: `christof.koolen@kuleuven.be`

