# api.jobl.ai

FastAPI service that processes raw job postings: cleans HTML descriptions, extracts expiry dates and application emails, categorizes jobs into 26 industry categories, generates embedding vectors for semantic search, and extracts skills from multilingual job descriptions.

Includes a background sync worker that pulls jobs from MySQL source databases into PostgreSQL.

## Production Architecture

```
Scraper flow (every new job):
  Scraper → POST /v1/process → category + clean description + embedding + skills
          → stores in scraper DB → flows to job boards

Job board flow (direct submissions):
  Resume/job submitted → POST /v1/embed → 1024-dim vector
          → stored in board's MariaDB VECTOR(1024) column

Resume extraction (paying candidates):
  Job board → OpenAI API (gpt-5-nano) with locked extraction prompt
            → 18 structured fields (not self-hosted)
```

### Production Models (30GB server, 4 workers)

| Component | RAM/worker | Purpose |
|---|---|---|
| Bi-encoder (ONNX) | ~2.5 GB | Generates 1024-dim embedding vectors for semantic search |
| LightGBM categorizer | ~50 MB | Classifies jobs into 26 industry categories |
| Skill taxonomy (ESCO) | ~50 MB | Matches 13,900 skills in 10 languages from Postgres |
| **Total per worker** | **~2.6 GB** | |
| **4 workers + Postgres + OS** | **~14.5 GB** | **~15 GB headroom** |

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [API Endpoint](#api-endpoint)
- [Processing Pipeline](#processing-pipeline)
- [Graceful Degradation](#graceful-degradation)
- [Database](#database)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running Locally](#running-locally)
- [Migrations](#migrations)
- [Categorizer Training](#categorizer-training)
  - [Hyperparameter Tuning](#2a-tune-hyperparameters-optional)
- [Manual Labelling](#manual-labelling)
- [Sync Worker](#sync-worker)
- [Resume XML Import](#resume-xml-import)
- [Production Deployment](#production-deployment)

---

## Architecture

```
POST /v1/process
      │
      ├─ Language detection (langid)
      │
      ├─ Description cleaning (BeautifulSoup + custom rules)
      │     ├─ HTML normalization (strip layout tags, attributes)
      │     ├─ Expiry date extraction (EN + FR patterns)
      │     └─ Email masking → ***email_hidden***
      │
      └─ EN/FR only ─────────────────────────────────────────┐
            ├─ Title normalization (seq2seq transformer)      │
            ├─ Email extraction                               │
            └─ Categorization (LightGBM, 26 categories)      │
                                                              │
      Non-EN/FR: title unchanged, original_category passed ──┘
```

Both the normalizer and categorizer are **optional**. The API starts and serves requests without them; those fields return `null`.

---

## Project Structure

```
api.jobl.ai/
├── app/
│   ├── main.py                  # FastAPI app, lifespan (model loading)
│   ├── core/config.py           # Settings (pydantic-settings, .env)
│   ├── db/
│   │   ├── base.py              # SQLAlchemy DeclarativeBase
│   │   └── session.py           # Engine + get_db() dependency
│   ├── models/
│   │   ├── category.py          # categories table
│   │   ├── category_map.py      # category_map table
│   │   ├── country.py           # countries lookup table
│   │   ├── job.py               # jobs table
│   │   ├── resume.py            # resumes table (training data, populated offline)
│   │   ├── source_country.py    # source MySQL DB → country mapping
│   │   └── sync_state.py        # sync cursor (last synced job ID per DB)
│   ├── services/
│   │   ├── language.py          # langid-based language detection
│   │   ├── cleaner.py           # HTML cleaner + expiry extractor
│   │   ├── normalizer.py        # seq2seq title normalizer
│   │   └── categorizer.py       # LightGBM job categorizer
│   └── api/v1/
│       ├── health.py            # GET /health
│       └── process.py           # POST /process
├── sync/
│   ├── config.py                # SyncSettings (SOURCE_DB_*, DATABASE_URL)
│   ├── worker.py                # SyncWorker: MySQL → PostgreSQL
│   ├── main.py                  # jobl-sync entry point
│   ├── language_backfill.py     # jobl-sync-language-backfill entry point
│   ├── resume_xml_worker.py     # XML feed → resumes table importer
│   └── resume_xml_import.py     # jobl-sync-resumes entry point
├── alembic/
│   └── versions/                # 22 migrations (linear chain)
├── labelling/
│   ├── main.py                    # FastAPI labelling web app (port 8002)
│   └── templates/index.html       # Single-page labelling UI
├── scripts/
│   ├── extract_labelling_data.py           # Populate job_labelling table (balanced per class)
│   ├── evaluate_cleaner_extractor.py       # Re-run cleaner/extractor on verified rows
│   ├── generate_training_data.py           # Training CSV from job_labelling table
│   ├── train_categorizer.py               # Train LightGBM, save .pkl artifact
│   └── tune_categorizer.py                # Optuna hyperparameter search for LightGBM
├── sql/
│   ├── seed_categories.sql      # 26 category rows
│   ├── seed_countries.sql
│   ├── seed_source_countries.sql
│   └── category_map.csv         # source category → local category_id (1,315 rows)
├── deploy/
│   ├── jobl-api.service         # systemd unit
│   └── nginx/api.jobl.ai        # nginx site config
└── pyproject.toml
```

---

## API Endpoint

### `POST /v1/process`

#### Request

```json
{
  "title": "Senior Software Engineer - Full Time #ABC123",
  "description": "<div><p>Apply by April 30 2026. Send resume to jobs@example.com</p></div>",
  "original_category": "Technology"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `title` | string (1–512 chars) | yes | Raw job title from source |
| `description` | string | yes | Raw HTML job description |
| `original_category` | string | no | Source category label; passed through for non-EN/FR jobs |

#### Response

```json
{
  "title_normalized": "Senior Software Engineer",
  "description_clean": "<p>Apply by April 30 2026. Send resume to ***email_hidden***</p>",
  "application_email": "jobs@example.com",
  "expiry_date": "2026-04-30",
  "category": {
    "id": 4,
    "title": "Information Technology",
    "confidence": 0.872
  }
}
```

| Field | Type | Description |
|---|---|---|
| `title_normalized` | string | Cleaned/normalized title; unchanged for non-EN/FR |
| `description_clean` | string | Cleaned HTML; email replaced with `***email_hidden***` |
| `application_email` | string \| null | Extracted email, or null |
| `expiry_date` | string \| null | ISO date (YYYY-MM-DD), or null if not found / non-EN/FR |
| `category` | object \| null | `{id, title, confidence}` — null if model not loaded |

`category.confidence` is the model's softmax probability for the predicted class (0–1). It is `null` for non-EN/FR jobs where the model is not run.

For non-EN/FR jobs, `category` is `{"id": null, "title": "<original_category>", "confidence": null}` if `original_category` was provided.

---

## Processing Pipeline

### 1. Language Detection

Language is auto-detected from the job content — it is **not** an input parameter.

`app/services/language.py` uses `langid` to classify the title + description blob. Supported codes: `en`, `fr`, `de`, `es`, `pt`, `it`, `nl`, `da`, `uk`, `gr`.

Fallback chain:
1. `langid` detection (if result is in the allowlist)
2. Country-code lookup (e.g. US → `en`, DE → `de`, FR → `fr`)
3. Source DB name heuristic (e.g. `americas` → `es`)

For mixed-language countries (CA, CH, SG), langid result is used directly if in the allowlist, otherwise `None`.

### 2. Description Cleaning

`app/services/cleaner.py` applies an HTML normalization pipeline:

1. URL-decode injected markup
2. Strip JSON-style backslash escapes (`\"` → `"`)
3. Strip malformed sentinel blocks
4. Unwrap layout tags (`div`, `span`, `table`, `font`, `center`, `section`, …)
5. Strip all HTML attributes
6. Remove `<br>` inside bold tags
7. Unwrap nested bold tags
8. Merge consecutive bold headers
9. Promote standalone bold to `<h3>`
10. Wrap bare `<li>` elements (no parent `<ul>`) in `<ul>` before block-splitting
11. Collapse `<br>`-delimited content into `<p>` blocks
12. Promote leading bold in paragraphs to section headers
13. Fix invalid `<h3>` nested inside `<p>`
14. Unwrap double-nested `<p>`
15. Wrap bare text nodes in `<p>`
16. Remove duplicate consecutive `<h3>` headings
17. Remove UI navigation artifacts (see below)
18. Drop empty blocks
19. Enforce allowlist: `p`, `h2`, `h3`, `h4`, `ul`, `ol`, `li`, `strong`, `em`, `a`

**UI artifact removal** strips job-board chrome that leaks into descriptions — standalone paragraphs or headings matching patterns such as:

- Navigation: `Previous job Next job`, `Back to search results`
- Portal prompts: `Are you a [Company] Employee?`, `Open My Career Portal`, `Current Employees should apply …`
- Email widget: `Email This Job To`, `Your email is on its way…`, `Email has not been sent`
- Print notice: `Please print/save this job description …`
- Separator lines, job metadata labels, EEO codes

**Expiry extraction** (`extract_expiry` / `extract_expiry_raw`) uses two paths:

**Path 1 — `<span class="hl-date">` tags**: the raw HTML is searched first. Each tagged date is classified as either an application deadline or a non-deadline date (start date, contract end, etc.) by inspecting the preceding sibling text:
- Classified as **non-deadline** when preceded by: `start date`, `available from`, `begin on`, `end on`, `term start/end`, `fixed to` (fixed-term contract end), `date de début`, `du` (French range start), `entrée en fonction`, …
- All other tagged dates are classified as the application deadline.

**Path 2 — plain-text scan**: applied when no hl-date tag yields a deadline. Scans line by line for:
- Label-based: `Application deadline:`, `Closing date:`, `Closing`, `Closing on:`, `Close date:`, `Posting End Date:`, `Job Posting End Date:`, `Apply by date:`, `Open until:`, `Prior to`, `Application window close`, `Fin de l'affichage`, and French equivalents (`Date limite pour postuler:`, `Date de clôture:`, `Date d'affichage:`, `Période d'inscription:`, `Avant le`, …)
- Inline prose: `posting will close on …`, `posting will expire on or before …`, `apply by April 1, 2026`, `submit a résumé by …`
- French date ranges: `Du 2026-02-26 au 2026-03-15` (the `au` date is extracted as the deadline)
- Date formats: `YYYY-MM-DD`, `MM/DD/YYYY`, `DD/MM/YYYY`, `Month D, YYYY`, `D Month YYYY`, `D mois AAAA`, `M/D/YY` (2-digit year interpreted as 20xx)

**HTML formatting guards** prevent non-deadline bold text from being incorrectly split into separate paragraphs:
- `_is_section_header` skips strings containing `@` (emails), `#` followed by digits (reference codes like `#11-26`), or matching phone number patterns.
- `_MID_SENTENCE_ENDS_RE` blocks mid-sentence bold tags (preceded by articles/prepositions) from triggering a paragraph flush.
- Bold inside `<li>` is never promoted to `<h3>`.

`extract_expiry` returns an ISO date for future deadlines, `"expired"` for past dates, or `None`. `extract_expiry_raw` always returns the date regardless of whether it has passed (used when building training data).

### 3. Title Normalization (EN/FR only)

`app/services/normalizer.py` uses a seq2seq transformer (`AutoModelForSeq2SeqLM`) loaded from `MODEL_DIR`.

**Pre-processing** strips noise before feeding the model:
- Salary markers (`$100k`, `€50/hr`)
- Job codes (`#ENGR-123`, `(1234-5678)`)
- Employment type keywords (`full-time`, `permanent`, `casual`)
- Filler phrases (`apply now`, `hiring now`, `multiple openings`)

**Post-processing**:
- Title-case output if model returns lowercase
- Normalize separators to ` - ` format
- Restore legal gender suffixes (`m/w/d`, `m/f`)

When `MODEL_DIR` is not set, a rules-only path applies the same pre/post steps without the model inference.

### 4. Email Extraction and Masking (EN/FR only)

A regex (`[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}`) scans the description for candidate emails. Extraction uses two strategies, tried in order:

1. **`<span class="hl-email">` tags** — the raw HTML is searched first. The tag's parent element text plus the immediately preceding sibling element text are checked against an exclusion list.
2. **Plain-text fallback** — scans the cleaned description plain text; requires apply/submit/CV keywords within 250 chars and absence of exclusion keywords within the same window (checked backward-only so boilerplate after the email does not suppress it).

**Exclusion keywords** (email is skipped when these appear in context):
- Accommodation / accessibility language (`accommodation`, `disability`, `handicap`, `accessibility`, …)
- Fraud / data-privacy notices (`fraud`, `scam`, `GDPR`, `CCPA`, `privacy notice`, …)
- EEO boilerplate (`equal employment opportunity`, `institutional equity`)
- Explicit rejection: `not accept[ing] applications via/by/through email`

**Local-part exclusions** suppress the email regardless of context: `noreply`, `donotreply`, `support`, `helpdesk`, `fraud`, `scam`, `eeo`, …

The first accepted email is returned as `application_email`; all occurrences are replaced with `***email_hidden***` in the HTML output.

### 5. Categorization (EN/FR only)

`app/services/categorizer.py` wraps a pickled artifact (TF-IDF vectorizer + native LightGBM booster) trained to classify into 26 categories.

Input text: `f"{title} {title} {title} {description_plain}"` — the raw title is repeated 3× to amplify its signal relative to the full description.

The raw (un-normalized) `title` from the request is used so that training and inference operate on identical input distributions.

Returns `{"id": N, "title": "...", "confidence": 0.xx}` where `confidence` is the softmax probability of the winning class. Returns `None` when `CATEGORIZER_MODEL_PATH` is not set.

---

## Graceful Degradation

| Condition | Behaviour |
|---|---|
| `MODEL_DIR` not set | Rules-only title normalization for EN/FR |
| `CATEGORIZER_MODEL_PATH` not set | `category: null` for EN/FR |
| Either model fails to load at startup | App starts; affected field is `null`; error logged |
| Non-EN/FR language detected | Title unchanged; description cleaned; `original_category` passed through if provided |
| No email found in description | `application_email: null` |
| Expiry date not found or past | `expiry_date: null` |

---

## Database

PostgreSQL. Schema managed by Alembic.

### Tables

| Table | Description |
|---|---|
| `jobs` | Synced job postings (title, description, location, salary, AI fields, `category`) |
| `categories` | 26 industry categories (id, title) |
| `category_map` | Maps source category strings → local `category_id` |
| `countries` | Country lookup (code, name, alternate_names, language_codes) |
| `source_countries` | MySQL source DB → country/currency/config mapping |
| `sync_state` | Sync cursor: last synced job ID per (source_db, destination) pair |
| `resumes` | Raw resume postings for model training (`source_website` + `external_id`, location, salary, `language_code`); populated by `jobl-sync-resumes`. **Not anonymized** — descriptions may contain names, emails, phone numbers, addresses; PII scrubbing happens downstream in the ML pipeline's data-prep step, not at ingest |

**`resumes` columns:** `id`, `source_website`, `external_id`, `title`, `description`, `city_title`, `region_title`, `country_code`, `salary`, `salary_period`, `salary_currency`, `contract`, `published_at`, `is_remote`, `language_code`. Unique on `(source_website, external_id)`.

### Migrations

22 migrations in a linear chain. Current head: `c8f2a1b3d4e5` (add_resumes_table).

---

## Setup

**Requirements**: Python 3.12, PostgreSQL 17+, Ubuntu 24.04 LTS

**System prerequisites**:

```bash
# Python build tools
sudo apt install -y python3.12-venv python3.12-dev

# pgvector extension (required for embedding columns)
sudo apt install -y postgresql-17-pgvector
# Then enable it in the database (must be done as postgres superuser):
sudo -u postgres psql -d jobl -c "CREATE EXTENSION IF NOT EXISTS vector"
```

```bash
cd api.jobl.ai
python3 -m venv .venv
    source .venv/bin/activate

# Production install
pip install -e .

# Development install (adds pytest)
pip install -e ".[dev]"

# Copy and edit environment
cp .env.example .env
$EDITOR .env

# Run migrations
alembic upgrade head

# Seed reference data
psql $DATABASE_URL < sql/seed_categories.sql
psql $DATABASE_URL < sql/seed_countries.sql
psql $DATABASE_URL < sql/seed_source_countries.sql
psql $DATABASE_URL -c "\copy category_map (original_category, category_id) FROM 'sql/category_map.csv' CSV HEADER"

# Seed skill taxonomy (download ESCO CSVs from https://esco.ec.europa.eu/en/use-esco/download)
# Select: Version=v1.2.1, Content=Classification, Format=CSV, one file per language
python sql/seed_skills.py --dir data/esco/
```

### Dependency scopes

| Scope | Command | Includes |
|---|---|---|
| Production | `pip install -e .` | API, sync worker, training scripts, test endpoint script |
| Development | `pip install -e ".[dev]"` | Production + pytest |
| Scripts | `pip install -e ".[scripts]"` | Production + optuna (required for `tune_categorizer.py`) |

---

## Configuration

All settings are read from environment variables (or `.env`).

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/jobl` | PostgreSQL connection URL |
| `APP_NAME` | `"Jobl API"` | OpenAPI title |
| `APP_VERSION` | `"0.1.0"` | OpenAPI version |
| `API_PREFIX` | `"/v1"` | URL prefix for all routes |
| `MODEL_DIR` | — | Path to seq2seq normalizer model directory |
| `NUM_BEAMS` | `4` | Beam search width |
| `MAX_NEW_TOKENS` | `32` | Max tokens generated per title |
| `MAX_INPUT_LENGTH` | `128` | Input truncation length |
| `CATEGORIZER_MODEL_PATH` | — | Path to `categorizer.pkl` artifact |
| `BIENCODER_MODEL_PATH` | — | Path to bi-encoder ONNX directory (e.g. `models/biencoder-onnx`) |
| `EXTRACTOR_MODEL_PATH` | — | Path to extractor GGUF file (e.g. `models/extractor-gguf/model-Q4_K_M.gguf`) |
| `VERIFIED_LABELLING` | `false` | Labelling UI: show only `verified=true` rows when `true` |

**Sync worker only:**

| Variable | Default | Description |
|---|---|---|
| `SOURCE_DB_HOST` | — | MySQL host |
| `SOURCE_DB_PORT` | `3306` | MySQL port |
| `SOURCE_DB_USER` | — | MySQL username |
| `SOURCE_DB_PASSWORD` | — | MySQL password |
| `SOURCE_DB_DRIVER` | `mysql+pymysql` | SQLAlchemy driver string |
| `SOURCE_DB_SSL_DISABLED` | `true` | Disable MySQL SSL |
| `EXPORT_DESTINATION` | `jobl.ai` | Destination tag written to `sync_state` |
| `SYNC_BATCH_SIZE` | `1000` | Jobs per sync batch |

**Resume XML import only:**

| Variable | Default | Description |
|---|---|---|
| `RESUME_XML_FEEDS_DIR` | — | Directory of per-board XML feeds (required for import) |
| `RESUME_XML_LOOKBACK_HOURS` | `24` | Only process `*.xml` files modified within this window |
| `RESUME_XML_BATCH_SIZE` | `500` | Rows per DB insert batch |

---

## Running Locally

```bash
# Start API (degraded mode — no ML models)
uvicorn app.main:app --reload

# Test EN job (full pipeline, rules-only normalization)
curl -X POST http://localhost:8001/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Software Engineer - Full Time #ABC123",
    "description": "<p>Apply by April 30, 2026. Email jobs@example.com</p>",
    "original_category": "Technology"
  }'

# Test FR job (expiry auto-detected)
curl -X POST http://localhost:8001/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Ingénieur logiciel",
    "description": "<p>Date limite pour postuler: 2 avril 2026</p>"
  }'

# Test non-EN/FR (description cleanup only, category passed through)
curl -X POST http://localhost:8001/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Softwareentwickler",
    "description": "<div><span>Wir suchen.</span></div>",
    "original_category": "IT"
  }'
```

### With ML models

Set `MODEL_DIR` and `CATEGORIZER_MODEL_PATH` in `.env`, then restart. The API will load both models during startup and use them for EN/FR jobs.

---

## Migrations

```bash
# Apply all pending migrations
alembic upgrade head

# Roll back one step
alembic downgrade -1

# Show migration history
alembic history

# Generate a new migration after model changes
alembic revision --autogenerate -m "describe change"
# Then review and trim the generated file before committing
```

> **Note**: Autogenerate detects all live DB differences. Always review the generated file and remove any unintended changes before applying.

---

## Categorizer Training

### 1. Generate training data

Exports all rows from the `job_labelling` table (all 26 categories) to a training CSV. The raw `title` column is used (not the normalized title) so that training and inference see identical input distributions.

```bash
python scripts/generate_training_data.py --output data/
```

Output:
- `data/categorizer_training.csv` — columns: `title`, `original_category`, `description_plaintext`, `category_id`

```
Wrote 4800 rows to data/categorizer_training.csv
```

`DATABASE_URL` is read from `.env` automatically.

### 2. Train the model

```bash
python scripts/train_categorizer.py \
  --data data/categorizer_training.csv \
  --output models/categorizer.pkl
```

Pipeline: `TfidfVectorizer(max_features=20_000, ngram_range=(1,2), min_df=5, sublinear_tf=True)` → native `lgb.train()` with multiclass objective and early stopping.

The title is repeated 3× in the input text (`"{title} {title} {title} {description}"`) to amplify its signal relative to the full description, matching inference-time behaviour. The full description is used — no truncation.

Artifact format:
```python
{
    "tfidf": TfidfVectorizer,                     # fitted
    "booster": lgb.Booster,                       # trained native booster
    "id_to_category": {int: {"id": int, "title": str}},
    "num_classes": int,
}
```

Training prints validation loss every 10 rounds, stops automatically when it stops improving, then prints a per-class classification report:

```
[10]    val's multi_logloss: 0.761
[20]    val's multi_logloss: 0.610
Early stopping, best iteration is:
[23]    val's multi_logloss: 0.608

Evaluating on validation set ...
                                    precision    recall  f1-score   support

  Manufacturing & Industrial Prod.      0.923     0.900     0.911        40
                        Automotive      0.971     0.971     0.971        35
  ...
                         macro avg      0.924     0.918     0.921       480
                      weighted avg      0.927     0.921     0.924       480
```

### 2a. Tune hyperparameters (optional)

Run before training to find better LightGBM params via Optuna. Requires the `scripts` dependency scope.

```bash
pip install -e ".[scripts]"

python scripts/tune_categorizer.py \
  --data data/categorizer_training.csv \
  --n-trials 50
```

Uses the same text vectorization and train/val split as `train_categorizer.py`. Each trial varies `num_leaves`, `learning_rate`, `min_data_in_leaf`, `lambda_l1`, `lambda_l2`, `feature_fraction`, `bagging_fraction`, and `bagging_freq`.

Output:
```
Running 50 trials ...

Best validation accuracy: 0.9312
Best params (paste into train_categorizer.py):
    params = {
        "objective": "multiclass",
        "num_class": 26,
        "metric": "multi_logloss",
        "num_leaves": 127,
        "learning_rate": 0.047312,
        ...
    }
```

Paste the printed `params` dict into `train_categorizer.py`, then retrain.

### 3. Activate the model

Set `CATEGORIZER_MODEL_PATH=models/categorizer.pkl` in `.env` and restart the API.

### 4. Evaluate against live data

With the API running, test categorization quality against random jobs from the database:

```bash
python scripts/test_process_endpoint.py --limit 100
python scripts/test_process_endpoint.py --limit 500 --url http://localhost:8001
```

Each row prints: job ID, latency, original title → normalized title, and ✓/✗ for email/expiry/category. Final summary shows hit rates and average latency.

```
[ 1/100] id=12345   142ms | 'Software Engineer - Full Time' -> 'Software Engineer' | email=✗ expiry=✗ category=Information Technology
...
─────────────────────────────────────
Results (100 jobs)
─────────────────────────────────────
  OK:            98 / 100
  Errors:        2
  With email:    12 (12.0%)
  With expiry:   8 (8.0%)
  With category: 95 (95.0%)
  Avg latency:   138ms
─────────────────────────────────────
```

---

## Manual Labelling

For higher-quality training data, jobs can be manually reviewed and corrected via a web UI. Labels are stored in the `job_labelling` table and exported with `generate_training_data_labelled.py`.

### 1. Populate the labelling table

Extracts a balanced sample of jobs (up to `--limit` per category, classes 1–26), cleans descriptions, and auto-assigns categories via `category_map` or heuristics:

```bash
python scripts/extract_labelling_data.py --limit 200 --countries=us,ca
```

Output per class:
```
  [ 4] inserted 198 (total for class: 198/200)
  [ 8] inserted 200 (total for class: 200/200)
  ...
Done. Total inserted: 4712
```

Re-running is safe — already-present jobs are skipped. To top up a class, run again; only the deficit is filled.

### 2. Review in the labelling UI

```bash
.venv/bin/python labelling/main.py
```

Opens at `http://<host>:8002/`. Features:
- Category dropdown in the top bar — jobs load dynamically per category
- Two-column table: original description (left) | cleaned description + category assignment (right)
- Dates highlighted in yellow, emails in blue in both columns
- Category dropdown per job — change saves instantly via Ajax, no submit button
- Auto-assigned / Reviewed badge tracks review status
- **⎘ Copy** button in each column header copies all visible descriptions (separated by `---`) to clipboard
- **Verify / ⏳ Pending** button (top and bottom of each right-hand cell) — toggles `verified = true/false` in the DB

**Evaluate-cycle mode** — set `VERIFIED_LABELLING=true` in `.env` to show only `verified = true` rows.
Use this after running `scripts/evaluate_cleaner_extractor.py` to review updated cleaning results.
Once satisfied, click **⏳ Pending** to reset `verified → false` for that job and move on.

### 3. Export training data

```bash
python scripts/generate_training_data.py --output data/
```

Exports all rows from `job_labelling` (all 26 categories) to `data/categorizer_training.csv`. Train the model with `train_categorizer.py` as usual.

---

## Sync Worker

Pulls jobs from MySQL source databases into the PostgreSQL `jobs` table. Tracks progress in `sync_state` (one row per source DB + destination pair).

```bash
# Sync all configured source databases
jobl-sync

# Sync specific databases only
jobl-sync --db americas --db australia

# Sync specific countries only
jobl-sync --country=us,ca

# Re-sync from the beginning (resets cursor, skips export_ai filter)
# Use this to backfill jobs.category after adding the category column
jobl-sync --resync
jobl-sync --resync --country=us,ca

# Backfill language_code for existing jobs without one
jobl-sync-language-backfill
jobl-sync-language-backfill --batch-size 5000 --limit 50000 --from-id 0
jobl-sync-language-backfill --overwrite   # recompute even if already set
```

Source databases are configured in the `source_countries` table (columns: `db_name`, `country_code`, `currency`, `config`).

`--resync` behaviour:
- Resets `last_job_id` to 0 for all matched sources (jobs are re-fetched from the beginning)
- Removes the `export_ai` deduplication filter so already-exported jobs are processed again
- Uses `COALESCE(new_value, existing_value)` for `category`, so existing values are never overwritten with NULL

---

## Resume XML Import

Imports resume XML feeds from `RESUME_XML_FEEDS_DIR` into the `resumes` table. Each job board uploads a file like `au.workus.org.xml`.

### Feed format

```xml
<?xml version="1.0" encoding="UTF-8"?>
<resumes>
    <website><![CDATA[au.workus.org]]></website>
    <resume>
        <id><![CDATA[12345]]></id>
        <country_code><![CDATA[AU]]></country_code>
        <city_title><![CDATA[Melbourne]]></city_title>
        <region_title><![CDATA[Victoria]]></region_title>
        <position><![CDATA[Environmental Professional]]></position>
        <description><![CDATA[...]]></description>
        <salary><![CDATA[80000.00]]></salary>
        <currency><![CDATA[AUD]]></currency>
        <salary_period><![CDATA[year]]></salary_period>
        <contract_code><![CDATA[full_time]]></contract_code>
        <remote><![CDATA[0]]></remote>
        <created_at><![CDATA[2026-06-05 14:17:19]]></created_at>
    </resume>
</resumes>
```

Field mapping:

| XML element | `resumes` column |
|---|---|
| `<website>` | `source_website` |
| `<id>` | `external_id` |
| `<position>` | `title` |
| `<description>` | `description` |
| `<city_title>` | `city_title` |
| `<region_title>` | `region_title` |
| `<country_code>` | `country_code` |
| `<salary>` | `salary` |
| `<salary_period>` | `salary_period` |
| `<currency>` | `salary_currency` |
| `<contract_code>` | `contract` |
| `<created_at>` | `published_at` |
| `<remote>` | `is_remote` (`1` / `true` → remote) |

Each `<resume>` must include `<id>` (numeric), non-empty `<position>`, and non-empty `<description>`. Rows missing any of these are skipped. `language_code` is detected on import via `app/services/language.py`.

Only resumes not already present (`source_website` + `external_id`) are inserted. By default, only `*.xml` files modified in the last 24 hours are scanned.

```bash
# Import feeds changed in the last 24 hours (default)
jobl-sync-resumes

# Initial backfill — process every XML file in the directory
jobl-sync-resumes --all-files

# Import one board only
jobl-sync-resumes --all-files --file=au.workus.org.xml

# Custom lookback window and batch size
jobl-sync-resumes --lookback-hours=48 --batch-size=1000
```

Example cron (hourly, after boards upload feeds):

```bash
15 * * * * cd /home/webadmin/Jobl/api.jobl.ai && .venv/bin/jobl-sync-resumes >> /home/webadmin/Jobl/logs/jobl-sync-resumes.log 2>&1
```

---

## Production Deployment

### 1. Install application

```bash
sudo mkdir -p /home/webadmin/Jobl/api.jobl.ai
# Copy repo to /home/webadmin/Jobl/api.jobl.ai
cd /home/webadmin/Jobl/api.jobl.ai
python3 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env
$EDITOR .env   # set DATABASE_URL, model paths, etc.
```

### 2. Run database migrations and seed data

```bash
.venv/bin/alembic upgrade head
psql $DATABASE_URL < sql/seed_categories.sql
psql $DATABASE_URL < sql/seed_countries.sql
psql $DATABASE_URL < sql/seed_source_countries.sql
psql $DATABASE_URL -c "\copy category_map (original_category, category_id) FROM 'sql/category_map.csv' CSV HEADER"
```

### 3. Configure systemd

```bash
sudo cp deploy/jobl-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable jobl-api
sudo systemctl start jobl-api
sudo systemctl status jobl-api
```

The service runs as `webadmin` on `127.0.0.1:8001` with 2 Uvicorn workers.

### 4. Configure nginx + TLS

```bash
sudo cp deploy/nginx/api.jobl.ai /etc/nginx/sites-available/api.jobl.ai
sudo ln -s /etc/nginx/sites-available/api.jobl.ai /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Obtain Let's Encrypt certificate
sudo certbot --nginx -d api.jobl.ai
sudo systemctl reload nginx
```

### 5. Sync workers (cron or systemd timer)

```bash
mkdir -p /home/webadmin/Jobl/logs

# Example cron: run every 12 hours (00:00 and 12:00)
10 */12 * * * cd /home/webadmin/Jobl/api.jobl.ai && .venv/bin/jobl-sync >> /home/webadmin/Jobl/logs/jobl-sync.log 2>&1

# Example cron: import resume XML feeds hourly
40 * * * * cd /home/webadmin/Jobl/api.jobl.ai && .venv/bin/jobl-sync-resumes >> /home/webadmin/Jobl/logs/jobl-sync-resumes.log 2>&1
```

### Logs

```bash
# API logs
sudo journalctl -u jobl-api -f

# Sync logs (cron)
tail -f /home/webadmin/Jobl/logs/jobl-sync.log
tail -f /home/webadmin/Jobl/logs/jobl-sync-resumes.log
```

---

## ML Pipeline — Phase 0 Data Prep

Offline scripts for preparing training data for the matching pipeline (bi-encoder, reranker, extractor). These live in `ml/scripts/` and operate on JSONL files in `ml/data/`. They do not affect the production API.

### Setup

```bash
# Install ML dependencies (adds datasketch, spacy, fasttext-wheel)
pip install -e ".[ml]"

# Download spaCy NER models (for PII scrubbing)
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
python -m spacy download pt_core_news_sm
python -m spacy download fr_core_news_sm

# Download fastText language ID model (for language detection)
wget -P ml/models/ https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

### Step 1 — Export Corpus

Exports the `jobs` and `resumes` tables from PostgreSQL into JSONL. Reuses the existing SQLAlchemy models (`app/models/job.py`, `app/models/resume.py`). Streams rows with a server-side cursor to avoid loading ~1.5M jobs into memory.

```bash
python -u ml/scripts/export_corpus.py
python -u ml/scripts/export_corpus.py --table jobs --limit 1000   # test run
python -u ml/scripts/export_corpus.py --sample 20                 # spot-check
```

| Output | Description |
|---|---|
| `ml/data/raw/jobs.jsonl` | ~1.5M job postings with all fields (title, description, country, salary, category, etc.) |
| `ml/data/raw/resumes.jsonl` | ~225K resumes with all fields (title, description, location, salary, etc.) |

### Step 2 — Deduplicate Jobs

Three-pass deduplication of the jobs corpus: exact duplicates dropped by SHA-256 content hash, near-duplicates flagged via MinHash+LSH (Jaccard similarity over 5-word shingles). Near-duplicates are flagged (`is_near_duplicate`, `near_duplicate_of`), not dropped — they feed the `is_duplicate_signal` field in the extractor's training schema.

```bash
python -u ml/scripts/dedup.py
python -u ml/scripts/dedup.py --skip-near-dup     # exact dedup only, faster
python -u ml/scripts/dedup.py --threshold 0.85    # stricter near-dup match
```

| Output | Description |
|---|---|
| `ml/data/interim/jobs.jsonl` | Deduped jobs with `is_near_duplicate` and `near_duplicate_of` fields |
| `ml/data/interim/near_dup_sample.jsonl` | Up to 50 near-dup pairs for manual sanity-check |

**Gate:** log dedup ratio; sanity-check flagged near-dup pairs manually.

### Step 3 — PII-Scrub Resumes

Strips personally identifiable information (names, emails, phones, addresses, URLs) from resume descriptions before any training data is constructed. Uses spaCy NER (language-specific models for EN/ES/PT/FR) plus regex patterns for structured PII, with a validation filter to suppress common false positives.

Handles ALL CAPS names (common resume header convention) via NER on title-cased ALL CAPS lines plus a header-line heuristic.

```bash
python -u ml/scripts/pii_scrub.py --sample 200   # gate review first
python -u ml/scripts/pii_scrub.py                 # full corpus
```

| Output | Description |
|---|---|
| `ml/data/interim/resumes.jsonl` | PII-scrubbed resumes with `[NAME]`, `[EMAIL]`, `[PHONE]`, `[URL]`, `[ADDRESS]` placeholders |
| `ml/data/interim/pii_scrub_sample.jsonl` | Before/after pairs for manual gate review (with `--sample`) |

**Gate (hard blocker):** manually review 200 scrubbed samples before proceeding. Confirm no PII leakage remains; accept that some over-tagging of non-PII content is tolerable.

### Step 4 — Language Detection

Runs fastText `lid.176.bin` (176 languages) over all jobs and resumes, adding `language_fasttext` and `language_confidence` fields. The existing `language_code` field (from the sync pipeline's langid, covering ~10 languages) is preserved unchanged — it's the baseline for comparison.

```bash
python -u ml/scripts/lang_detect.py
python -u ml/scripts/lang_detect.py --table jobs     # jobs only
python -u ml/scripts/lang_detect.py --table resumes   # resumes only
```

Updates `ml/data/interim/jobs.jsonl` and `ml/data/interim/resumes.jsonl` in place.

**Gate:** check language distribution and confidence histogram in the output. Flag low-confidence (<0.5) records — these get tagged as `unknown` in the locale prefix during training.

### Step 4b — Classify Resumes by Type

Classifies resumes as `full_cv`, `cover_letter`, or `minimal` based on text length and structural heuristics (section headers, bullet points, line count). No ML needed — pure regex + length checks.

Many resumes in the corpus are actually short cover letters addressed to specific employers (a consequence of job board submission flows). This classification enables downstream steps to handle them differently:
- **Bi-encoder/reranker training:** use all types (the model should match based on whatever text is available)
- **Extractor training:** weight/stratify by type so the teacher doesn't waste labeling budget on minimal-text resumes that produce mostly-null extractions
- **Validation:** ensure a representative mix of types

```bash
python -u ml/scripts/classify_resumes.py
```

Updates `ml/data/interim/resumes.jsonl` in place, adding a `resume_type` field (`full_cv` / `cover_letter` / `minimal`).

### Step 5 — Stratified Train/Val Split

Splits jobs and resumes by time (not random) into train and validation pools. The oldest 90% go to train, the most recent 10% to val. If the val set covers fewer than 15 countries, records from underrepresented countries are augmented from the train set.

The val pools are where Phase 1's teacher labeling draws validation pairs from; the actual (resume, job, score) triples are constructed during teacher labeling, not here.

```bash
python -u ml/scripts/make_splits.py
python -u ml/scripts/make_splits.py --val-fraction 0.15 --min-countries 20
```

| Output | Description |
|---|---|
| `ml/data/splits/train_pool_jobs.jsonl` | Older 90% of deduped+lang-tagged jobs |
| `ml/data/splits/val_pool_jobs.jsonl` | Most recent 10% of jobs |
| `ml/data/splits/train_pool_resumes.jsonl` | Older 90% of PII-scrubbed+lang-tagged resumes |
| `ml/data/splits/val_pool_resumes.jsonl` | Most recent 10% of resumes |

**Gate:** val set spans ≥15 distinct country codes (script prints GATE PASSED/FAILED explicitly).

### Step 6 — Validate Extraction Prompt

Tests the Component 3 (Extractor) prompt against ~20 sample job postings via the OpenAI API to confirm the output schema is valid before committing to it for Phase 1 teacher labeling on the A100.

The prompt extracts 18 structured fields from raw job postings: `normalized_title` (English-normalized), `original_title`, `seniority`, `occupation_category` (from the existing 26 categories), `employment_type`, `contract_type`, `work_mode`, `location_city`, `location_country`, `language`, `salary_present/min/max/currency`, `skills` (English-normalized array), `experience_years_min`, `is_expired_signal`, `is_duplicate_signal`.

All structured fields (title, skills) are normalized to English for cross-lingual matching consistency. Original titles are preserved in the `original_title` field.

```bash
pip install -e ".[ml]"                                          # adds openai
export OPENAI_API_KEY="sk-..."
python -u ml/scripts/validate_extraction_prompt.py
python -u ml/scripts/validate_extraction_prompt.py --limit 30 --debug    # verbose
python -u ml/scripts/validate_extraction_prompt.py --model gpt-4o        # different model
```

Validates each response against the schema: all 18 required fields present, enum values conform, `occupation_category` is one of the 26 valid categories, `location_country` is ISO 3166-1 alpha-2, `skills` is an array.

**Gate:** all samples produce valid extractions (script prints GATE PASSED/FAILED). Once passed, the `SYSTEM_PROMPT` constant in this file is the locked prompt template for Phase 1's `teacher_extract.py`.

### Step 7 — Download Base Model Weights

Downloads all four base models from HuggingFace into `ml/models/base/` and verifies each loads correctly. Run on the CPU server before the A100 rental starts so Day 1 isn't spent downloading ~50GB over the GPU instance's network.

| Model | HuggingFace repo | Size | Purpose |
|---|---|---|---|
| Harrier-OSS-v1-0.6B | `microsoft/harrier-oss-v1-0.6b` | ~1.2 GB | Bi-encoder base |
| XLM-RoBERTa-large | `FacebookAI/xlm-roberta-large` | ~2.2 GB | Reranker base |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | ~15 GB | Extractor base (LoRA) |
| Qwen2.5-72B-Instruct-AWQ | `Qwen/Qwen2.5-72B-Instruct-AWQ` | ~41 GB | Teacher model (Day 1-2) |

```bash
pip install -e ".[ml]"
hf auth login                                           # enter HuggingFace token

python -u ml/scripts/download_models.py                 # all four models
python -u ml/scripts/download_models.py --skip-teacher  # skip the 41GB teacher
python -u ml/scripts/download_models.py --model harrier # single model
python -u ml/scripts/download_models.py --verify-only   # check existing downloads
```

**Gate:** all models download and verify (tokenizer loads, config parses, embedding model produces vectors). Script prints GATE PASSED/FAILED.

### Step 8 — A100 Training Environment

The A100 GPU instance needs additional Python packages beyond the CPU server's `.[ml]` extras — specifically `vllm` (for serving the 72B teacher), `peft`/`trl` (for LoRA fine-tuning), and `datasets`. These have CUDA-specific build requirements that should be tested before the GPU clock starts.

A pinned requirements file is provided at `ml/requirements-training.txt`. On the A100 instance:

```bash
# Create a fresh environment on the A100
python3 -m venv .venv-training
source .venv-training/bin/activate

# Install training dependencies (requires CUDA 12.x)
pip install -r ml/requirements-training.txt

# Smoke-test: confirm vLLM can see the GPU
python -c "import vllm; print('vLLM OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"
python -c "from peft import LoraConfig; print('PEFT OK')"
python -c "from trl import SFTTrainer; print('TRL OK')"
```

If testing on the CPU server (no GPU), install everything except `vllm` to confirm the Python dependencies resolve without conflicts:

```bash
pip install sentence-transformers>=3 transformers peft trl datasets onnxruntime
```

**Gate:** all imports succeed, `torch.cuda.is_available()` returns True on the A100.

### Step 9 — Artifact Push Script

Pushes ML pipeline artifacts to remote storage via `scp` after each training phase on the A100. Designed for incremental backup — run after each phase so instance death mid-window loses at most one day of work.

```bash
# Configure (add to .bashrc or export before use)
export PUSH_REMOTE_HOST=webadmin@your-server.com
export PUSH_REMOTE_BASE=/home/webadmin/Jobl/ml-artifacts

# Dry-run to verify paths and SSH connectivity
./ml/scripts/push_artifacts.sh teacher_labels --dry-run

# Push after each phase
./ml/scripts/push_artifacts.sh teacher_labels    # after Day 1-2
./ml/scripts/push_artifacts.sh biencoder         # after Day 3-4
./ml/scripts/push_artifacts.sh reranker          # after Day 5-6
./ml/scripts/push_artifacts.sh extractor         # after Day 7-8
./ml/scripts/push_artifacts.sh corpus_outputs    # after Day 9
./ml/scripts/push_artifacts.sh final             # Day 10: push everything
```

Requires SSH key auth to the remote host. Each phase creates the remote directory structure and transfers the relevant files. The `final` phase pushes all models, data, splits, configs, and scripts.

**Gate:** dry-run completes without SSH errors; verify at least one test file is accessible on the remote host.

**Companion: `pull_data.sh`** — the reverse direction. Run on the A100 at the start of Day 1 to pull training data from the CPU server:

```bash
export PULL_REMOTE_HOST=webadmin@cpu-server.com
export PULL_REMOTE_BASE=/home/webadmin/Jobl/api.jobl.ai
./ml/scripts/pull_data.sh              # pulls interim data, splits, scripts, configs
python -u ml/scripts/download_models.py   # then download models from HuggingFace directly
```

### Pipeline Order

Steps must run in this order since each reads the previous step's output:

```
1.  export_corpus.py                →  ml/data/raw/{jobs,resumes}.jsonl
2.  dedup.py                        →  ml/data/interim/jobs.jsonl
3.  pii_scrub.py                    →  ml/data/interim/resumes.jsonl
4.  lang_detect.py                  →  updates interim/{jobs,resumes}.jsonl in place
4b. classify_resumes.py             →  updates interim/resumes.jsonl in place (adds resume_type)
5.  make_splits.py                  →  ml/data/splits/{train,val}_pool_{jobs,resumes}.jsonl
6.  validate_extraction_prompt.py   →  tests prompt via OpenAI API (no file output)
7.  download_models.py              →  ml/models/base/{harrier,reranker,extractor,teacher}/
8.  A100 environment setup          →  manual (see instructions above)
9.  push_artifacts.sh <phase>       →  scp artifacts to remote storage after each training phase
```

All data files are gitignored (`data/` pattern matches at any depth). Model files in `ml/models/` are also gitignored.

---

## ML Pipeline — Phase 1: Teacher Labeling (A100 GPU)

Phase 1 runs on the rented A100 80GB SXM instance. The teacher model (Qwen2.5-72B-Instruct-AWQ) generates training labels for the three production models.

### Step 1 — A100 Instance Setup

See [BACKUP.md](../BACKUP.md) (in the workspace root, outside the repo) for the full boot sequence: venv restore, data pull, model download. Summary:

```bash
sudo apt install -y python3.12-venv python3.12-dev
# restore venv, pull scripts, pull data, pull/download models — see BACKUP.md
source /home/webadmin/.venv-training/bin/activate
```

### Step 2 — Launch Teacher Model

Uses **Qwen2.5-32B-Instruct-AWQ** (not the 72B originally planned) — the 32B model generates ~6x faster on a single A100 with minimal quality loss for structured extraction. The AWQ quantization uses ~17GB VRAM, leaving ~57GB for KV cache which supports high-concurrency batching (32 workers).

The 72B model (`ml/models/base/teacher`) is available as a fallback if extraction quality needs improvement, but at ~34s/record vs ~4s/record for 32B, it's impractical for 20K+ extractions on a single GPU.

```bash
# Download 32B model (if not already on the instance):
hf download Qwen/Qwen2.5-32B-Instruct-AWQ --local-dir ml/models/base/teacher

# Launch vLLM in the background (runs continuously during Day 1-2):
nohup vllm serve ml/models/base/teacher \
  --quantization awq \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8000 > /tmp/vllm.log 2>&1 &

# Monitor startup (wait for CUDA graph capture to finish):
tail -f /tmp/vllm.log
```

Verify it's running:

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
# Monitor GPU during inference:
watch -n 2 nvidia-smi
```

### Step 3 — Teacher Extraction

Run the locked extraction prompt (validated in Phase 0 step 6) against job postings. Produces structured JSON fields (normalized_title, seniority, occupation_category, skills, salary, etc.) for each job.

Target: **20K stratified extractions** (diversity across countries/languages matters more than volume for LoRA fine-tuning; 20K is sufficient for all three production models). The `--stratified` flag samples proportionally across country+language buckets with a minimum of 10 per bucket, ensuring underrepresented countries still get coverage.

```bash
# Test run first (foreground, watch output):
python -u ml/scripts/teacher_extract.py --model ml/models/base/teacher --workers 32 --limit 100

# Full run (background, ~13 hours at ~25 records/min):
nohup python -u ml/scripts/teacher_extract.py \
  --model ml/models/base/teacher \
  --workers 32 \
  --limit 20000 \
  --stratified > /tmp/teacher_extract.log 2>&1 &
tail -f /tmp/teacher_extract.log

# Resume after interruption:
nohup python -u ml/scripts/teacher_extract.py \
  --model ml/models/base/teacher \
  --workers 32 \
  --limit 20000 \
  --stratified \
  --resume > /tmp/teacher_extract.log 2>&1 &
```

Verify diversity after completion:

```bash
cat ml/data/teacher_labels/extractions.jsonl | python3 -c "
import json, sys, collections
langs = collections.Counter()
countries = collections.Counter()
for line in sys.stdin:
    r = json.loads(line)
    langs[r.get('language_code','')] += 1
    countries[r.get('country_code','')] += 1
print('Top languages:', langs.most_common(10))
print('Top countries:', countries.most_common(10))
print('Total countries:', len(countries))
"
```

| Output | Description |
|---|---|
| `ml/data/teacher_labels/extractions.jsonl` | Job ID + 18-field structured extraction per record |

### Step 4 — Teacher Matching

Generate scored (resume, job) pairs and hard negatives. Candidate pairs are pre-filtered by country + language + title keyword overlap, then scored by the teacher on a 4-point scale (0.0 / 0.3 / 0.7 / 1.0). Positive pairs (score ≥ 0.7) get hard negatives generated — plausible-but-wrong jobs with specific mismatch dimensions.

Target: 50K scored pairs → triples.

```bash
# Test run first (foreground):
python -u ml/scripts/teacher_match.py --model ml/models/base/teacher --workers 32 --limit 1000

# Full run (background):
nohup python -u ml/scripts/teacher_match.py \
  --model ml/models/base/teacher \
  --workers 32 > /tmp/teacher_match.log 2>&1 &
tail -f /tmp/teacher_match.log

# Resume after interruption:
nohup python -u ml/scripts/teacher_match.py \
  --model ml/models/base/teacher \
  --workers 32 \
  --resume > /tmp/teacher_match.log 2>&1 &
```

| Output | Description |
|---|---|
| `ml/data/teacher_labels/match_scores.jsonl` | All scored (resume, job) pairs with relevance score + reasoning |
| `ml/data/teacher_labels/match_triples.jsonl` | Positive pairs + hard negatives for training |

### Step 5 — Backup & Spot-Check

Run after each day's work — do not let a day's labels live only on the instance.

```bash
# Push artifacts to CPU server
export PUSH_REMOTE_HOST=webadmin@46.224.133.10
export PUSH_REMOTE_BASE=/home/webadmin/Jobl/ml-artifacts
./ml/scripts/push_artifacts.sh teacher_labels

# Spot-check: review ~50 extractions and ~50 match judgments
# Focus on underrepresented languages (Ukrainian, Greek, Indonesian/Malay)
head -50 ml/data/teacher_labels/extractions.jsonl | python3 -m json.tool | less
head -50 ml/data/teacher_labels/match_scores.jsonl | python3 -m json.tool | less
```

**Gate:** spot-check ~50 extractions and ~50 match judgments by hand. If quality is bad for a specific language, fix the prompt or flag for manual augmentation before moving to Phase 2.

---

## ML Pipeline — Phase 2: Bi-Encoder Training (A100 GPU)

Fine-tunes the Harrier-OSS-v1-0.6B embedding model on teacher-generated match triples. The trained bi-encoder retrieves the top-100 candidate jobs/resumes for a given query via ANN search in production.

### Step 1 — Build Training Data

Converts teacher match outputs (scored pairs + hard negatives) into (anchor, positive, negative) triples for Sentence Transformers training.

```bash
python -u ml/scripts/build_biencoder_data.py
```

| Output | Description |
|---|---|
| `ml/data/splits/biencoder_train.jsonl` | ~13K training triples (90% split) |
| `ml/data/splits/biencoder_val.jsonl` | ~1.5K validation triples (10% split) |

### Step 2 — Train Bi-Encoder

Fine-tunes Harrier-OSS-v1-0.6B with `MultipleNegativesRankingLoss` (in-batch negatives + mined hard negatives from the teacher). Uses bf16 mixed precision on the A100.

Training time: ~30 minutes for 3 epochs with batch size 16 on a single A100.

```bash
# Make sure vLLM is stopped first (frees GPU memory):
pkill -f "vllm serve"

python -u ml/scripts/train_biencoder.py --epochs 3 --batch-size 16

# Or run in background:
nohup python -u ml/scripts/train_biencoder.py --epochs 3 --batch-size 16 > /tmp/train_biencoder.log 2>&1 &
tail -f /tmp/train_biencoder.log
```

Expected training loss: starts ~1.6, drops to ~0.2 by epoch 3.

| Output | Description |
|---|---|
| `ml/models/exported/biencoder/` | Trained Sentence Transformers model (1024-dim embeddings) |
| `ml/models/checkpoints/biencoder/` | Training checkpoints (latest 2 kept) |

### Step 3 — Evaluate Bi-Encoder

Evaluates retrieval quality on the validation set: encodes all val anchors and candidates, computes top-K nearest neighbors, checks if the true positive is retrieved.

```bash
python -u ml/scripts/eval_biencoder.py
```

Metrics reported: Recall@10, Recall@100, MRR — with per-country breakdown.

**Gate:** Recall@100 ≥ 85% (script prints GATE PASSED/FAILED). Our result: **Recall@100 = 98.8%, Recall@10 = 85.2%, MRR = 0.51**.

### Step 4 — Backup

```bash
export PUSH_REMOTE_HOST=webadmin@46.224.133.10
export PUSH_REMOTE_BASE=/home/webadmin/Jobl/ml-artifacts
./ml/scripts/push_artifacts.sh biencoder
```

ONNX export for production CPU inference is deferred to Phase 6 (Day 10) — it doesn't require GPU.

---

## ML Pipeline — Phase 3: Reranker Training (A100 GPU)

Fine-tunes XLM-RoBERTa-large as a CrossEncoder reranker. The reranker scores individual (resume, job) pairs from the bi-encoder's top-100 candidates, producing a relevance score 0.0–1.0 for final ranking.

Uses the 50K teacher-scored pairs directly as training data (graded relevance at four levels: 0.0/0.3/0.7/1.0), skipping the heavy hard-negative mining step since the teacher already provided diverse scored pairs at all quality levels.

### Step 1 — Build Training Data

Converts teacher match scores into CrossEncoder format. Enriches pairs that also appear in match_triples with full resume/job text.

```bash
python -u ml/scripts/build_reranker_data.py
```

| Output | Description |
|---|---|
| `ml/data/splits/reranker_train.jsonl` | ~44.5K training pairs (text_a, text_b, score) |
| `ml/data/splits/reranker_val.jsonl` | ~5K validation pairs |

### Step 2 — Train Reranker

Fine-tunes XLM-RoBERTa-large as a CrossEncoder with regression loss. Uses bf16 mixed precision.

Training time: ~37 minutes for 5 epochs with batch size 32 on a single A100.

```bash
# Make sure vLLM is stopped first:
pkill -f "vllm serve"

nohup python -u ml/scripts/train_reranker.py --epochs 5 --batch-size 32 > /tmp/train_reranker.log 2>&1 &
tail -f /tmp/train_reranker.log
```

Expected training loss: stabilizes around ~0.48 by epoch 5.

Note: the `CrossEncoderTrainer` saves checkpoints in subdirectories. After training, save the final model explicitly:

```bash
python -c "
from sentence_transformers.cross_encoder import CrossEncoder
import glob
latest = sorted(glob.glob('ml/models/exported/reranker/checkpoint-*'))[-1]
model = CrossEncoder(latest)
model.save('ml/models/exported/reranker-final')
print(f'Saved from {latest} to ml/models/exported/reranker-final')
"
```

| Output | Description |
|---|---|
| `ml/models/exported/reranker-final/` | Trained CrossEncoder model |

### Step 3 — Evaluate Reranker

Groups val pairs by resume, reranks with the CrossEncoder, computes NDCG@10 and MAP.

```bash
python -u ml/scripts/eval_reranker.py --model ml/models/exported/reranker-final
```

**Gate:** NDCG@10 ≥ 0.7 (script prints GATE PASSED/FAILED). Our result: **NDCG@10 = 0.854, MAP = 0.429**.

### Step 4 — Backup

```bash
export PUSH_REMOTE_HOST=webadmin@46.224.133.10
export PUSH_REMOTE_BASE=/home/webadmin/Jobl/ml-artifacts
./ml/scripts/push_artifacts.sh reranker
```

ONNX export for production CPU inference is deferred to Phase 6 (Day 10).

---

## ML Pipeline — Phase 4: Extractor Training (A100 GPU)

Fine-tunes Qwen2.5-7B-Instruct with LoRA for structured field extraction from job postings and resumes. This is the universal Extractor (Component 3) that replaces the existing `normalizer.py` + `categorizer.py`, producing 18 structured fields (normalized_title, seniority, occupation_category, skills, salary, etc.) across all 39 countries.

### Step 1 — Build SFT Dataset

Converts teacher extractions into chat-format SFT examples: system prompt + job posting → structured JSON output.

```bash
# Pull teacher labels if on a fresh instance:
mkdir -p ml/data/teacher_labels
scp webadmin@46.224.133.10:/home/webadmin/Jobl/ml-artifacts/data/teacher_labels/* ml/data/teacher_labels/

python -u ml/scripts/build_extractor_data.py
```

| Output | Description |
|---|---|
| `ml/data/splits/extractor_train.jsonl` | ~17K training examples (chat format) |
| `ml/data/splits/extractor_val.jsonl` | ~1.9K validation examples |

### Step 2 — Train LoRA

Fine-tunes Qwen2.5-7B-Instruct with LoRA (rank 32, alpha 64) targeting all attention + MLP projections. Uses TRL SFTTrainer with bf16 mixed precision.

Training time: ~3.5 hours for 3 epochs on a single A100.

```bash
# Make sure vLLM is stopped:
pkill -f "vllm serve"

nohup python -u ml/scripts/train_extractor_lora.py --epochs 3 --batch-size 4 > /tmp/train_extractor.log 2>&1 &
tail -f /tmp/train_extractor.log
```

LoRA config: 80.7M trainable params (1.05% of 7.7B total). Expected loss: starts ~2.1, drops to ~0.52 by epoch 3. Token accuracy reaches ~88%.

| Output | Description |
|---|---|
| `ml/models/checkpoints/extractor-lora/` | LoRA adapter weights + tokenizer |

### Step 3 — Merge LoRA

Merges LoRA adapter weights into the base model, producing a standalone model for GGUF conversion and evaluation.

```bash
python -u ml/scripts/merge_lora.py
```

| Output | Description |
|---|---|
| `ml/models/exported/extractor-merged/` | Full merged 7B model (bf16) |

### Step 4 — Evaluate Extractor

Runs inference on val examples, parses JSON output, computes field-level accuracy against teacher extractions.

```bash
python -u ml/scripts/eval_extractor.py --limit 200
```

Our results:

| Field | Accuracy |
|---|---|
| location_country | 99.5% |
| language | 99.5% |
| work_mode | 97.4% |
| salary_present | 96.3% |
| employment_type | 94.2% |
| contract_type | 92.2% |
| seniority | 85.9% |
| occupation_category | 80.1% |
| normalized_title | 69.6% |

JSON validity: 95.5%. Overall field accuracy: **90.5%**.

**Gate:** overall field accuracy ≥ 90% (script prints GATE PASSED/FAILED). Note: `normalized_title` accuracy is low because the model produces valid but differently-phrased titles (e.g. "Software Developer" vs "Software Engineer"), not because the extractions are wrong.

### Step 5 — Backup

```bash
export PUSH_REMOTE_HOST=webadmin@46.224.133.10
export PUSH_REMOTE_BASE=/home/webadmin/Jobl/ml-artifacts
./ml/scripts/push_artifacts.sh extractor
```

GGUF conversion for production CPU inference is done in Phase 6 (Day 10).

---

## ML Pipeline — Phase 5: Full-Corpus Inference (A100 GPU)

Encodes all jobs and resumes with the trained bi-encoder, producing dense vectors for ANN search in production. The extractor is NOT run over the full corpus (1.4M jobs at ~1-2s/record would take ~16 days) — it's designed for incremental use on new postings as they arrive.

### Step 1 — Encode Full Corpus

Encodes all jobs (train + val pools = ~1.4M) and resumes (~225K) with locale prefixes matching the training format.

```bash
# Test with 1000 first:
python -u ml/scripts/full_corpus_embed.py --limit 1000

# Full run (~3 hours on A100):
nohup python -u ml/scripts/full_corpus_embed.py --batch-size 256 > /tmp/full_corpus_embed.log 2>&1 &
tail -f /tmp/full_corpus_embed.log
```

Our results: 1,617,995 vectors encoded in ~3 hours at 148-203 texts/sec.

| Output | Size | Description |
|---|---|---|
| `ml/data/vectors/job_vectors.npy` | 5.71 GB | 1,392,877 × 1024 float32 matrix |
| `ml/data/vectors/job_ids.json` | — | Ordered list of job IDs matching vector rows |
| `ml/data/vectors/resume_vectors.npy` | 0.92 GB | 225,118 × 1024 float32 matrix |
| `ml/data/vectors/resume_ids.json` | — | Ordered list of resume IDs matching vector rows |

### Step 2 — Backup

```bash
export PUSH_REMOTE_HOST=webadmin@46.224.133.10
export PUSH_REMOTE_BASE=/home/webadmin/Jobl/ml-artifacts
./ml/scripts/push_artifacts.sh corpus_outputs
```

---

## ML Pipeline — Phase 6: Packaging & Export (A100 GPU or CPU)

Convert trained models to production-serving formats: ONNX for the bi-encoder and reranker (CPU inference via onnxruntime), GGUF Q4_K_M for the extractor (CPU inference via llama.cpp).

**Disk space note:** the GGUF conversion needs ~30GB free (15GB f16 intermediate + 4.4GB quantized output + working space). Delete base/teacher models first if disk is tight:

```bash
rm -rf ml/models/base/teacher ml/models/base/teacher-32b
```

### Step 1 — ONNX Export (Bi-Encoder + Reranker)

Exports both models to ONNX format for fast CPU inference. Tries `optimum` first, falls back to `torch.onnx.export`. The Harrier bi-encoder requires `position_ids` as an ONNX input — the export script handles this automatically.

```bash
pip install optimum[onnxruntime] onnxscript
python -u ml/scripts/export_onnx.py --model both
```

TracerWarnings during export are normal and can be ignored. Verification runs automatically after export.

| Output | Description |
|---|---|
| `ml/models/exported/biencoder-onnx/` | ONNX bi-encoder + tokenizer (requires input_ids, attention_mask, position_ids) |
| `ml/models/exported/reranker-onnx/` | ONNX reranker + tokenizer |

### Step 2 — GGUF Conversion (Extractor)

Converts the merged 7B extractor to GGUF Q4_K_M format (4.4 GB) for CPU inference via llama-cpp-python. Clones llama.cpp automatically for the converter and quantizer.

**Prerequisites:** `cmake` and `build-essential` must be installed for building the llama.cpp quantizer:

```bash
sudo apt install -y cmake build-essential
```

**Important:** copy the original base model tokenizer files to the merged model directory before conversion (the LoRA merge may save incompatible tokenizer metadata):

```bash
cp ml/models/base/extractor/tokenizer* ml/models/exported/extractor-merged/
cp ml/models/base/extractor/vocab* ml/models/exported/extractor-merged/ 2>/dev/null
cp ml/models/base/extractor/merges* ml/models/exported/extractor-merged/ 2>/dev/null
cp ml/models/base/extractor/special_tokens* ml/models/exported/extractor-merged/ 2>/dev/null
```

Then run conversion:

```bash
./ml/scripts/convert_gguf.sh
# Or with a different quantization level:
./ml/scripts/convert_gguf.sh ml/models/exported/extractor-merged ml/models/exported/extractor-gguf Q5_K_M
```

Our result: 14.5 GB (f16) → **4.4 GB (Q4_K_M)**, 4.91 bits per weight. Quantization took ~80 seconds.

| Output | Description |
|---|---|
| `ml/models/exported/extractor-gguf/model-Q4_K_M.gguf` | Quantized 7B extractor (4.4 GB) |

If field-level accuracy drops >5% vs the full-precision model after quantization, fall back to Q5_K_M or Q8_0.

### Step 3 — Final Backup

Push all artifacts to remote storage — models, vectors, training data, configs. After backup, the A100 can be shut down.

```bash
export PUSH_REMOTE_HOST=webadmin@46.224.133.10
export PUSH_REMOTE_BASE=/home/webadmin/Jobl/ml-artifacts
./ml/scripts/push_artifacts.sh final

# Archive venv before shutting down
tar czf /tmp/venv-training.tar.gz -C /home/webadmin .venv-training
scp /tmp/venv-training.tar.gz webadmin@46.224.133.10:/home/webadmin/Jobl/
```

### Production Artifacts Summary

All trained, exported, and ready for deployment on the CPU production server.

| Component | Format | Size | Runtime | When it runs |
|---|---|---|---|---|
| Bi-encoder | ONNX | ~1.2 GB | onnxruntime | Real-time, every query |
| Reranker | ONNX | ~2.2 GB | onnxruntime | Real-time, every query |
| Extractor | GGUF Q4_K_M | 4.4 GB | llama-cpp-python | Batch, on new postings |
| Job vectors | numpy .npy | 5.71 GB | pgvector / Qdrant | Loaded at startup |
| Resume vectors | numpy .npy | 0.92 GB | pgvector / Qdrant | Loaded at startup |

### Training Results Summary

| Model | Metric | Result | Gate |
|---|---|---|---|
| Bi-encoder | Recall@100 | 98.8% | ≥ 85% ✅ |
| Reranker | NDCG@10 | 0.854 | ≥ 0.7 ✅ |
| Extractor | Field accuracy | 90.5% | ≥ 90% ✅ |
