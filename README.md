# api.jobl.ai

FastAPI service that processes raw job postings: cleans HTML descriptions, extracts expiry dates and application emails, categorizes jobs into 26 industry categories, generates embedding vectors for semantic search, extracts skills from multilingual job descriptions, predicts salary ranges, estimates revenue potential, and matches jobs to a normalized keyword taxonomy.

Includes a background sync worker that pulls jobs from MySQL source databases into PostgreSQL.

## Production Architecture

```
Scraper flow (every new job):
  Scraper → POST /v1/process → category + clean description + embedding + skills
                              + salary prediction + profitability + keyword
          → stores in scraper DB → flows to job boards

Job board flow (direct submissions):
  Resume/job submitted → POST /v1/embed → 1024-dim vector + keyword
          → type=resume: + doc_type (resume/cover_letter/stub/other)
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
| Salary predictor | ~10 MB | LightGBM per-country salary range prediction |
| Profitability predictor | ~10 MB | Two-stage LightGBM revenue/RPS prediction |
| Keyword matching | DB query | Nearest keyword from 1,391 normalized job titles via pgvector |
| **Total per worker** | **~2.6 GB** | |
| **4 workers + Postgres + OS** | **~14.5 GB** | **~15 GB headroom** |

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [API Endpoint](#api-endpoint)
  - [`POST /v1/classify`](#post-v1classify)
  - [`POST /v1/embed`](#post-v1embed)
  - [`POST /v1/process`](#post-v1process)
- [Processing Pipeline](#processing-pipeline)
- [Graceful Degradation](#graceful-degradation)
- [Database](#database)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running Locally](#running-locally)
- [Migrations](#migrations)
- [Categorizer Training](#categorizer-training)
  - [Hyperparameter Tuning](#2a-tune-hyperparameters-optional)
- [Intelligence Models](#intelligence-models)
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
      ├─ EN/FR only ─────────────────────────────────────────┐
      │       ├─ Email extraction                            │
      │       └─ Categorization (LightGBM, 26 categories)   │
      │   Non-EN/FR: original_category passed through ──────┘
      │
      ├─ Embedding (bi-encoder ONNX, 1024-dim vector)
      │
      ├─ Skill extraction (ESCO taxonomy, 13,900 skills × 10 languages)
      │
      ├─ Keyword matching (nearest from 1,391 normalized titles via pgvector)
      │
      ├─ Salary prediction (LightGBM per-country, skipped if salary stated)
      │
      └─ Profitability prediction (two-stage LightGBM: classifier + regressor)
```

```
POST /v1/classify
      │
      └─ Document classification (regex heuristic, no model, no DB)
           → doc_type: resume | cover_letter | stub | other
```

```
POST /v1/embed
      │
      ├─ Embedding (bi-encoder ONNX, 1024-dim vector)
      │
      ├─ Keyword matching (nearest from 1,391 normalized titles via pgvector)
      │
      └─ type=resume only ──────────────────────────────────────────────────┐
              └─ Document classification (regex heuristic, no model)        │
                   → doc_type: resume | cover_letter | stub | other  ───────┘
```

The bi-encoder, categorizer, and skill extractor are **optional**. The API starts and serves requests without them; those fields return `null`.

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
│   │   ├── language.py              # langid-based language detection
│   │   ├── cleaner.py               # HTML cleaner + expiry extractor
│   │   ├── normalizer.py            # seq2seq title normalizer
│   │   ├── categorizer.py           # LightGBM job categorizer
│   │   ├── doc_classifier.py        # resume_classify: doc_type heuristic (no model)
│   │   ├── salary_predictor.py      # LightGBM salary range prediction
│   │   └── profitability_predictor.py  # Two-stage LightGBM revenue prediction
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

### `POST /v1/classify`

Lightweight document-type classifier. Accepts a title and optional description; returns `doc_type` only. No model loading, no embedding, no DB access — pure in-process heuristic.

#### Request

```json
{
  "title": "Senior Software Engineer",
  "description": "Work Experience\n- 5 years Python\n\nEducation\nB.Sc. Computer Science"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `title` | string (1–512 chars) | yes | Resume position title |
| `description` | string | no | Plain-text or HTML body |

#### Response

```json
{ "doc_type": "resume" }
```

`doc_type` is one of `resume`, `cover_letter`, `stub`, or `other` (see [`POST /v1/embed`](#post-v1embed) for value definitions).

---

### `POST /v1/embed`

Generates a 1024-dim embedding vector for a job title or resume, matches the nearest keyword, and — when `type=resume` — classifies the document type.

#### Request

```json
{
  "title": "Senior Software Engineer",
  "description": "...",
  "type": "resume",
  "language": "en",
  "country": "AU",
  "category_id": 4
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `title` | string (1–512 chars) | yes | Job title or resume position |
| `description` | string | no | Raw HTML or plain-text body |
| `type` | `"job"` \| `"resume"` | no | Controls embedding prefix and enables classification (default: `"job"`) |
| `language` | string | no | ISO 2-letter language code; auto-detected if omitted |
| `country` | string | no | ISO 2-letter country code |
| `category_id` | int | no | Restricts keyword matching to this category |

#### Response

```json
{
  "embedding": [0.012, -0.034, "..."],
  "language": "en",
  "keyword": {
    "canonical_id": 18,
    "title": "Software Engineer",
    "distance": 0.2341
  },
  "doc_type": "resume"
}
```

| Field | Type | Description |
|---|---|---|
| `embedding` | array | 1024-dim float vector |
| `language` | string \| null | Detected or provided language code |
| `keyword` | object \| null | `{canonical_id, title, distance}` — nearest normalized title within distance 0.5; null if no match |
| `doc_type` | string \| null | `resume`, `cover_letter`, `stub`, or `other` — set only when `type=resume`; null for jobs |

**`doc_type` values:**

| Value | Meaning |
|---|---|
| `resume` | Structured CV — has recognised section headers (`Work Experience`, `Education`, etc.) and/or substantial formatted content |
| `cover_letter` | Short prose letter — no or minimal section headers, under ~1,500 chars |
| `stub` | Too thin to be useful — description under 100 chars |
| `other` | Doesn't look like a candidate document (e.g. a job posting accidentally in the resume feed) |

Classification runs in-process with no model loading and adds negligible latency.

---

### `POST /v1/process`

#### Request

```json
{
  "title": "Senior Software Engineer - Full Time #ABC123",
  "description": "<div><p>Apply by April 30 2026. Send resume to jobs@example.com</p></div>",
  "original_category": "Technology",
  "country_code": "AU",
  "city_title": "Sydney",
  "region_title": "New South Wales",
  "company_name": "Acme Corp",
  "contract": "permanent",
  "is_remote": false,
  "salary_min": null,
  "salary_max": null,
  "salary_period": null,
  "destination": "au.workus.org"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `title` | string (1–512 chars) | yes | Raw job title from source |
| `description` | string | yes | Raw HTML job description |
| `original_category` | string | no | Source category label; passed through for non-EN/FR jobs |
| `country_code` | string | no | ISO 2-letter country code (enables salary prediction) |
| `city_title` | string | no | City name (improves salary/profitability accuracy) |
| `region_title` | string | no | Region/state name |
| `company_name` | string | no | Company name (used by profitability model) |
| `contract` | string | no | Contract type (permanent, contract, etc.) |
| `is_remote` | bool | no | Whether the job is remote (default: false) |
| `salary_min` | float | no | Stated salary min (if present, salary prediction is skipped) |
| `salary_max` | float | no | Stated salary max |
| `salary_period` | string | no | Salary period (hour, day, week, month, year) |
| `destination` | string | no | Target job board domain (used by profitability model) |

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
  },
  "embedding": [0.012, -0.034, ...],
  "skills": ["Python", "Django", "PostgreSQL"],
  "keyword": {
    "canonical_id": 18,
    "title": "Software Engineer",
    "distance": 0.2341
  },
  "salary_prediction": {
    "salary_min": 91000.0,
    "salary_max": 120000.0,
    "salary_period": "yearly"
  },
  "profitability": {
    "predicted_revenue": 0.045231,
    "predicted_rps": 0.008123,
    "priority_tier": "medium"
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
| `embedding` | array \| null | 1024-dim float vector; null if bi-encoder not loaded |
| `skills` | array \| null | Extracted skill names; null if skill extractor not loaded |
| `keyword` | object \| null | `{canonical_id, title, distance}` — nearest normalized job title; null if no match within distance 0.5 or category/embedding missing |
| `salary_prediction` | object \| null | `{salary_min, salary_max, salary_period}` — predicted salary range; null if salary already stated, country not supported (NZ, ZA), or model not loaded |
| `profitability` | object \| null | `{predicted_revenue, predicted_rps, priority_tier}` — revenue potential estimate; null if model not loaded |

`category.confidence` is the model's softmax probability for the predicted class (0–1). It is `null` for non-EN/FR jobs where the model is not run.

For non-EN/FR jobs, `category` is `{"id": null, "title": "<original_category>", "confidence": null}` if `original_category` was provided.

`salary_prediction` is skipped when `salary_min > 0` is provided in the request (salary already stated). Supported countries: AU, CA, GB, SG, US. Period is `monthly` for SG, `yearly` for others.

`profitability.priority_tier` is one of `high`, `medium`, `low`, `minimal` based on the combined revenue prediction (classifier probability × regressor output).

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
| `BIENCODER_MODEL_PATH` not set | `embedding: null`, `keyword: null` |
| `INTELLIGENCE_MODELS_PATH` not set | `salary_prediction: null`, `profitability: null` |
| Any model fails to load at startup | App starts; affected field is `null`; error logged |
| Non-EN/FR language detected | Title unchanged; description cleaned; `original_category` passed through if provided |
| No email found in description | `application_email: null` |
| Expiry date not found or past | `expiry_date: null` |
| Salary already stated (`salary_min > 0`) | `salary_prediction: null` (skipped) |
| Country not supported for salary (NZ, ZA) | `salary_prediction: null` |
| No matching keyword within distance 0.5 | `keyword: null` |
| `type=job` on `/v1/embed` | `doc_type: null` — classification only runs for resumes |

---

## Database

PostgreSQL. Schema managed by Alembic.

### Tables

| Table | Description |
|---|---|
| `jobs` | Synced job postings (title, description, location, salary, AI fields, predictions) |
| `keywords` | Normalized job title taxonomy (1,391 canonical titles × en/fr with embeddings) |
| `job_keywords` | Junction table: job → nearest keyword with distance |
| `categories` | 26 industry categories (id, title) |
| `category_map` | Maps source category strings → local `category_id` |
| `countries` | Country lookup (code, name, alternate_names, language_codes) |
| `source_countries` | MySQL source DB → country/currency/config mapping |
| `sync_state` | Sync cursor: last synced job ID per (source_db, destination) pair |
| `resumes` | Raw resume postings for model training (`source_website` + `external_id`, location, salary, `language_code`, `doc_type`); populated by `jobl-sync-resumes`. **Not anonymized** — descriptions may contain names, emails, phone numbers, addresses; PII scrubbing happens downstream in the ML pipeline's data-prep step, not at ingest |

**`resumes` columns:** `id`, `source_website`, `external_id`, `title`, `description`, `city_title`, `region_title`, `country_code`, `salary`, `salary_period`, `salary_currency`, `contract`, `published_at`, `is_remote`, `language_code`, `embedding`, `skills`, `doc_type`. Unique on `(source_website, external_id)`.

### Migrations

23 migrations in a linear chain. Current head: `e1f2a3b4c5d6` (add_doc_type_to_resumes).

---

## Setup

**Requirements**: Python 3.12, PostgreSQL 17+, Ubuntu 24.04 LTS

**System prerequisites**:

```bash
# Python build tools + LightGBM runtime dependency
sudo apt install -y python3.12-venv python3.12-dev libgomp1

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
python sql/seed_skills_tech.py
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
| `INTELLIGENCE_MODELS_PATH` | — | Path to directory containing salary/profitability `.pkl` models (e.g. `models`) |
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

## Intelligence Models

Three additional models provide salary predictions, profitability estimates, and keyword matching for processed jobs.

### Salary Predictor

LightGBM regression models — one per country — predicting `salary_min` and `salary_max` in the default period (yearly for AU/CA/GB/US, monthly for SG). NZ and ZA excluded due to insufficient data.

```bash
# Extract features (requires DATABASE_URL)
python -u ml/scripts/salary_features.py

# Train (with optional Optuna tuning)
python -u ml/scripts/train_salary.py
python -u ml/scripts/train_salary.py --tune --tune-trials 20

# Evaluate
python -u ml/scripts/eval_salary.py
```

Outputs: `models/salary_{CC}.pkl` per country.

Features: target-encoded title, city, region, category; country, work mode, contract type, description word count.

### Profitability Ranker

Two-stage LightGBM model predicting revenue potential:
1. **Classifier** — predicts P(revenue > 0) for a job
2. **Regressor** — predicts how much revenue / revenue per session (trained on positive-revenue jobs only)

Final prediction: `P(revenue > 0) × predicted_revenue`

```bash
# Export GA data (requires service account keys in data/ga_accounts/)
python -u scripts/export_ga.py --key data/ga_accounts/Workus.json
python -u scripts/export_ga.py --key data/ga_accounts/JobRadars.json
python -u scripts/export_ga.py --key data/ga_accounts/SnapJobSearch.json

# Parse and join with jobl jobs
python -u ml/scripts/parse_ga.py

# Extract features
python -u ml/scripts/profitability_features.py

# Train
python -u ml/scripts/train_profitability.py --tune --tune-trials 20
```

Outputs: `models/profitability_classifier.pkl`, `models/profitability_revenue.pkl`, `models/profitability_rps.pkl`

Features: target-encoded title, company, city, region, category; country, destination, salary info, work mode, contract, description length, city population tier, day of week, month.

Priority tiers: `high` (revenue > 0.1), `medium` (> 0.01), `low` (> 0.001), `minimal`.

### Keyword Taxonomy

1,391 normalized job titles (en + fr) generated by clustering 517K distinct job titles with k-means, naming clusters with Qwen2.5-72B, and reviewing with GPT. Each keyword has a `canonical_id` linking English and French variants, and a `category_id` for disambiguation.

```bash
# Generate taxonomy (GPU host with vLLM)
python -u ml/scripts/generate_taxonomy.py --model ml/models/base/qwen-72b-awq

# Deduplicate
python -u ml/scripts/dedup_taxonomy.py

# Review with OpenAI
python -u ml/scripts/review_taxonomy.py

# Load into database (generates biencoder embeddings)
python -u scripts/load_keywords.py --clear

# Assign active jobs to nearest keywords
python -u scripts/assign_keywords.py
```

Keyword matching at request time uses pgvector cosine distance, filtered by `language_code` and `category_id`. Jobs with no keyword within distance 0.5 return `keyword: null`.

### Activate intelligence models

Set `INTELLIGENCE_MODELS_PATH=models` in `.env` and restart the API. The `keywords` table must be populated in the database for keyword matching.

### Backfill predictions on scraper

```bash
# On scraper host (with API running on GPU):
php backfill_predictions.php australia

# Sync predictions to jobl:
python -u scripts/sync_predictions_from_scraper.py
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
| *(derived)* | `language_code` — auto-detected from title + description |
| *(derived)* | `doc_type` — classified from title + description (`resume` / `cover_letter` / `stub` / `other`) |

Each `<resume>` must include `<id>` (numeric), non-empty `<position>`, and non-empty `<description>`. Rows missing any of these are skipped. `language_code` is detected on import via `app/services/language.py`. `doc_type` is classified on import via `app/services/doc_classifier.py` — no API call or model required.

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


