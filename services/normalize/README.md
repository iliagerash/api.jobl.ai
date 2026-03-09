# Job Normalize Service (`services/normalize`)

Worker service that fills:
- `jobs.title_normalized`
- `jobs.description_clean`
- `jobs.description_html`

The command runs once and exits (cron-friendly).

Current API-safe normalization rules:
- removes trailing work-mode markers in titles (`remote`, `hybrid`, `on-site`, `% remote`, `remote position`)
- removes obvious non-title title noise aligned with Google for Jobs guidance (hiring spam terms, job/ref codes, salary/date fragments, likely company/address suffixes)
- removes trailing location parts from titles using context columns (`city_title`, `region_title`, `country_code`) and `countries.name`
- also uses `countries.alternate_names` (`text[]`) for country aliases like `UK`, `Great Britain`
- preserves legal title markers like `(m/w/d)` and `(m/f/d)` for frontend/API use
- keeps useful location text when mode markers appear in trailing parentheses
- cleans HTML descriptions into readable text with list bullets and paragraph breaks

## Run locally

```bash
cd services/normalize
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
cp .env.example .env
jobl-normalize
```

## Useful flags

```bash
jobl-normalize --batch-size=2000
jobl-normalize --max-batches=10
jobl-normalize --from-id=500000
```

## Build language-proportional samples for evaluation

After running migrations, build samples directly from `jobs`.

Generate a language-balanced sample (default total: `50000`):

```bash
cd /home/<user>/Jobl/api.jobl.ai/services/normalize
source .venv/bin/activate
jobl-normalize-sample
```

Generate a custom total and replace existing samples:

```bash
jobl-normalize-sample --total=15000 --replace
```

Notes:
- `--batch-tag` is no longer a CLI option; the script generates one automatically.
- sampling is driven by target language proportions across:
  `en, es, de, fr, pt, it, gr, nl, da, uk`
- rows with `jobs.language_code IS NULL` are skipped

## Run evaluation on samples

Fill generated columns and match flags in `normalization_samples`:

```bash
jobl-normalize-eval --batch-tag=eval_v1 --only-pending
```

Evaluate only a subset:

```bash
jobl-normalize-eval --batch-tag=eval_v1 --limit=500
```

## Label Samples With LLM (Raw Input, No Rules)

Use this flow when you want labels from raw unprocessed data.
`jobl-normalize-llm-label` sends `title_raw` and `description_raw` directly to OpenAI and writes outputs to:
- `expected_title_normalized`
- `expected_description_html`

Implementation notes:
- strict structured output (`json_schema`) is enforced
- categorization is intentionally omitted from this pipeline

Configure `.env`:

```bash
OPENAI_API_KEY="..."
OPENAI_MODEL="gpt-5-nano"
OPENAI_TIMEOUT_SECONDS=60
OPENAI_MAX_RETRIES=3
OPENAI_BATCH_COMPLETION_WINDOW="24h"
OPENAI_BATCH_POLL_SECONDS=15
LLM_PROMPT_VERSION="v1"
```

Label next pending rows and set a processing tag (OpenAI Batch API is default):

```bash
jobl-normalize-llm-label --batch-tag=train_v1_0001 --limit=1000
```

Randomly sample pending rows (useful for labeling new training batches):

```bash
jobl-normalize-llm-label --batch-tag=train_v1_0002 --limit=2000 --random
```

Resume polling/apply for an already submitted OpenAI batch:

```bash
jobl-normalize-llm-label --batch-id=batch_xxx --batch-tag=train_v1_0001
```

Disable batch mode and run one-by-one requests:

```bash
jobl-normalize-llm-label --batch-tag=train_v1_0001 --limit=200 --no-batch
```

Debug direct mode (full prompt + full raw response):

```bash
jobl-normalize-llm-label --batch-tag=train_v1_debug --limit=20 --no-batch --debug
```

Notes:
- `--batch-tag` now writes/overwrites `normalization_samples.batch_tag` for processed rows.
- selection is driven by `--limit` over unprocessed rows only (not by existing batch tag).
- `--batch-id` resumes a previously submitted batch and applies its output to matching `normalization_samples.id` rows.

## ML-only title normalization

For model training prep, use ML-only title cleanup that strips legal/gender markers like `(m/w/d)`.

Important:
- `jobl-normalize` (main worker) is unchanged and remains API/frontend-safe.
- ML cleanup is isolated and should be used only for training datasets.

Normalize one title:

```bash
jobl-normalize-ml-title --text="Senior Accountant (m/f/d) - Berlin"
```

Normalize titles in a CSV and append `title_ml` column:

```bash
jobl-normalize-ml-title \
  --input-csv=/home/<user>/Downloads/normalization_samples.csv \
  --output-csv=/home/<user>/Downloads/normalization_samples_ml.csv \
  --title-column=title_raw
```
