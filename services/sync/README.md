# Job Sync Service (`services/sync`)

Worker service that:
- loads enabled source DBs from AI Postgres `source_countries`
- fetches new jobs from scraper country DBs
- upserts them into AI storage
- marks exported jobs in scraper DB `export` table with `destination='jobl.ai'`
- tracks per-DB high-watermark (`sync_state.last_job_id`) to fetch only new source rows
- detects and stores `jobs.language_code` for training routing

Configuration model:
- source host connection is shared (`SOURCE_DB_HOST/PORT/USER/PASSWORD`)
- SSL can be disabled for source MySQL with `SOURCE_DB_SSL_DISABLED=1` (default)
- DB names and default currency are discovered from `source_countries` (`db_name`, `currency`)
- `source_countries.config` is optional JSON:
  - empty/null: default behavior
  - `{"country_code_in_city": 1}`: take country code from `city.country_code` during fetch
  - `{"currency_in_job": 1}`: take currency from `job.salary_currency`; otherwise use `source_countries.currency`
  - `{"region_in_city": "column_name"}`: take region title from `city.column_name` instead of joining `region` table

Language detection:
- allowed output set is restricted to: `en, es, de, fr, pt, it, gr, uk, nl, da`
- any other detected/undetected locale is stored as `NULL` (excluded from training)

## Run locally

```bash
cd services/sync
python -m venv .venv
source .venv/bin/activate
pip install -e ../../libs/common
pip install -e .
cp .env.example .env
jobl-sync
```

Run only specific source DB(s):

```bash
jobl-sync --db=americas
jobl-sync --db=americas --db=australia
```

The command runs once and exits. For hourly execution, use cron.

Example crontab (runs every hour at minute 5):

```cron
5 * * * * cd /home/<user>/Jobl/api.jobl.ai/services/sync && . .venv/bin/activate && jobl-sync >> /tmp/jobl-sync.log 2>&1
```

## Backfill language for existing jobs

```bash
cd /home/<user>/Jobl/api.jobl.ai/services/sync
source .venv/bin/activate
jobl-sync-language-backfill --batch-size=2000
```

Recompute all rows:

```bash
jobl-sync-language-backfill --overwrite --batch-size=2000
```
