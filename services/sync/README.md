# Job Sync Service (`services/sync`)

Worker service that:
- loads enabled source DBs from AI Postgres `source_countries`
- fetches new jobs from scraper country DBs
- upserts them into AI storage
- marks exported jobs in scraper DB `export` table with `destination='jobl.ai'`

Configuration model:
- source host connection is shared (`SOURCE_DB_HOST/PORT/USER/PASSWORD`)
- DB names are discovered from `source_countries.db_name`
- `source_countries.config` is optional JSON:
  - empty/null: default behavior
  - `{"country_code_in_job": 1}`: take country code from source job row during fetch

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
