# Jobl AI Backend (`api.jobl.ai`)

Backend monorepo for Jobl AI services.

## Repository layout

- `services/api`: FastAPI service, SQLAlchemy models, Alembic migrations.
- `services/sync`: scraper ingestion worker (fetch + export marking + upsert pipeline).
- `services/normalize`: backlog normalization worker for processed text fields.
- `services/llm`: LLM-focused workers/pipelines (separate from HTTP API).
- `libs/common`: shared Python utilities used across services.

Keeping API in `services/api` is the right approach because it isolates dependencies, migration tooling, and deployment lifecycle from LLM experiments/workers.

## Dependencies installation

From repository root:

```bash
cd /home/<user>/Jobl/api.jobl.ai
```

Install API dependencies:

```bash
cd services/api
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Install sync dependencies:

```bash
cd /home/<user>/Jobl/api.jobl.ai/services/sync
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ../../libs/common
pip install -e .
```

Install optional dev dependencies (inside `services/api` or `services/sync`):

```bash
pip install -e ".[dev]"
```

Install normalize dependencies:

```bash
cd /home/<user>/Jobl/api.jobl.ai/services/normalize
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## FastAPI bootstrap

```bash
cd services/api
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
cp .env.example .env
uvicorn app.main:app --reload
```

Health check: `GET http://127.0.0.1:8000/v1/health`

## Alembic bootstrap

```bash
cd services/api
alembic revision -m "init"
alembic upgrade head
```

To apply latest schema/index changes (including `jobs_active` view, normalization backlog index, and countries lookup):

```bash
cd services/api
alembic upgrade head
```

## Seed lookup tables

Seeder SQL file:
- `services/api/sql/seed_source_countries.sql`
- `services/api/sql/seed_countries.sql`

Current seed data:
- `db_name = 'americas'`
- `country_code = NULL`
- `currency = NULL`
- `config = {"country_code_in_city": 1}`

Behavior:
- idempotent upsert (`ON CONFLICT (db_name) DO UPDATE`)
- safe to run multiple times

```bash
cd services/api
psql "$DATABASE_URL" -f sql/seed_source_countries.sql
psql "$DATABASE_URL" -f sql/seed_countries.sql
```

## Sync worker bootstrap

```bash
cd services/sync
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ../../libs/common
pip install -e .
cp .env.example .env
jobl-sync
```

## Normalize worker bootstrap

```bash
cd services/normalize
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
cp .env.example .env
jobl-normalize --batch-size=2000
```

## Related repositories

- Documentation: [`docs.jobl.ai`](https://github.com/iliagerash/docs.jobl.ai)
- Frontend (React): [`jobl.ai`](https://github.com/iliagerash/jobl.ai)
