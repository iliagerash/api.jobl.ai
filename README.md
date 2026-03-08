# Jobl AI Backend (`api.jobl.ai`)

Backend monorepo for Jobl AI services.

## Repository layout

- `services/api`: FastAPI service, SQLAlchemy models, Alembic migrations.
- `services/sync`: scraper ingestion worker (fetch + export marking + upsert pipeline).
- `services/llm`: LLM-focused workers/pipelines (separate from HTTP API).
- `libs/common`: shared Python utilities used across services.

Keeping API in `services/api` is the right approach because it isolates dependencies, migration tooling, and deployment lifecycle from LLM experiments/workers.

## Dependencies installation

From repository root:

```bash
cd /Users/webadmin/Work/Jobl/api.jobl.ai
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
cd /Users/webadmin/Work/Jobl/api.jobl.ai/services/sync
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

## Seed source countries

Seeder SQL file:
- `services/api/sql/seed_source_countries.sql`

Current seed data:
- `db_name = 'americas'`
- `country_code = NULL`
- `config = {"country_code_in_job": 1}`

Behavior:
- idempotent upsert (`ON CONFLICT (db_name) DO UPDATE`)
- safe to run multiple times

```bash
cd services/api
psql "$DATABASE_URL" -f sql/seed_source_countries.sql
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

## Related repositories

- Documentation: [`docs.jobl.ai`](https://github.com/iliagerash/docs.jobl.ai)
- Frontend (React): [`jobl.ai`](https://github.com/iliagerash/jobl.ai)
