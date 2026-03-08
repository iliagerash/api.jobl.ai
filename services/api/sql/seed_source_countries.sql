-- Seed initial source country DB mappings for sync service.
-- country_code is NULL for "americas" because country is read from job rows.

INSERT INTO source_countries (db_name, country_code, config)
VALUES (
    'americas',
    NULL,
    '{"country_code_in_job": 1}'::json
)
ON CONFLICT (db_name) DO UPDATE
SET country_code = EXCLUDED.country_code,
    config = EXCLUDED.config;
