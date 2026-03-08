-- Seed initial source country DB mappings for sync service.
-- country_code is NULL for "americas" because country is read from city rows.

INSERT INTO source_countries (db_name, country_code, currency, config)
VALUES (
    'americas',
    NULL,
    NULL,
    '{"country_code_in_city": 1, "currency_in_job": 1}'::json
),
(
    'australia',
    'AU',
    'AUD',
    NULL::json
),
(
    'austria',
    'AT',
    'EUR',
    NULL::json
),
(
    'belgium',
    'BE',
    'EUR',
    NULL::json
),
(
    'brazil',
    'BR',
    'BRL',
    NULL::json
),
(
    'canada',
    'CA',
    'CAD',
    '{"region_in_city": "region"}'::json
),
(
    'denmark',
    'DK',
    'DKK',
    NULL::json
),
(
    'france',
    'FR',
    'EUR',
    NULL::json
),
(
    'germany',
    'DE',
    'EUR',
    NULL::json
),
(
    'greece',
    'GR',
    'EUR',
    '{"region_in_city": "region_title"}'::json
),
(
    'india',
    'IN',
    'INR',
    NULL::json
),
(
    'indonesia',
    'ID',
    'IDR',
    NULL::json
),
(
    'italy',
    'IT',
    'EUR',
    NULL::json
),
(
    'malaysia',
    'MY',
    'MYR',
    '{"region_in_city": "region"}'::json
),
(
    'netherlands',
    'NL',
    'EUR',
    NULL::json
),
(
    'newzealand',
    'NZ',
    'NZD',
    NULL::json
),
(
    'nigeria',
    'NG',
    'NGN',
    NULL::json
),
(
    'pakistan',
    'PK',
    'PKR',
    '{"region_in_city": "region_title"}'::json
),
(
    'philippines',
    'PH',
    'PHP',
    '{"region_in_city": "region"}'::json
),
(
    'portugal',
    'PT',
    'EUR',
    NULL::json
),
(
    'saudiarabia',
    'SA',
    'SAR',
    '{"region_in_city": "region_title"}'::json
),
(
    'singapore',
    'SG',
    'SGD',
    '{"region_in_city": "region_title"}'::json
),
(
    'southafrica',
    'ZA',
    'ZAR',
    '{"region_in_city": "region"}'::json
),
(
    'switzerland',
    'CH',
    'CHF',
    '{"region_in_city": "region_title"}'::json
),
(
    'uae',
    'AE',
    'AED',
    '{"region_in_city": "region_title"}'::json
),
(
    'uk',
    'GB',
    'GBP',
    NULL::json
),
(
    'ukraine',
    'UA',
    'UAH',
    NULL::json
),
(
    'usa',
    'US',
    'USD',
    '{"region_in_city": "region"}'::json
)
ON CONFLICT (db_name) DO UPDATE
SET country_code = EXCLUDED.country_code,
    currency = EXCLUDED.currency,
    config = EXCLUDED.config;
