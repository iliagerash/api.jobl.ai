# Job Training Service (`services/training`)

Utilities to prepare labeled normalization data and run an initial LoRA fine-tuning pilot.

## Install

```bash
cd /home/<user>/Jobl/api.jobl.ai/services/training
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
cp .env.example .env
```

For model training dependencies:

```bash
pip install -e ".[train]"
```

## Scripts reference

### `jobl-training-export`

Purpose:
- Reads labeled rows from Postgres `normalization_samples`.
- Produces the raw supervised dataset in JSONL.

Selection logic:
- `title_raw` and `description_raw` must be non-empty.
- `expected_title_normalized` and `expected_description_html` must be non-empty.
- `language_code` must be one of: `en, es, de, fr, pt, it, gr, uk, nl, da`.
- Optional: filter by one `batch_tag`.

Arguments:
- `--out`: output JSONL path, default `data/raw/labeled.jsonl`.
- `--limit`: max rows, `0` means all.
- `--batch-tag`: optional batch filter.

Output row fields:
- `id`, `language_code`, `country_code`, `country_name`, `region_title`, `city_title`
- `title_raw`, `description_raw`
- `expected_title_normalized`, `expected_description_html`

Example:

```bash
jobl-training-export --out=data/raw/labeled.jsonl --limit=1000
jobl-training-export --batch-tag=train_v1_0001 --out=data/raw/labeled.jsonl
```

### `jobl-training-split`

Purpose:
- Splits exported JSONL into `train/val/test`.
- Preserves language distribution by stratifying on `language_code`.

Arguments:
- `--in`: input JSONL path.
- `--out-dir`: output directory.
- `--train`, `--val`, `--test`: ratios; must sum to `1.0`.
- `--seed`: deterministic shuffle seed.

Outputs:
- `train.jsonl`, `val.jsonl`, `test.jsonl`
- `manifest.json` with total counts and language distribution per split.

Example:

```bash
jobl-training-split --in=data/raw/labeled.jsonl --out-dir=data/splits --train=0.8 --val=0.1 --test=0.1 --seed=42
```

### `jobl-training-build-jsonl`

Purpose:
- Converts split rows into instruction-format JSONL for SFT.
- Builds `messages` in `system/user/assistant` format.

Behavior:
- `system`: normalization policy prompt.
- `user`: JSON payload with raw fields (`title_raw`, `description_raw`, location context).
- `assistant`: JSON payload with expected labels (`title_normalized`, `description_html`).

Arguments:
- `--split`: one of `train|val|test`, default `train`.
- `--in`: split JSONL path, default `data/splits/<split>.jsonl`.
- `--out`: instruction JSONL path, default `data/sft/<split>.jsonl`.
- `--prompt-version`: stored in user payload, default `v1`.

Example:

```bash
jobl-training-build-jsonl
jobl-training-build-jsonl --split=val
jobl-training-build-jsonl --split=test
```

### `jobl-training-train-lora`

Purpose:
- Runs initial LoRA SFT pilot on instruction JSONL.
- Saves adapter checkpoint for local inference/evaluation.

Arguments:
- `--train-jsonl`, `--val-jsonl`: instruction datasets.
  Defaults: `data/sft/train.jsonl` and `data/sft/val.jsonl`.
- `--model`: HF base model id (default `microsoft/Phi-3-mini-4k-instruct`).
- `--out-dir`: training output directory.
- `--epochs`, `--batch-size`, `--grad-accum`, `--lr`, `--max-seq-len`: training hyperparameters.
- `--memory-safe`: CPU-safe profile for low-memory hosts.
  On CPU it switches default model to `Qwen/Qwen2.5-0.5B-Instruct`, forces `batch_size=1`,
  increases gradient accumulation, caps sequence length, and limits epochs for first pilot.

Outputs:
- Trainer checkpoints under `--out-dir`.
- Final adapter in `--out-dir/adapter`.

Example:

```bash
jobl-training-train-lora

jobl-training-train-lora \
  --model=microsoft/Phi-3-mini-4k-instruct \
  --out-dir=artifacts/lora-normalize-v1 \
  --epochs=2 \
  --batch-size=2 \
  --grad-accum=8 \
  --lr=2e-4 \
  --max-seq-len=2048
```

Memory-safe run (recommended on 32GB RAM CPU host):

```bash
jobl-training-train-lora --memory-safe
```

### `jobl-training-eval-lora`

Purpose:
- Evaluates a trained LoRA adapter on `data/sft/test.jsonl`.
- Computes quality metrics and exports mismatches for manual review.

Arguments:
- `--test-jsonl`: test instruction JSONL path (default `data/sft/test.jsonl`).
- `--model`: optional base model id override.
  If omitted, evaluator auto-detects model from `<adapter-dir>/adapter_config.json`.
- `--adapter-dir`: adapter directory (default `artifacts/lora-normalize-v1/adapter`).
- `--limit`: optional max rows to evaluate, `0` means all.
- `--max-new-tokens`, `--temperature`: inference params.
- `--out-dir`: output artifacts directory (default `artifacts/lora-normalize-v1/eval`).

Metrics:
- `valid_json_rate`
- `title_exact_rate`
- `html_exact_rate`
- `title_non_empty_rate`
- `html_non_empty_rate`
- `html_allowed_tags_only_rate`

Outputs:
- `summary.json`
- `mismatches.jsonl`
- `mismatches.csv`

Example:

```bash
jobl-training-eval-lora
jobl-training-eval-lora --limit=100 --out-dir=artifacts/lora-normalize-v1/eval_smoke
```

## End-to-end quick run

```bash
jobl-training-export --out=data/raw/labeled.jsonl --limit=1000
jobl-training-split --in=data/raw/labeled.jsonl --out-dir=data/splits
jobl-training-build-jsonl
jobl-training-build-jsonl --split=val
jobl-training-train-lora
```

All scripts handle `Ctrl+C` gracefully and exit with code `130`.
