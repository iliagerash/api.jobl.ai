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

### `jobl-training-build-chunks`

Purpose:
- Converts SFT JSONL into chunk-level SFT JSONL for long-description training.
- Keeps full `title_normalized`, splits `description_raw`/`description_html` into chunk pairs.

Arguments:
- `--in`: input SFT JSONL path (required).
- `--out`: output chunk-level JSONL path (required).
- `--max-chars`: target max chars per chunk (default `3500`).

Example:

```bash
jobl-training-build-chunks --in=data/sft/train.jsonl --out=data/sft_chunks/train.jsonl --max-chars=3500
jobl-training-build-chunks --in=data/sft/val.jsonl --out=data/sft_chunks/val.jsonl --max-chars=3500
```

### `jobl-training-train-lora`

Purpose:
- Runs initial LoRA SFT pilot on instruction JSONL.
- Saves adapter checkpoint for local inference/evaluation.

Arguments:
- `--train-jsonl`, `--val-jsonl`: instruction datasets.
  Defaults: `data/sft_chunks/train.jsonl` and `data/sft_chunks/val.jsonl`.
- `--model`: HF base model id (default `microsoft/Phi-3-mini-4k-instruct`).
- `--out-dir`: training output directory.
- `--epochs`, `--batch-size`, `--grad-accum`, `--lr`, `--max-seq-len`: training hyperparameters.
- `--memory-safe`: CPU-safe profile for low-memory hosts.
  On CPU it switches only the implicit default model to `Qwen/Qwen2.5-0.5B-Instruct`.
  If you explicitly pass `--model`, it is preserved.
  It also forces `batch_size=1`,
  increases gradient accumulation, caps sequence length, and limits epochs for first pilot.

Outputs:
- Trainer checkpoints under `--out-dir`.
- Final adapter in `--out-dir/adapter`.

Example:

```bash
jobl-training-build-chunks --in=data/sft/train.jsonl --out=data/sft_chunks/train.jsonl --max-chars=3500
jobl-training-build-chunks --in=data/sft/val.jsonl --out=data/sft_chunks/val.jsonl --max-chars=3500

jobl-training-train-lora

jobl-training-train-lora \
  --train-jsonl=data/sft_chunks/train.jsonl \
  --val-jsonl=data/sft_chunks/val.jsonl \
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

### `jobl-training-train-lora-deepseek`

Purpose:
- Runs the same LoRA SFT pipeline but with DeepSeek R1 7B defaults in a separate command/output path.

Defaults:
- Base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (alias for requested `deepseek-r1:7b`).
- Train/val JSONL: `data/sft/train.jsonl` and `data/sft/val.jsonl` (non-chunked by default).
- Output dir: `artifacts/lora-normalize-deepseek-r1-7b`.

Example:

```bash
jobl-training-train-lora-deepseek
jobl-training-train-lora-deepseek --memory-safe
jobl-training-train-lora-deepseek --model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --epochs=2 --batch-size=2
jobl-training-train-lora-deepseek --train-jsonl=data/sft_chunks/train.jsonl --val-jsonl=data/sft_chunks/val.jsonl
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
- `--batch-size`: inference batch size (default `8`). On GPU try `4-16`.
- `--max-new-tokens`, `--temperature`: inference params.
- `--chunked`: run chunked inference for long `description_raw`.
  Default is enabled; use `--no-chunked` to disable.
- `--changed-titles-only`: evaluate only rows where `title_raw` differs from expected normalized title.
  Default is enabled; use `--all-titles` (or `--no-changed-titles-only`) to include unchanged rows.
- `--chunk-max-chars`: max chars per input chunk (default `3500`).
- `--progress-every`: progress log interval in rows (default `10`).
- `--out-dir`: output artifacts directory (default `artifacts/lora-normalize-v1/eval`).

Metrics:
- `valid_json_rate`
- `title_exact_rate`
- `html_exact_rate`
- `title_non_empty_rate`
- `html_non_empty_rate`
- `html_allowed_tags_only_rate`
- `title_similarity_avg` / `title_similarity_ge_0_8_rate`
- `html_text_similarity_avg` / `html_text_similarity_ge_0_8_rate`

Outputs:
- `summary.json`
- `mismatches.jsonl`
- `mismatches.csv`

Example:

```bash
jobl-training-eval-lora
jobl-training-eval-lora --no-chunked
jobl-training-eval-lora --all-titles
jobl-training-eval-lora --limit=100 --out-dir=artifacts/lora-normalize-v1/eval_smoke
jobl-training-eval-lora --progress-every=5
jobl-training-eval-lora --batch-size=8 --max-new-tokens=256
jobl-training-eval-lora --chunked --chunk-max-chars=3500 --batch-size=8 --max-new-tokens=1024
```

### `jobl-training-eval-lora-deepseek`

Purpose:
- Evaluates the DeepSeek adapter with DeepSeek-specific defaults.

Defaults:
- Base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- Adapter dir: `artifacts/lora-normalize-deepseek-r1-7b/adapter`
- Output dir: `artifacts/lora-normalize-deepseek-r1-7b/eval`
- Chunked inference: enabled by default (`--chunked`, disable with `--no-chunked`)
- Changed-title filtering: enabled by default (`--changed-titles-only`, disable with `--all-titles`)

Examples:

```bash
jobl-training-eval-lora-deepseek --limit=100 --progress-every=1
jobl-training-eval-lora-deepseek --all-titles --limit=100
jobl-training-eval-lora-deepseek --max-new-tokens=1024 --batch-size=4
jobl-training-eval-lora-deepseek --no-chunked --max-new-tokens=1024 --batch-size=4
jobl-training-eval-lora-deepseek --chunked --chunk-max-chars=3500 --batch-size=8 --max-new-tokens=1024
```

## End-to-end quick run

```bash
jobl-training-export --out=data/raw/labeled.jsonl --limit=1000
jobl-training-split --in=data/raw/labeled.jsonl --out-dir=data/splits
jobl-training-build-jsonl
jobl-training-build-jsonl --split=val
jobl-training-build-chunks --in=data/sft/train.jsonl --out=data/sft_chunks/train.jsonl --max-chars=3500
jobl-training-build-chunks --in=data/sft/val.jsonl --out=data/sft_chunks/val.jsonl --max-chars=3500
jobl-training-train-lora
```

All scripts handle `Ctrl+C` gracefully and exit with code `130`.
