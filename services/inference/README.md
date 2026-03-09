# Job Inference Service (`services/inference`)

Latency-focused local inference benchmark for normalization models.

## Install

```bash
cd /home/<user>/Jobl/api.jobl.ai/services/inference
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
cp .env.example .env
```

## Benchmark inference speed

Defaults assume artifacts from `services/training`:
- input: `../training/data/sft/test.jsonl`
- adapter: `../training/artifacts/lora-normalize-v1/adapter`
- base model is auto-detected from adapter metadata

Run full normalization benchmark (title + description):

```bash
jobl-infer-benchmark --task=full --limit=30 --warmup=2 --progress-every=1
```

Run title-only benchmark (production-fast path estimate):

```bash
jobl-infer-benchmark --task=title --limit=50 --warmup=3 --progress-every=1
```

Optional flags:
- `--max-new-tokens=...` (default: `512` for `full`, `64` for `title`)
- `--temperature=0`
- `--model=...` to override auto-detected base model
- `--out=artifacts/benchmark_summary.json`

## Output

Writes a JSON summary file with:
- latency stats: `min`, `p50`, `p95`, `max`, `avg`
- throughput: `rows_per_sec`, `output_tokens_per_sec`, `avg_output_tokens_per_row`
- model/adapter/task/settings used for the benchmark
