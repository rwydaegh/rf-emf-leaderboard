#!/usr/bin/env bash
# Full pipeline. Needs uv, the `claude` CLI (for screening) and optionally OPENALEX_API_KEY in .env.
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] && set -a && . ./.env && set +a
uv run python src/fetch.py              # 1. download candidate papers from OpenAlex  -> data/raw/
uv run python src/build.py              # 2. dedupe, merge author IDs, flag WAVES      -> data/*.parquet
uv run python src/classify.py data/screen_input.jsonl --model sonnet   # 3. LLM screening (cached) -> data/labels/
uv run python src/rank.py               # 4. leaderboards for every slice              -> results/
uv run python src/network.py            # 5. co-authorship network figure              -> results/
uv run python src/fetch_validation.py   # 6. focal author's full record (recall check) and a Haiku second opinion
uv run python src/classify.py data/validation/wout_screen.jsonl --model sonnet
uv run python src/classify.py data/validation/agreement_sample.jsonl --model haiku
uv run python src/validate.py           #    quality checks                            -> results/validation/
