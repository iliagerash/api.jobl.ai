"""
teacher_match.py
──────────────────
Phase 1, Day 1-2: generate scored (resume, job) pairs and hard negatives via
the vLLM-served teacher model.

Workflow:
  1. Pre-filter candidate pairs using cheap heuristics (country match +
     language match + title keyword overlap)
  2. Score each pair with the teacher: graded relevance 0.0/0.3/0.7/1.0 + reasoning
  3. For positive pairs (score >= 0.7), generate hard negatives: ask the teacher
     to explain why a plausible-but-wrong alternative doesn't match

Target: 50K scored triples (anchor, positive, negative).

Usage (on the A100, with vLLM running):
    python -u ml/scripts/teacher_match.py
    python -u ml/scripts/teacher_match.py --limit 1000                 # test run
    python -u ml/scripts/teacher_match.py --workers 4
    python -u ml/scripts/teacher_match.py --max-pairs 100000           # more candidate pairs
    python -u ml/scripts/teacher_match.py --resume

Outputs:
    ml/data/teacher_labels/match_scores.jsonl     — all scored pairs
    ml/data/teacher_labels/match_triples.jsonl    — (anchor, positive, negative) triples
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

HEARTBEAT_EVERY = 500

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WORD_RE = re.compile(r"\b\w{3,}\b")

# ── Prompts ──────────────────────────────────────────────────────────────────

MATCH_SYSTEM_PROMPT = """You are a recruitment matching expert. You evaluate how well a candidate's resume matches a job posting.

Score the match on this scale:
- 1.0 — Strong match: skills, experience level, and role align closely
- 0.7 — Good fit: most requirements met, minor gaps acceptable
- 0.3 — Weak/partial: some relevant skills but significant mismatches (wrong seniority, missing key skills, wrong domain)
- 0.0 — No match: completely different field, role, or requirements

Consider:
- Job title vs resume title/experience alignment
- Seniority fit (junior resume vs senior role, and vice versa)
- Skill overlap between required skills and resume skills
- Location and work-authorization compatibility
- Language fit (job language vs resume language)
- Contract type compatibility
- Industry/domain relevance

Return strict JSON:
{
  "score": <number 0.0|0.3|0.7|1.0>,
  "reasoning": "<1-2 sentence explanation>"
}""".strip()

HARD_NEG_SYSTEM_PROMPT = """You are a recruitment matching expert. Given a resume that is a GOOD match for a job, generate a description of a plausible but WRONG job that this resume should NOT match well.

The wrong job should be similar enough to seem relevant (same industry or adjacent field) but have a critical mismatch in one of these dimensions:
- Wrong seniority level (too junior or too senior for the candidate)
- Different required tech stack / skill set
- Different industry or domain despite similar title
- Incompatible work arrangement (remote vs onsite, different country)
- Different contract type (freelance vs permanent)

Return strict JSON:
{
  "hard_negative_title": "<plausible but wrong job title>",
  "hard_negative_description": "<2-3 sentence job description that sounds relevant but doesn't actually match>",
  "mismatch_reason": "<which dimension makes this a bad match>"
}""".strip()


# ── Text helpers ─────────────────────────────────────────────────────────────

def _clean_text(text: str | None) -> str:
    t = text or ""
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:2000]


def _title_keywords(title: str) -> set[str]:
    """Extract lowercase keywords (3+ chars) from a title."""
    return set(_WORD_RE.findall((title or "").lower()))


# ── Candidate pair generation ────────────────────────────────────────────────

def generate_candidate_pairs(
    jobs: list[dict],
    resumes: list[dict],
    max_pairs: int,
) -> list[tuple[dict, dict]]:
    """Pre-filter (resume, job) pairs using cheap heuristics."""
    print("  Building candidate pair index ...")

    # Index jobs by country + language
    job_index: dict[str, list[dict]] = defaultdict(list)
    for job in jobs:
        country = job.get("country_code") or "XX"
        lang = job.get("language_fasttext") or job.get("language_code") or "xx"
        job_index[f"{country}:{lang}"].append(job)

    pairs: list[tuple[dict, dict]] = []
    seen: set[tuple] = set()

    for resume in resumes:
        r_country = resume.get("country_code") or "XX"
        r_lang = resume.get("language_fasttext") or resume.get("language_code") or "xx"
        r_keywords = _title_keywords(resume.get("title", ""))

        # Find jobs in same country + language
        bucket_key = f"{r_country}:{r_lang}"
        candidate_jobs = job_index.get(bucket_key, [])

        if not candidate_jobs:
            continue

        # Score by title keyword overlap, pick top candidates
        scored = []
        for job in candidate_jobs:
            j_keywords = _title_keywords(job.get("title", ""))
            overlap = len(r_keywords & j_keywords)
            if overlap > 0 or random.random() < 0.1:  # some random pairs for diversity
                scored.append((overlap, job))

        scored.sort(key=lambda x: -x[0])

        for _, job in scored[:5]:  # top 5 matches per resume
            pair_key = (resume.get("id"), job.get("id"))
            if pair_key not in seen:
                seen.add(pair_key)
                pairs.append((resume, job))

            if len(pairs) >= max_pairs:
                break

        if len(pairs) >= max_pairs:
            break

    random.shuffle(pairs)
    return pairs[:max_pairs]


# ── Teacher API calls ────────────────────────────────────────────────────────

def score_pair(client, model: str, resume: dict, job: dict, max_retries: int = 3) -> dict | None:
    """Score a (resume, job) pair via the teacher model."""
    user_msg = json.dumps({
        "resume_title": resume.get("title", ""),
        "resume_text": _clean_text(resume.get("description"))[:1500],
        "resume_country": resume.get("country_code"),
        "resume_language": resume.get("language_fasttext") or resume.get("language_code"),
        "job_title": job.get("title", ""),
        "job_text": _clean_text(job.get("description"))[:1500],
        "job_country": job.get("country_code"),
        "job_language": job.get("language_fasttext") or job.get("language_code"),
    }, ensure_ascii=False)

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": MATCH_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                max_tokens=300,
                temperature=0.0,
            )
            content = resp.choices[0].message.content or ""
            result = json.loads(content)

            score = result.get("score")
            if score not in (0.0, 0.3, 0.7, 1.0):
                # Snap to nearest valid score
                valid = [0.0, 0.3, 0.7, 1.0]
                score = min(valid, key=lambda v: abs(v - float(score or 0)))
                result["score"] = score

            return result
        except Exception as e:
            if attempt >= max_retries:
                return {"score": None, "reasoning": f"error: {e}"}
            time.sleep(2 ** attempt)

    return None


def generate_hard_negative(client, model: str, resume: dict, job: dict, max_retries: int = 3) -> dict | None:
    """Generate a hard negative job for a positive (resume, job) pair."""
    user_msg = json.dumps({
        "resume_title": resume.get("title", ""),
        "resume_text": _clean_text(resume.get("description"))[:1000],
        "matched_job_title": job.get("title", ""),
        "matched_job_text": _clean_text(job.get("description"))[:1000],
    }, ensure_ascii=False)

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": HARD_NEG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=0.3,
            )
            content = resp.choices[0].message.content or ""
            return json.loads(content)
        except Exception as e:
            if attempt >= max_retries:
                return {"_error": str(e)}
            time.sleep(2 ** attempt)

    return None


# ── I/O ──────────────────────────────────────────────────────────────────────

def iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _json_default(v):
    from datetime import date, datetime
    from decimal import Decimal
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    raise TypeError(type(v).__name__)


def load_processed_pairs(path: str) -> set[tuple]:
    pairs = set()
    if not os.path.exists(path):
        return pairs
    for record in iter_jsonl(path):
        pairs.add((record.get("resume_id"), record.get("job_id")))
    return pairs


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Score (resume, job) pairs and generate hard negatives via teacher model")
    parser.add_argument("--jobs-input", default="ml/data/splits/train_pool_jobs.jsonl")
    parser.add_argument("--resumes-input", default="ml/data/splits/train_pool_resumes.jsonl")
    parser.add_argument("--scores-output", default="ml/data/teacher_labels/match_scores.jsonl")
    parser.add_argument("--triples-output", default="ml/data/teacher_labels/match_triples.jsonl")
    parser.add_argument("--api-base", default="http://localhost:8000/v1", help="vLLM server URL")
    parser.add_argument("--model", default="ml/models/base/teacher", help="Model name in vLLM")
    parser.add_argument("--max-pairs", type=int, default=100_000, help="Max candidate pairs to generate (default: 100K)")
    parser.add_argument("--limit", type=int, default=None, help="Max pairs to actually score (for testing)")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers (default: 4)")
    parser.add_argument("--resume", action="store_true", help="Skip already-scored pairs")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.scores_output), exist_ok=True)

    from openai import OpenAI
    client = OpenAI(base_url=args.api_base, api_key="not-needed")

    # Load data
    print(f"Loading jobs from {args.jobs_input} ...")
    jobs = list(iter_jsonl(args.jobs_input))
    print(f"  {len(jobs):,} jobs loaded")

    print(f"Loading resumes from {args.resumes_input} ...")
    resumes = list(iter_jsonl(args.resumes_input))
    print(f"  {len(resumes):,} resumes loaded")

    # Generate candidate pairs
    print(f"Generating candidate pairs (max {args.max_pairs:,}) ...")
    pairs = generate_candidate_pairs(jobs, resumes, args.max_pairs)
    print(f"  {len(pairs):,} candidate pairs generated")

    # Free memory
    del jobs, resumes

    # Resumability
    processed_pairs = set()
    if args.resume:
        processed_pairs = load_processed_pairs(args.scores_output)
        pairs = [(r, j) for r, j in pairs if (r.get("id"), j.get("id")) not in processed_pairs]
        print(f"  Resume mode: {len(processed_pairs):,} already scored, {len(pairs):,} remaining")

    if args.limit:
        pairs = pairs[:args.limit]
        print(f"  Limited to {len(pairs):,} pairs")

    # Phase 1: Score all pairs
    print(f"\nScoring {len(pairs):,} pairs with {args.workers} workers ...")
    total = 0
    positive_pairs: list[tuple[dict, dict, dict]] = []  # (resume, job, score_result)

    scores_mode = "a" if args.resume and os.path.exists(args.scores_output) else "w"
    with open(args.scores_output, scores_mode, encoding="utf-8") as scores_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for resume, job in pairs:
                future = executor.submit(score_pair, client, args.model, resume, job)
                futures[future] = (resume, job)

            for future in as_completed(futures):
                resume, job = futures[future]
                result = future.result()
                total += 1

                score_record = {
                    "resume_id": resume.get("id"),
                    "job_id": job.get("id"),
                    "resume_title": resume.get("title"),
                    "job_title": job.get("title"),
                    "country_code": job.get("country_code"),
                    "score": result.get("score") if result else None,
                    "reasoning": result.get("reasoning") if result else None,
                }
                scores_f.write(json.dumps(score_record, default=_json_default, ensure_ascii=False))
                scores_f.write("\n")
                scores_f.flush()

                # Collect positive pairs for hard negative generation
                if result and result.get("score") is not None and result["score"] >= 0.7:
                    positive_pairs.append((resume, job, result))

                if total % HEARTBEAT_EVERY == 0:
                    print(f"  ... {total:,} scored, {len(positive_pairs):,} positives so far")

    # Score distribution
    print(f"\nScoring complete: {total:,} pairs scored")
    print(f"  Positive pairs (score >= 0.7): {len(positive_pairs):,}")

    # Phase 2: Generate hard negatives for positive pairs
    print(f"\nGenerating hard negatives for {len(positive_pairs):,} positive pairs ...")
    triple_count = 0

    with open(args.triples_output, "w", encoding="utf-8") as triples_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for resume, job, score_result in positive_pairs:
                future = executor.submit(generate_hard_negative, client, args.model, resume, job)
                futures[future] = (resume, job, score_result)

            for future in as_completed(futures):
                resume, job, score_result = futures[future]
                hard_neg = future.result()
                triple_count += 1

                triple_record = {
                    "resume_id": resume.get("id"),
                    "job_id": job.get("id"),
                    "resume_title": resume.get("title"),
                    "resume_text": _clean_text(resume.get("description"))[:1000],
                    "job_title": job.get("title"),
                    "job_text": _clean_text(job.get("description"))[:1000],
                    "match_score": score_result.get("score"),
                    "match_reasoning": score_result.get("reasoning"),
                    "hard_negative": hard_neg,
                }
                triples_f.write(json.dumps(triple_record, default=_json_default, ensure_ascii=False))
                triples_f.write("\n")
                triples_f.flush()

                if triple_count % HEARTBEAT_EVERY == 0:
                    print(f"  ... {triple_count:,} triples generated")

    print(f"\nDone.")
    print(f"  Scored pairs:  {total:,} -> {args.scores_output}")
    print(f"  Triples:       {triple_count:,} -> {args.triples_output}")


if __name__ == "__main__":
    main()
