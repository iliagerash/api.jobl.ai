"""
Stress test: simulate production load.

- 1000 random jobs through /v1/process (concurrent)
- 200 random jobs through /v1/embed as jobs (concurrent)
- 100 random resumes through /v1/embed as resumes (concurrent)

All three run simultaneously.

Usage:
    python -u tests/test_stress.py
    python -u tests/test_stress.py --process-count 500 --embed-jobs 100 --embed-resumes 50
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://localhost:8001/v1"


def load_random_jobs(limit: int) -> list[dict]:
    from sqlalchemy import text
    from app.db.session import SessionLocal
    db = SessionLocal()
    rows = db.execute(
        text("SELECT title, description FROM jobs ORDER BY RANDOM() LIMIT :limit"),
        {"limit": limit},
    ).fetchall()
    db.close()
    return [{"title": r[0] or "", "description": r[1] or ""} for r in rows]


def load_random_resumes(limit: int) -> list[dict]:
    from sqlalchemy import text
    from app.db.session import SessionLocal
    db = SessionLocal()
    rows = db.execute(
        text("SELECT title, description FROM resumes ORDER BY RANDOM() LIMIT :limit"),
        {"limit": limit},
    ).fetchall()
    db.close()
    return [{"title": r[0] or "", "description": r[1] or ""} for r in rows]


def call_process(client: httpx.Client, job: dict) -> dict:
    start = time.time()
    try:
        resp = client.post(f"{BASE_URL}/process", json=job, timeout=30)
        ms = (time.time() - start) * 1000
        if resp.status_code == 200:
            data = resp.json()
            return {
                "status": "ok", "ms": ms,
                "has_embedding": data.get("embedding") is not None,
                "has_skills": len(data.get("skills") or []) > 0,
            }
        return {"status": "error", "ms": ms, "code": resp.status_code}
    except Exception as e:
        return {"status": "error", "ms": (time.time() - start) * 1000, "error": str(e)}


def call_embed(client: httpx.Client, text: str, record_type: str) -> dict:
    start = time.time()
    try:
        resp = client.post(f"{BASE_URL}/embed", json={"text": text, "type": record_type}, timeout=30)
        ms = (time.time() - start) * 1000
        if resp.status_code == 200:
            data = resp.json()
            return {
                "status": "ok", "ms": ms,
                "dim": len(data.get("embedding", [])),
            }
        return {"status": "error", "ms": ms, "code": resp.status_code}
    except Exception as e:
        return {"status": "error", "ms": (time.time() - start) * 1000, "error": str(e)}


def print_stats(label: str, results: list[dict], total_time: float):
    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] == "error"]
    latencies = sorted([r["ms"] for r in ok])

    print(f"\n  {label} ({len(results)} requests):")
    print(f"    OK: {len(ok)}, Errors: {len(errors)}")
    if latencies:
        print(f"    Latency avg: {sum(latencies)/len(latencies):.0f}ms")
        print(f"    Latency p50: {latencies[len(latencies)//2]:.0f}ms")
        print(f"    Latency p95: {latencies[int(len(latencies)*0.95)]:.0f}ms")
        print(f"    Latency p99: {latencies[int(len(latencies)*0.99)]:.0f}ms")
        print(f"    Throughput:  {len(ok)/total_time:.1f} req/s")

    # Extra stats for /process
    if ok and "has_embedding" in ok[0]:
        with_embedding = sum(1 for r in ok if r.get("has_embedding"))
        with_skills = sum(1 for r in ok if r.get("has_skills"))
        print(f"    With embedding: {with_embedding}/{len(ok)}")
        print(f"    With skills:    {with_skills}/{len(ok)}")


def main():
    parser = argparse.ArgumentParser(description="Stress test API endpoints")
    parser.add_argument("--process-count", type=int, default=1000)
    parser.add_argument("--embed-jobs", type=int, default=200)
    parser.add_argument("--embed-resumes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    total_needed = max(args.process_count, args.embed_jobs)
    print(f"Loading {total_needed} random jobs ...")
    jobs = load_random_jobs(total_needed)
    print(f"  {len(jobs)} jobs loaded")

    print(f"Loading {args.embed_resumes} random resumes ...")
    resumes = load_random_resumes(args.embed_resumes)
    print(f"  {len(resumes)} resumes loaded")

    process_results = []
    embed_job_results = []
    embed_resume_results = []

    print(f"\nStarting stress test ({args.workers} workers):")
    print(f"  /process:       {args.process_count} jobs")
    print(f"  /embed (jobs):   {args.embed_jobs} jobs")
    print(f"  /embed (resumes): {args.embed_resumes} resumes")
    print()

    start_time = time.time()

    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}

            # Submit /process
            for i in range(args.process_count):
                f = executor.submit(call_process, client, jobs[i])
                futures[f] = "process"

            # Submit /embed for jobs
            for i in range(args.embed_jobs):
                text = f"{jobs[i]['title']} {jobs[i]['description']}"[:2000]
                f = executor.submit(call_embed, client, text, "job")
                futures[f] = "embed_job"

            # Submit /embed for resumes
            for i in range(args.embed_resumes):
                text = f"{resumes[i]['title']} {resumes[i]['description']}"[:2000]
                f = executor.submit(call_embed, client, text, "resume")
                futures[f] = "embed_resume"

            done_counts = {"process": 0, "embed_job": 0, "embed_resume": 0}
            total_tasks = len(futures)

            for future in as_completed(futures):
                kind = futures[future]
                result = future.result()
                done_counts[kind] += 1

                if kind == "process":
                    process_results.append(result)
                elif kind == "embed_job":
                    embed_job_results.append(result)
                else:
                    embed_resume_results.append(result)

                done_total = sum(done_counts.values())
                if done_total % 100 == 0:
                    print(f"  ... {done_total}/{total_tasks} done (process={done_counts['process']}, embed_jobs={done_counts['embed_job']}, embed_resumes={done_counts['embed_resume']})")

    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"RESULTS (total time: {total_time:.1f}s)")
    print(f"{'='*60}")

    print_stats("/v1/process", process_results, total_time)
    print_stats("/v1/embed (jobs)", embed_job_results, total_time)
    print_stats("/v1/embed (resumes)", embed_resume_results, total_time)

    # Memory
    print(f"\n  Memory:")
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemTotal:", "MemAvailable:")):
                    key, val = line.split(":")
                    mb = int(val.strip().split()[0]) / 1024
                    print(f"    {key}: {mb:.0f} MB")
    except Exception:
        pass

    print()


if __name__ == "__main__":
    main()
