"""
Stress test: simulate production load.

- 1000 random jobs through /v1/process (concurrent)
- 10 random resumes through /v1/extract (concurrent with /process)

Usage:
    python -u tests/test_stress.py
    python -u tests/test_stress.py --process-count 500 --extract-count 5
    python -u tests/test_stress.py --process-workers 8 --extract-workers 2
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROCESS_URL = "http://localhost:8001/v1/process"
EXTRACT_URL = "http://localhost:8002/v1/extract"


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
    return [{"title": r[0] or "", "description": r[1] or "", "type": "resume"} for r in rows]


def call_process(client: httpx.Client, job: dict) -> dict:
    start = time.time()
    try:
        resp = client.post(PROCESS_URL, json=job, timeout=30)
        ms = (time.time() - start) * 1000
        if resp.status_code == 200:
            data = resp.json()
            has_embedding = data.get("embedding") is not None
            has_skills = len(data.get("skills") or []) > 0
            return {"status": "ok", "ms": ms, "has_embedding": has_embedding, "has_skills": has_skills}
        return {"status": "error", "ms": ms, "code": resp.status_code}
    except Exception as e:
        ms = (time.time() - start) * 1000
        return {"status": "error", "ms": ms, "error": str(e)}


def call_extract(client: httpx.Client, resume: dict) -> dict:
    start = time.time()
    try:
        resp = client.post(EXTRACT_URL, json=resume, timeout=120)
        ms = (time.time() - start) * 1000
        if resp.status_code == 200:
            data = resp.json()
            has_extraction = "_error" not in (data.get("extraction") or {})
            return {"status": "ok", "ms": ms, "has_extraction": has_extraction}
        return {"status": "error", "ms": ms, "code": resp.status_code}
    except Exception as e:
        ms = (time.time() - start) * 1000
        return {"status": "error", "ms": ms, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Stress test API endpoints")
    parser.add_argument("--process-count", type=int, default=1000)
    parser.add_argument("--extract-count", type=int, default=10)
    parser.add_argument("--process-workers", type=int, default=8)
    parser.add_argument("--extract-workers", type=int, default=2)
    args = parser.parse_args()

    print(f"Loading {args.process_count} random jobs ...")
    jobs = load_random_jobs(args.process_count)
    print(f"  {len(jobs)} jobs loaded")

    print(f"Loading {args.extract_count} random resumes ...")
    resumes = load_random_resumes(args.extract_count)
    print(f"  {len(resumes)} resumes loaded")

    process_results = []
    extract_results = []

    print(f"\nStarting stress test:")
    print(f"  /process: {len(jobs)} jobs, {args.process_workers} workers")
    print(f"  /extract: {len(resumes)} resumes, {args.extract_workers} workers")
    print()

    start_time = time.time()

    # Run both concurrently
    with httpx.Client() as process_client, httpx.Client() as extract_client:
        with ThreadPoolExecutor(max_workers=args.process_workers + args.extract_workers) as executor:
            # Submit process jobs
            process_futures = {
                executor.submit(call_process, process_client, job): ("process", i)
                for i, job in enumerate(jobs)
            }
            # Submit extract jobs
            extract_futures = {
                executor.submit(call_extract, extract_client, resume): ("extract", i)
                for i, resume in enumerate(resumes)
            }

            all_futures = {**process_futures, **extract_futures}
            done_process = 0
            done_extract = 0

            for future in as_completed(all_futures):
                kind, idx = all_futures[future]
                result = future.result()

                if kind == "process":
                    process_results.append(result)
                    done_process += 1
                    if done_process % 100 == 0:
                        print(f"  /process: {done_process}/{len(jobs)} done")
                else:
                    extract_results.append(result)
                    done_extract += 1
                    print(f"  /extract: {done_extract}/{len(resumes)} done ({result['ms']:.0f}ms)")

    total_time = time.time() - start_time

    # Process stats
    print(f"\n{'='*60}")
    print(f"RESULTS (total time: {total_time:.1f}s)")
    print(f"{'='*60}")

    if process_results:
        ok = [r for r in process_results if r["status"] == "ok"]
        errors = [r for r in process_results if r["status"] == "error"]
        latencies = [r["ms"] for r in ok]
        with_embedding = sum(1 for r in ok if r.get("has_embedding"))
        with_skills = sum(1 for r in ok if r.get("has_skills"))

        print(f"\n  /v1/process ({len(process_results)} requests):")
        print(f"    OK: {len(ok)}, Errors: {len(errors)}")
        if latencies:
            latencies.sort()
            print(f"    Latency avg: {sum(latencies)/len(latencies):.0f}ms")
            print(f"    Latency p50: {latencies[len(latencies)//2]:.0f}ms")
            print(f"    Latency p95: {latencies[int(len(latencies)*0.95)]:.0f}ms")
            print(f"    Latency p99: {latencies[int(len(latencies)*0.99)]:.0f}ms")
            print(f"    Throughput:  {len(ok)/total_time:.1f} req/s")
        print(f"    With embedding: {with_embedding}/{len(ok)}")
        print(f"    With skills:    {with_skills}/{len(ok)}")

    if extract_results:
        ok = [r for r in extract_results if r["status"] == "ok"]
        errors = [r for r in extract_results if r["status"] == "error"]
        latencies = [r["ms"] for r in ok]

        print(f"\n  /v1/extract ({len(extract_results)} requests):")
        print(f"    OK: {len(ok)}, Errors: {len(errors)}")
        if latencies:
            print(f"    Latency avg: {sum(latencies)/len(latencies):.0f}ms")
            print(f"    Latency min: {min(latencies):.0f}ms")
            print(f"    Latency max: {max(latencies):.0f}ms")

    # Memory check
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
