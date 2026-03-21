"""
labelling/main.py
─────────────────
FastAPI web app for manually reviewing and correcting job category labels
stored in the job_labelling table.

Usage:
    python labelling/main.py
    uvicorn labelling.main:app --reload

Environment variables (.env):
    VERIFIED_LABELLING=true   Show only rows with verified = true
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import text

from app.core.config import settings
from app.db.session import SessionLocal

_verified_only: bool = settings.verified_labelling

app = FastAPI(title="Job Labelling")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

CATEGORIES = [
    (1,  "Manufacturing & Industrial Production"),
    (2,  "Automotive"),
    (3,  "Food & Beverage Manufacturing"),
    (4,  "Information Technology"),
    (5,  "Telecommunications & Internet"),
    (6,  "Construction & Infrastructure"),
    (7,  "Professional Services"),
    (8,  "Human Resources"),
    (9,  "Transportation & Logistics"),
    (10, "Healthcare & Medical Services"),
    (11, "Aerospace & Defense"),
    (12, "Financial Services & Banking"),
    (13, "Real Estate & Architecture"),
    (14, "Marketing, Advertising & Media"),
    (15, "Hospitality & Restaurants"),
    (16, "Retail & Wholesale"),
    (17, "Education & Training"),
    (18, "Energy & Natural Resources"),
    (19, "Engineering Services"),
    (20, "Nonprofit & Government"),
    (21, "Arts, Entertainment & Recreation"),
    (22, "Legal Services"),
    (23, "Science & Research"),
    (24, "Customer Service & Support"),
    (25, "Security & Surveillance"),
    (26, "Other"),
]

CATEGORY_MAP = {cid: title for cid, title in CATEGORIES}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    verified_filter = "WHERE verified = true" if _verified_only else ""
    db = SessionLocal()
    try:
        counts = db.execute(
            text(f"""
                SELECT category_id, COUNT(*) AS n,
                       SUM(CASE WHEN labelled_at IS NOT NULL THEN 1 ELSE 0 END) AS reviewed
                FROM job_labelling
                {verified_filter}
                GROUP BY category_id
                ORDER BY category_id
            """)
        ).fetchall()
    finally:
        db.close()

    count_map = {int(r[0]): {"total": int(r[1]), "reviewed": int(r[2])} for r in counts}

    categories_with_counts = [
        {
            "id": cid,
            "title": title,
            "total": count_map.get(cid, {}).get("total", 0),
            "reviewed": count_map.get(cid, {}).get("reviewed", 0),
        }
        for cid, title in CATEGORIES
    ]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "categories": categories_with_counts,
            "all_categories": CATEGORIES,
        },
    )


@app.get("/api/jobs/{category_id}")
async def get_jobs(category_id: int) -> list[dict[str, Any]]:
    db = SessionLocal()
    try:
        extra = "AND verified = true" if _verified_only else ""
        rows = db.execute(
            text(f"""
                SELECT id, title, description, description_clean, company_name,
                       country_code, original_category, email,
                       expiry_date, category_id, labelled_at, verified
                FROM job_labelling
                WHERE category_id = :cat_id {extra}
                ORDER BY labelled_at NULLS FIRST, id
            """),
            {"cat_id": category_id},
        ).fetchall()
    finally:
        db.close()

    return [
        {
            "id": r[0],
            "title": r[1],
            "description_raw": r[2] or "",
            "description_clean": r[3] or "",
            "company_name": r[4],
            "country_code": r[5],
            "original_category": r[6],
            "email": r[7],
            "expiry_date": str(r[8]) if r[8] else None,
            "category_id": r[9],
            "labelled_at": r[10].isoformat() if r[10] else None,
            "verified": bool(r[11]),
        }
        for r in rows
    ]


class LabelUpdate(BaseModel):
    category_id: int


@app.patch("/api/jobs/{job_labelling_id}/label")
async def update_label(job_labelling_id: int, body: LabelUpdate) -> dict[str, Any]:
    if body.category_id not in CATEGORY_MAP:
        raise HTTPException(status_code=422, detail="Invalid category_id")

    db = SessionLocal()
    try:
        result = db.execute(
            text("""
                UPDATE job_labelling
                SET category_id = :cat_id,
                    labelled_at  = :now
                WHERE id = :id
                RETURNING id, category_id
            """),
            {"cat_id": body.category_id, "now": datetime.now(timezone.utc), "id": job_labelling_id},
        ).fetchone()
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        db.close()

    if result is None:
        raise HTTPException(status_code=404, detail="Job labelling entry not found")

    return {"id": result[0], "category_id": result[1]}


@app.patch("/api/jobs/{job_labelling_id}/verify")
async def toggle_verify(job_labelling_id: int) -> dict[str, Any]:
    db = SessionLocal()
    try:
        result = db.execute(
            text("""
                UPDATE job_labelling
                SET verified = NOT verified
                WHERE id = :id
                RETURNING id, verified
            """),
            {"id": job_labelling_id},
        ).fetchone()
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        db.close()

    if result is None:
        raise HTTPException(status_code=404, detail="Job labelling entry not found")

    return {"id": result[0], "verified": result[1]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("labelling.main:app", host="0.0.0.0", port=8002, reload=True)
