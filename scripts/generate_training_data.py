"""
generate_training_data.py
─────────────────────────
Generates bootstrapping training data for the LightGBM categorizer by
querying EN/FR jobs from the database and applying keyword-based heuristics
to assign a preliminary category_id.

Usage:
    python scripts/generate_training_data.py --output data/

Outputs:
    data/categories.csv           — id,title for all 26 categories
    data/categorizer_training.csv — title,original_category,description_plaintext,category_id
"""

import argparse
import os
import re
import sys

# Allow running from the project root without installing
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import csv
from bs4 import BeautifulSoup
from sqlalchemy import text
from app.db.session import SessionLocal

# ---------------------------------------------------------------------------
# Keyword rules: list of (pattern, category_id) checked in order.
# First match wins; unmatched rows fall back to 26 (Other).
# ---------------------------------------------------------------------------
_RULES: list[tuple[re.Pattern, int]] = [
    # 1  Manufacturing & Industrial Production
    (re.compile(r"\b(manufactur|production|assembl|machin|weld|fabricat|opérateur de production|soudeur)\b", re.I), 1),
    # 2  Automotive
    (re.compile(r"\b(automotiv|automobile|mécanique auto|technicien automobile|car technician|mechanic)\b", re.I), 2),
    # 3  Food & Beverage Manufacturing
    (re.compile(r"\b(food|beverage|boulanger|cuisinier|chef|cuisine|restaur|alimentation)\b", re.I), 3),
    # 4  Information Technology
    (re.compile(r"\b(software|developer|développeur|programmer|informatique|devops|data engineer|backend|frontend|fullstack|python|java\b|\.net|cloud|sre)\b", re.I), 4),
    # 5  Telecommunications & Internet
    (re.compile(r"\b(telecom|télécommunication|network engineer|ingénieur réseau|fiber|fibre|5g|isp)\b", re.I), 5),
    # 6  Construction & Infrastructure
    (re.compile(r"\b(construct|charpentier|carpenter|electrician|électricien|plumber|plombier|foreman|contremaître|civil engineer|génie civil)\b", re.I), 6),
    # 7  Professional Services
    (re.compile(r"\b(consultant|conseiller|analyst|analyste|advisor|accountant|comptable|auditor|auditeur|tax|impôt)\b", re.I), 7),
    # 8  Human Resources
    (re.compile(r"\b(human resources|ressources humaines|recruiter|recruteur|hr manager|talent acquisition|payroll|paie)\b", re.I), 8),
    # 9  Transportation & Logistics
    (re.compile(r"\b(driver|chauffeur|truck|camion|logistics|logistique|warehouse|entrepôt|dispatcher|transport)\b", re.I), 9),
    # 10 Healthcare & Medical Services
    (re.compile(r"\b(nurse|infirmier|infirmière|physician|médecin|doctor|pharmacist|pharmacien|healthcare|soins de santé|psw|préposé)\b", re.I), 10),
    # 11 Aerospace & Defense
    (re.compile(r"\b(aerospace|aérospatial|aviat|defense|défense|aircraft|avion|missile)\b", re.I), 11),
    # 12 Financial Services & Banking
    (re.compile(r"\b(bank|banque|finance|financier|investment|investissement|portfolio|trader|credit|crédit|mortgage|hypothèque)\b", re.I), 12),
    # 13 Real Estate & Architecture
    (re.compile(r"\b(real estate|immobilier|architect|architecte|urban planner|urbaniste|property manager|gestionnaire immobilier)\b", re.I), 13),
    # 14 Marketing, Advertising & Media
    (re.compile(r"\b(marketing|publicité|advertising|media|brand|marque|content|contenu|seo|sem|social media|journalist|journaliste)\b", re.I), 14),
    # 15 Hospitality & Restaurants
    (re.compile(r"\b(hotel|hôtel|hospitality|hôtellerie|server|serveur|bartender|barman|housekeep|femme de chambre|front desk)\b", re.I), 15),
    # 16 Retail & Wholesale
    (re.compile(r"\b(retail|détail|store|magasin|cashier|caissier|sales associate|associé aux ventes|merchandis)\b", re.I), 16),
    # 17 Education & Training
    (re.compile(r"\b(teacher|enseignant|professor|professeur|educator|éducateur|tutor|tuteur|school|école|university|université|trainer|formateur)\b", re.I), 17),
    # 18 Energy & Natural Resources
    (re.compile(r"\b(energy|énergie|oil|pétrole|gas|gaz|mining|mines|renewable|renouvelable|électricité|electricity|solar|wind)\b", re.I), 18),
    # 19 Engineering Services
    (re.compile(r"\b(engineer|ingénieur|mechanical|mécanique|structural|électrique|electrical engineer|process engineer|ingénieur de procédés)\b", re.I), 19),
    # 20 Nonprofit & Government
    (re.compile(r"\b(nonprofit|organisme sans but lucratif|government|gouvernement|public sector|secteur public|social worker|travailleur social|ngo|ong)\b", re.I), 20),
    # 21 Arts, Entertainment & Recreation
    (re.compile(r"\b(artist|artiste|design|designer|graphic|graphique|music|musique|film|entertainment|divertissement|sport|recreation)\b", re.I), 21),
    # 22 Legal Services
    (re.compile(r"\b(lawyer|avocat|attorney|legal|juridique|paralegal|notary|notaire|compliance|conformité)\b", re.I), 22),
    # 23 Science & Research
    (re.compile(r"\b(scientist|scientifique|research|recherche|laboratory|laboratoire|biologist|biologiste|chemist|chimiste|physicist|physicien)\b", re.I), 23),
    # 24 Customer Service & Support
    (re.compile(r"\b(customer service|service client|support agent|call center|centre d'appel|helpdesk|help desk|customer success)\b", re.I), 24),
    # 25 Security & Surveillance
    (re.compile(r"\b(security guard|agent de sécurité|surveillance|guard|gardien|cybersecurity|cybersécurité|information security)\b", re.I), 25),
]

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


def _assign_category(title: str, description_plaintext: str) -> int:
    text = f"{title} {description_plaintext[:500]}"
    for pattern, cat_id in _RULES:
        if pattern.search(text):
            return cat_id
    return 26  # Other


def _strip_html(html: str) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "lxml").get_text(separator=" ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate categorizer training data from DB")
    parser.add_argument("--output", default="data/", help="Output directory (default: data/)")
    parser.add_argument("--limit", type=int, default=100_000, help="Max rows to fetch (default: 100000)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    categories_path = os.path.join(args.output, "categories.csv")
    training_path = os.path.join(args.output, "categorizer_training.csv")

    # Write categories CSV
    with open(categories_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title"])
        writer.writerows(CATEGORIES)
    print(f"Wrote {len(CATEGORIES)} categories to {categories_path}")

    # Query jobs
    db = SessionLocal()
    try:
        rows = db.execute(
            text("""
                SELECT title, title_clean, description
                FROM jobs
                WHERE language_code IN ('en', 'fr')
                  AND (title IS NOT NULL OR title_clean IS NOT NULL)
                LIMIT :limit
            """),
            {"limit": args.limit},
        ).fetchall()
    finally:
        db.close()

    print(f"Fetched {len(rows)} jobs from DB")

    written = 0
    with open(training_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "original_category", "description_plaintext", "category_id"])
        for title, title_clean, description in rows:
            effective_title = title_clean or title or ""
            desc_plain = _strip_html(description or "")
            category_id = _assign_category(effective_title, desc_plain)
            writer.writerow([effective_title, "", desc_plain[:1000], category_id])
            written += 1

    print(f"Wrote {written} training rows to {training_path}")


if __name__ == "__main__":
    main()
