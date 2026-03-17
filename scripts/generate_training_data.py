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
# Keyword rules: list of (pattern, category_id, languages) checked in order.
# languages=None means the rule applies to all languages.
# First match wins; unmatched rows fall back to 26 (Other).
# ---------------------------------------------------------------------------
_RULES: list[tuple[re.Pattern, int, set[str] | None]] = [
    # 1  Manufacturing & Industrial Production — assembl\w* title-only ("assembly process/subassembly" common in engineering descriptions)
    (re.compile(r"\b(assembl\w*)\b", re.I), 1, None, True),
    (re.compile(r"\b(manufactur\w*|production|machin\w*|weld\w*|fabricat\w*|opérateur de production|soudeur)\b", re.I), 1, None),
    # 2  Automotive
    (re.compile(r"\b(automotive|automobile|mécanique auto|technicien automobile|car technician|auto mechanic|vehicle mechanic|vehicle inspection|pdi inspector)\b", re.I), 2, None),
    # 3  Food & Beverage Manufacturing — generic terms title-only to avoid matching food-service/hospitality jobs
    (re.compile(r"\b(restaur\w*|food|beverage)\b", re.I), 3, None, True),
    (re.compile(r"\b(boulang\w*|alimentation|traiteur|pâtissier|brasseur)\b", re.I), 3, None),
    # 4  Information Technology
    (re.compile(r"\b(software|developer|développeur|programmer|informatique|devops|data engineer|backend|frontend|fullstack|python|java\b|\.net|cloud|sre|technical writer|rédacteur technique|it support|systems? analyst|product manager|product owner|scrum master|agile coach|saas)\b", re.I), 4, None),
    # 5  Telecommunications & Internet
    (re.compile(r"\b(telecom\w*|télécommunication\w*|network engineer|ingénieur réseau|fiber|fibre|5g|isp)\b", re.I), 5, None),
    # 6  Construction & Infrastructure
    (re.compile(r"\b(construction (?:worker|manager|supervisor|site|project|estimator|superintendent)|general contractor|charpentier|carpenter|electrician|électricien|plumber|plombier|foreman|contremaître|civil engineer|génie civil|maçon|mason|ironworker|ferrailleur)\b", re.I), 6, None),
    # 7  Professional Services — "tax/impôt" title-only (appears in compliance clauses of any job)
    (re.compile(r"\b(tax|impôt)\b", re.I), 7, None, True),
    (re.compile(r"\b(consultant|conseiller|business analyst|analyste d.affaires|advisor)\b", re.I), 7, None),
    # 8  Human Resources
    (re.compile(r"\b(human resources|ressources humaines|recruiter|recruteur|hr manager|talent acquisition|payroll|paie)\b", re.I), 8, None),
    # 9  Transportation & Logistics — bare "transport" is title-only (appears in benefits sections)
    (re.compile(r"\b(transport)\b", re.I), 9, None, True),
    (re.compile(r"\b(driver|chauffeur|truck|camion|logistics|logistique|warehouse|entrepôt|dispatcher)\b", re.I), 9, None),
    # 8  HR — early title-only check so hospital HR roles beat Healthcare body keywords
    (re.compile(r"\b(HR|RH)\b"), 8, None, True),
    # 25 Security — title-only early check so security roles at hospitals beat Healthcare body keywords
    (re.compile(r"\b(security officer|security guard|agent de sécurité)\b", re.I), 25, None, True),
    # 10 Healthcare & Medical Services
    (re.compile(r"\b(nurs(?:e|es|ing)|infirmier|infirmière|physicians?|médecin|doctors?|pharmacist|pharmacien|soins de santé|psw|préposé aux bénéficiaires|hospital|hôpital|surgeon|surgery|surgical|clinic(?:al)?|patient care|operating room|salle d.opération|sterilization|sterile processing|decontamination|endoscop\w*|radiology|pathology|cardiology|oncology|optometrist|optometry|eyecare|eye care|opticien|dentist|dental|physiotherap\w*|physical therap\w*|kinésithérapeut\w*|orthopedic|orthopédie|rehabilitation|réadaptation|occupational therap\w*)\b", re.I), 10, None),
    # 11 Aerospace & Defense
    (re.compile(r"\b(aerospace|aérospatial\w*|aviat\w*|defense|défense|aircraft|avion|missile)\b", re.I), 11, None),
    # 12 Financial Services & Banking
    # 12 Financial Services — investment/portfolio title-only (appear in marketing/product descriptions)
    (re.compile(r"\b(investment|investissement|portfolio)\b", re.I), 12, None, True),
    (re.compile(r"\b(bank|banque|finance|financier|trader|credit|crédit|mortgage|hypothèque|accountant|comptable|auditor|auditeur|bookkeeper|trésorier)\b", re.I), 12, None),
    # 13 Real Estate & Architecture
    (re.compile(r"\b(real estate|immobilier|architect|architecte|urban planner|urbaniste|property manager|gestionnaire immobilier)\b", re.I), 13, None),
    # 14 Marketing, Advertising & Media — generic nouns title-only; specific terms match anywhere
    (re.compile(r"\b(brand|marque|content|contenu|media)\b", re.I), 14, None, True),
    (re.compile(r"\b(marketing|publicité|advertising|seo|sem|social media|journalist|journaliste|copywriter|rédacteur)\b", re.I), 14, None),
    # 15 Hospitality & Restaurants
    (re.compile(r"\b(hotel|hôtel|hospitality|hôtellerie|server|serveur|bartender|barman|dishwasher|plongeur|housekeep\w*|femme de chambre|front desk|kitchen|cuisinier|cuisine|food service|restauration|guest experience|check-in|check-out|furnished rental|furnished apartment)\b", re.I), 15, None),
    # 15 Hospitality — EN only: "chef" and "cook" refer to kitchen roles
    (re.compile(r"\b(chef|cook)\b", re.I), 15, {"en"}),
    # 16 Retail & Wholesale — bare "retail" checked in title only (too noisy in descriptions)
    (re.compile(r"\b(retail)\b", re.I), 16, None, True),
    (re.compile(r"\b(retailer|retailing|détail|magasin|cashier|caissier|sales associate|associé aux ventes|merchandis\w*)\b", re.I), 16, None),
    # 17 Education & Training
    (re.compile(r"\b(teacher|enseignant\w*|enseign\w*|professor|professeur|educator|éducateur|tutor|tuteur|school|école|university|université|trainer|formateur|cours particuliers|soutien scolaire|aide aux devoirs|pédagogue|pedagog\w*)\b", re.I), 17, None),
    # 18 Energy & Natural Resources
    (re.compile(r"\b(energy|énergie|oil|pétrole|gas|gaz|mining|mines|renewable|renouvelable|électricité|electricity|solar|wind)\b", re.I), 18, None),
    # 19 Engineering Services
    (re.compile(r"\b(engineer\w*|ingénieur\w*|mechanical|mécanique|structural|électrique|electrical engineer|process engineer|ingénieur de procédés)\b", re.I), 19, None),
    # 20 Nonprofit & Government
    (re.compile(r"\b(nonprofit|not-for-profit|organisme sans but lucratif|government|gouvernement|public sector|secteur public|social worker|travailleur social|ngo|ingo|ong|civil service|civil servant|public administration|administration publique|municipal(?:ity|ité)?|federal|ministry|ministère|county government|city of \w+|fonctionnaire|fonction publique|service public|collectivité\w*|mairie|préfecture|agent territorial|humanitarian|humanitaire|non.governmental|organisation internationale)\b", re.I), 20, None),
    # 21 Arts, Entertainment & Recreation
    (re.compile(r"\b(artist|artiste|graphic designer|graphiste|music|musique|film|entertainment|divertissement|sport|recreation)\b", re.I), 21, None),
    # 22 Legal Services — "compliance/conformité" title-only (too common in non-legal descriptions)
    (re.compile(r"\b(compliance|conformité)\b", re.I), 22, None, True),
    (re.compile(r"\b(lawyer|avocat|attorney|legal|juridique|paralegal|notary|notaire)\b", re.I), 22, None),
    # 23 Science & Research
    (re.compile(r"\b(scientist|scientifique|research|recherche|laboratory|laboratoire|biologist|biologiste|chemist|chimiste|physicist|physicien)\b", re.I), 23, None),
    # 24 Customer Service & Support
    (re.compile(r"\b(customer service|service client|support agent|call center|centre d'appel|helpdesk|help desk|customer success)\b", re.I), 24, None),
    # 25 Security & Surveillance
    (re.compile(r"\b(security officer|security guard|agent de sécurité|surveillance|guard|gardien|cybersecurity|cybersécurité|information security)\b", re.I), 25, None),
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


def _assign_category(title: str, description_plaintext: str, language: str) -> int:
    full_text = f"{title} {description_plaintext[:2000]}"
    for rule in _RULES:
        pattern, cat_id, languages = rule[0], rule[1], rule[2]
        title_only = rule[3] if len(rule) > 3 else False
        if languages is not None and language not in languages:
            continue
        if pattern.search(title if title_only else full_text):
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
                SELECT title, title_clean, description, language_code
                FROM jobs
                WHERE language_code IN ('en', 'fr')
                  AND (title IS NOT NULL OR title_clean IS NOT NULL)
                ORDER BY RANDOM()
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
        for title, title_clean, description, language_code in rows:
            effective_title = title_clean or title or ""
            desc_plain = _strip_html(description or "")
            category_id = _assign_category(effective_title, desc_plain, language_code or "en")
            writer.writerow([effective_title, "", desc_plain[:1000], category_id])
            written += 1

    print(f"Wrote {written} training rows to {training_path}")


if __name__ == "__main__":
    main()
