"""
seed_skills_tech.py
────────────────────
Add common tech skills missing from the ESCO taxonomy.
Run AFTER seed_skills.py (does not clear existing data).

Usage:
    python sql/seed_skills_tech.py
"""

import os
import sys

from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (uri, preferred_label_en, skill_type, [(lang, label), ...])
TECH_SKILLS = [
    # Cloud platforms
    ("tech:aws", "AWS", "skill", [("en", "AWS"), ("en", "Amazon Web Services")]),
    ("tech:azure", "Azure", "skill", [("en", "Azure"), ("en", "Microsoft Azure")]),
    ("tech:gcp", "Google Cloud Platform", "skill", [("en", "GCP"), ("en", "Google Cloud"), ("en", "Google Cloud Platform")]),

    # Containers & orchestration
    ("tech:docker", "Docker", "skill", [("en", "Docker")]),
    ("tech:kubernetes", "Kubernetes", "skill", [("en", "Kubernetes"), ("en", "K8s")]),

    # Frontend
    ("tech:react", "React", "skill", [("en", "React"), ("en", "React.js"), ("en", "ReactJS")]),
    ("tech:angular", "Angular", "skill", [("en", "Angular"), ("en", "AngularJS")]),
    ("tech:vue", "Vue.js", "skill", [("en", "Vue"), ("en", "Vue.js"), ("en", "VueJS")]),
    ("tech:nextjs", "Next.js", "skill", [("en", "Next.js"), ("en", "NextJS")]),
    ("tech:typescript", "TypeScript", "skill", [("en", "TypeScript"), ("en", "TS")]),

    # Backend
    ("tech:nodejs", "Node.js", "skill", [("en", "Node.js"), ("en", "NodeJS"), ("en", "Node")]),
    ("tech:django", "Django", "skill", [("en", "Django")]),
    ("tech:flask", "Flask", "skill", [("en", "Flask")]),
    ("tech:fastapi", "FastAPI", "skill", [("en", "FastAPI")]),
    ("tech:spring", "Spring Boot", "skill", [("en", "Spring"), ("en", "Spring Boot"), ("en", "Spring Framework")]),
    ("tech:laravel", "Laravel", "skill", [("en", "Laravel")]),
    ("tech:rails", "Ruby on Rails", "skill", [("en", "Ruby on Rails"), ("en", "Rails"), ("en", "RoR")]),
    ("tech:dotnet", ".NET", "skill", [("en", ".NET"), ("en", "ASP.NET"), ("en", "dotnet")]),
    ("tech:golang", "Go", "skill", [("en", "Golang"), ("en", "Go programming")]),
    ("tech:rust", "Rust", "skill", [("en", "Rust programming"), ("en", "Rust language")]),

    # Databases
    ("tech:postgresql", "PostgreSQL", "skill", [("en", "PostgreSQL"), ("en", "Postgres")]),
    ("tech:mysql", "MySQL", "skill", [("en", "MySQL")]),
    ("tech:mongodb", "MongoDB", "skill", [("en", "MongoDB"), ("en", "Mongo")]),
    ("tech:redis", "Redis", "skill", [("en", "Redis")]),
    ("tech:elasticsearch", "Elasticsearch", "skill", [("en", "Elasticsearch"), ("en", "Elastic Search"), ("en", "ELK")]),

    # DevOps & CI/CD
    ("tech:terraform", "Terraform", "skill", [("en", "Terraform")]),
    ("tech:ansible", "Ansible", "skill", [("en", "Ansible")]),
    ("tech:jenkins", "Jenkins", "skill", [("en", "Jenkins")]),
    ("tech:cicd", "CI/CD", "skill", [("en", "CI/CD"), ("en", "continuous integration"), ("en", "continuous deployment")]),
    ("tech:git", "Git", "skill", [("en", "Git"), ("en", "GitHub"), ("en", "GitLab")]),
    ("tech:linux", "Linux", "skill", [("en", "Linux"), ("en", "Ubuntu"), ("en", "CentOS")]),

    # Data & ML
    ("tech:pandas", "Pandas", "skill", [("en", "Pandas")]),
    ("tech:numpy", "NumPy", "skill", [("en", "NumPy")]),
    ("tech:tensorflow", "TensorFlow", "skill", [("en", "TensorFlow")]),
    ("tech:pytorch", "PyTorch", "skill", [("en", "PyTorch")]),
    ("tech:spark", "Apache Spark", "skill", [("en", "Spark"), ("en", "Apache Spark"), ("en", "PySpark")]),
    ("tech:tableau", "Tableau", "skill", [("en", "Tableau")]),
    ("tech:powerbi", "Power BI", "skill", [("en", "Power BI"), ("en", "PowerBI")]),

    # Mobile
    ("tech:swift", "Swift", "skill", [("en", "Swift"), ("en", "SwiftUI")]),
    ("tech:kotlin", "Kotlin", "skill", [("en", "Kotlin")]),
    ("tech:flutter", "Flutter", "skill", [("en", "Flutter")]),
    ("tech:reactnative", "React Native", "skill", [("en", "React Native")]),

    # Other
    ("tech:graphql", "GraphQL", "skill", [("en", "GraphQL")]),
    ("tech:restapi", "REST API", "skill", [("en", "REST API"), ("en", "RESTful"), ("en", "REST")]),
    ("tech:microservices", "Microservices", "skill", [("en", "Microservices"), ("en", "microservice architecture")]),
    ("tech:agile", "Agile", "skill", [("en", "Agile"), ("en", "Scrum"), ("en", "Kanban")]),
    ("tech:jira", "Jira", "skill", [("en", "Jira")]),
    ("tech:figma", "Figma", "skill", [("en", "Figma")]),
    ("tech:salesforce", "Salesforce", "skill", [("en", "Salesforce"), ("en", "SFDC")]),
    ("tech:sap", "SAP", "skill", [("en", "SAP"), ("en", "SAP ERP")]),
]


def main():
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        skill_count = 0
        label_count = 0
        for uri, preferred_en, skill_type, labels in TECH_SKILLS:
            db.execute(
                text("INSERT INTO skills (uri, preferred_label_en, skill_type) VALUES (:uri, :label, :type) ON CONFLICT DO NOTHING"),
                {"uri": uri, "label": preferred_en, "type": skill_type},
            )
            skill_count += 1
            for lang, label in labels:
                db.execute(
                    text("INSERT INTO skill_labels (skill_uri, language_code, label) VALUES (:uri, :lang, :label) ON CONFLICT DO NOTHING"),
                    {"uri": uri, "lang": lang, "label": label},
                )
                label_count += 1
        db.commit()
        print(f"Done. {skill_count} tech skills, {label_count} labels inserted")
    finally:
        db.close()


if __name__ == "__main__":
    main()
