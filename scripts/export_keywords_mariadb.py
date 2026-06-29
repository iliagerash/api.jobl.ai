"""Export keywords table to MariaDB 11.8 compatible SQL."""

import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text

from app.db.session import engine

load_dotenv()


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("keywords_mariadb.sql")

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, canonical_id, title, language_code, category_id, embedding::text "
            "FROM keywords ORDER BY id"
        )).fetchall()

    if not rows:
        print("No keywords found")
        return

    with open(output, "w") as f:
        batch_size = 50
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            f.write("INSERT INTO keywords (id, canonical_id, title, language_code, category_id, embedding) VALUES\n")
            values = []
            for row in batch:
                title_escaped = row.title.replace("\\", "\\\\").replace("'", "\\'")
                if row.embedding:
                    values.append(
                        f"({row.id}, {row.canonical_id}, '{title_escaped}', '{row.language_code}', "
                        f"{row.category_id}, VEC_FromText('{row.embedding}'))"
                    )
                else:
                    values.append(
                        f"({row.id}, {row.canonical_id}, '{title_escaped}', '{row.language_code}', "
                        f"{row.category_id}, NULL)"
                    )
            f.write(",\n".join(values))
            f.write(";\n\n")

    print(f"Wrote {len(rows)} keywords to {output}")


if __name__ == "__main__":
    main()
