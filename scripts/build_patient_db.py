"""Build the Data agent's SQLite database from Synthea CSV output.

Synthea (github.com/synthetichealth/synthea, Apache 2.0) generates realistic but
entirely fictional patient records. The database is generated with:

    java -jar synthea-with-dependencies.jar -s 599 -cs 599 -p 1000 \
        --exporter.csv.export=true --exporter.fhir.export=false \
        --exporter.hospital.fhir.export=false --exporter.practitioner.fhir.export=false \
        --exporter.years_of_history=10 California "Los Angeles"

    python scripts/build_patient_db.py output/csv data/patients.db

Only population-level fields are kept: no names, addresses, or ID numbers,
even though they are synthetic, and patient UUIDs become small integers. Observations are reduced to each patient's
latest numeric lab and vital-sign values to keep the file small.
"""

import csv
import sqlite3
import sys
from pathlib import Path

TABLES = {
    "patients": {
        "source": "patients.csv",
        "columns": {"Id": "id", "BIRTHDATE": "birthdate", "DEATHDATE": "deathdate", "GENDER": "gender",
                    "RACE": "race", "ETHNICITY": "ethnicity", "MARITAL": "marital", "CITY": "city",
                    "COUNTY": "county", "ZIP": "zip"},
    },
    "conditions": {
        "source": "conditions.csv",
        "columns": {"START": "start", "STOP": "stop", "PATIENT": "patient", "CODE": "code",
                    "DESCRIPTION": "description"},
    },
    "medications": {
        "source": "medications.csv",
        "columns": {"START": "start", "STOP": "stop", "PATIENT": "patient", "CODE": "code",
                    "DESCRIPTION": "description", "REASONDESCRIPTION": "reason", "DISPENSES": "dispenses"},
    },
    "encounters": {
        "source": "encounters.csv",
        "columns": {"START": "start", "STOP": "stop", "PATIENT": "patient",
                    "ENCOUNTERCLASS": "encounter_class", "DESCRIPTION": "description",
                    "REASONDESCRIPTION": "reason"},
    },
    "immunizations": {
        "source": "immunizations.csv",
        "columns": {"DATE": "date", "PATIENT": "patient", "CODE": "code", "DESCRIPTION": "description"},
    },
    "allergies": {
        "source": "allergies.csv",
        "columns": {"START": "start", "STOP": "stop", "PATIENT": "patient", "DESCRIPTION": "description",
                    "TYPE": "type", "CATEGORY": "category", "SEVERITY1": "severity"},
    },
}


def column_type(column: str) -> str:
    if column == "id":
        return "INTEGER PRIMARY KEY"
    return "INTEGER" if column in {"patient", "dispenses"} else "TEXT"


def load_table(db: sqlite3.Connection, name: str, spec: dict, csv_dir: Path, patient_ids: dict) -> int:
    """Load one CSV, replacing Synthea's UUIDs with small integer patient IDs."""
    columns = spec["columns"]
    db.execute(f"CREATE TABLE {name} ({', '.join(f'{c} {column_type(c)}' for c in columns.values())})")
    with open(csv_dir / spec["source"], encoding="utf-8", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            values = {dst: row[src] or None for src, dst in columns.items()}
            if name == "patients":
                values["id"] = patient_ids.setdefault(row["Id"], len(patient_ids) + 1)
            else:
                values["patient"] = patient_ids[row["PATIENT"]]
            rows.append(tuple(values.values()))
    db.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' for _ in columns)})", rows)
    return len(rows)


def load_latest_observations(db: sqlite3.Connection, csv_dir: Path, patient_ids: dict) -> int:
    latest: dict[tuple[str, str], dict] = {}
    with open(csv_dir / "observations.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["TYPE"] != "numeric" or row["CATEGORY"] not in {"laboratory", "vital-signs"}:
                continue
            key = (row["PATIENT"], row["CODE"])
            if key not in latest or row["DATE"] > latest[key]["DATE"]:
                latest[key] = row
    db.execute("CREATE TABLE latest_observations (patient INTEGER, date TEXT, category TEXT, code TEXT, "
               "description TEXT, value REAL, units TEXT)")
    db.executemany(
        "INSERT INTO latest_observations VALUES (?, ?, ?, ?, ?, ?, ?)",
        ((patient_ids[r["PATIENT"]], r["DATE"], r["CATEGORY"], r["CODE"], r["DESCRIPTION"], float(r["VALUE"]), r["UNITS"])
         for r in latest.values()),
    )
    return len(latest)


def main(csv_dir: Path, out_path: Path) -> None:
    out_path.unlink(missing_ok=True)
    db = sqlite3.connect(out_path)
    patient_ids: dict[str, int] = {}
    for name, spec in TABLES.items():  # patients first, so every UUID gets an ID
        print(f"{name:20} {load_table(db, name, spec, csv_dir, patient_ids):>8,} rows")
    print(f"{'latest_observations':20} {load_latest_observations(db, csv_dir, patient_ids):>8,} rows")
    for table in ("conditions", "medications", "encounters", "immunizations", "allergies", "latest_observations"):
        db.execute(f"CREATE INDEX idx_{table}_patient ON {table}(patient)")
    db.commit()
    db.execute("VACUUM")
    db.close()
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
