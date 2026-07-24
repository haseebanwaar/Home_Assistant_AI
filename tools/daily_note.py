"""Step 13 — generate the daily note + Pending-Merges into the Obsidian vault.

    python -m tools.daily_note                 # today
    python -m tools.daily_note 2026-07-23       # a specific day
    python -m tools.daily_note --vault obsidian_notes
"""
import argparse
import datetime

from dotenv import load_dotenv

from memory.stores.neo4j_store import Neo4jStore
from memory.summary.daily_summarizer import export_to_vault

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Generate the daily Obsidian note.")
    parser.add_argument("date", nargs="?", default=datetime.date.today().isoformat())
    parser.add_argument("--vault", default="obsidian_notes")
    parser.add_argument("--resolve", action="store_true",
                        help="run entity resolution first so Pending-Merges is fresh")
    args = parser.parse_args()

    with Neo4jStore() as store:
        store.verify()
        if args.resolve:
            n = store.resolve_entities()
            print(f"Entity resolution: {n} candidate(s).")
        paths = export_to_vault(store, args.vault, args.date)
        print(f"Wrote:\n  {paths['daily_note']}\n  {paths['pending_merges']}")


if __name__ == "__main__":
    main()
