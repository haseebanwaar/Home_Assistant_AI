"""Step 7 — Neo4j smoke test.

Connects to the local Neo4j instance, applies constraints + indexes, MERGEs a
single :Day node, reads it back, and confirms the round-trip. Idempotent — run
it as many times as you like.

    python -m tools.neo4j_smoke
"""
import datetime
import sys

from dotenv import load_dotenv

from memory.stores.neo4j_store import Neo4jStore

load_dotenv()


def main():
    try:
        store = Neo4jStore()
    except Exception as exc:
        print(f"Failed to construct driver: {exc}", file=sys.stderr)
        return 1

    try:
        store.verify()
        print(f"Connected: {store.uri} (database={store.database})")

        store.apply_schema()
        constraints = store.list_constraints()
        indexes = store.list_indexes()
        print(f"Constraints: {len(constraints)} | Indexes: {len(indexes)}")
        for c in constraints:
            print(f"  - {c.get('name')}: {c.get('labelsOrTypes')} {c.get('properties')}")

        today = datetime.date.today().isoformat()
        created = store.merge_day(today)
        print(f"\nMERGE :Day -> {created}")

        # Read it back independently.
        rows = store.run("MATCH (d:Day {date: $date}) RETURN d.date AS date", date=today)
        assert rows and rows[0]["date"] == today, "round-trip failed"
        print(f"Read back :Day {{date: {rows[0]['date']}}} — round-trip OK")

        # Idempotency: MERGE again, count stays 1.
        store.merge_day(today)
        cnt = store.run("MATCH (d:Day {date: $date}) RETURN count(d) AS n", date=today)[0]["n"]
        print(f":Day count for {today}: {cnt} (expect 1)")

    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()

    print("\nSmoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
