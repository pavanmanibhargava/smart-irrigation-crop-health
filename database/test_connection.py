"""
Database Connection Test
=========================
Verifies that the application can connect to PostgreSQL,
create the tables, insert a test record, read it back,
and clean up.

Usage:
    python -m database.test_connection
"""

import sys
from datetime import datetime, timezone

from sqlalchemy import text

from database.config import DATABASE_URL
from database.connection import engine, SessionLocal
from database.models import Base, SensorReading


def test_connection():
    """
    Run a series of checks against the configured
    PostgreSQL database.
    """
    # Mask the password in printed output
    safe_url = DATABASE_URL
    at_pos = safe_url.find("@")
    colon_pos = safe_url.find(":", safe_url.find("://") + 3)
    if at_pos > 0 and colon_pos > 0:
        safe_url = safe_url[:colon_pos + 1] + "****" + safe_url[at_pos:]

    print(f"Database URL: {safe_url}")
    print()

    errors = []

    # --------------------------------------------------
    # 1. Raw connection test
    # --------------------------------------------------
    print("[1/4] Testing raw connection...")
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            value = result.scalar()
            assert value == 1, f"Expected 1, got {value}"
        print("  [PASS] Connected to PostgreSQL.")
    except Exception as exc:
        errors.append(f"Connection failed: {exc}")
        print(f"  [FAIL] {exc}")
        print("\nCannot continue — fix the connection first.")
        return False

    # --------------------------------------------------
    # 2. Create tables
    # --------------------------------------------------
    print("[2/4] Creating tables...")
    try:
        Base.metadata.create_all(bind=engine)
        table_names = sorted(Base.metadata.tables.keys())
        print(f"  [PASS] {len(table_names)} tables ready: {table_names}")
    except Exception as exc:
        errors.append(f"Table creation failed: {exc}")
        print(f"  [FAIL] {exc}")
        return False

    # --------------------------------------------------
    # 3. Insert and read back a test sensor reading
    # --------------------------------------------------
    print("[3/4] Insert + read test record...")
    try:
        with SessionLocal() as session:
            test_record = SensorReading(
                timestamp=datetime.now(timezone.utc),
                soil_moisture=42.5,
                temperature=25.0,
                humidity=60.0,
                rainfall=1.2,
            )
            session.add(test_record)
            session.commit()

            # Read it back
            record_id = test_record.id
            fetched = session.get(SensorReading, record_id)

            assert fetched is not None, "Record not found"
            assert fetched.soil_moisture == 42.5
            assert fetched.temperature == 25.0

            print(f"  [PASS] Inserted and read back: {fetched}")

            # Clean up the test record
            session.delete(fetched)
            session.commit()
            print("  [PASS] Test record cleaned up.")

    except Exception as exc:
        errors.append(f"Insert/read failed: {exc}")
        print(f"  [FAIL] {exc}")
        return False

    # --------------------------------------------------
    # 4. Verify all tables exist in the database
    # --------------------------------------------------
    print("[4/4] Verifying tables in database...")
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' "
                "ORDER BY table_name"
            ))
            db_tables = [row[0] for row in result]
            print(f"  [PASS] Tables in database: {db_tables}")
    except Exception as exc:
        errors.append(f"Table verification failed: {exc}")
        print(f"  [FAIL] {exc}")
        return False

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    print()
    if errors:
        print("RESULT: SOME TESTS FAILED")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("RESULT: ALL TESTS PASSED")
        return True


# --------------------------------------------------
# CLI entry point
# --------------------------------------------------

if __name__ == "__main__":
    print("=" * 56)
    print("  Smart Irrigation — Database Connection Test")
    print("=" * 56)
    print()

    success = test_connection()
    sys.exit(0 if success else 1)
