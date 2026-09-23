"""
Database Initialization
========================
Creates all tables defined in database.models if they
do not already exist.

Usage:
    python -m database.init_db
"""

from database.connection import engine
from database.models import Base


def init_db():
    """
    Create all tables in the database.

    Uses ``checkfirst=True`` internally (the SQLAlchemy
    default), so existing tables are left untouched.
    """
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)

    # List what was created / already present
    table_names = sorted(Base.metadata.tables.keys())
    print(f"Tables ({len(table_names)}):")
    for name in table_names:
        print(f"  - {name}")

    print("\nDatabase initialization complete.")


# --------------------------------------------------
# CLI entry point
# --------------------------------------------------

if __name__ == "__main__":
    init_db()
