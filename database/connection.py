"""
Database Connection
====================
Creates the SQLAlchemy engine and a session factory.

Usage:
    from database.connection import get_engine, SessionLocal

    # Use a session
    with SessionLocal() as session:
        session.add(record)
        session.commit()
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.config import DATABASE_URL


# --------------------------------------------------
# Engine  (single instance, reused everywhere)
# --------------------------------------------------

engine = create_engine(
    DATABASE_URL,
    echo=False,                  # set True for SQL debug logging
    pool_pre_ping=True,          # verify connections before use
)


def get_engine():
    """Return the shared SQLAlchemy engine."""
    return engine


# --------------------------------------------------
# Session factory
# --------------------------------------------------

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)
