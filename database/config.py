"""
Database Configuration
=======================
Reads PostgreSQL connection parameters from environment
variables (loaded from a .env file if present).

Environment variables:
    DATABASE_HOST       default: localhost
    DATABASE_PORT       default: 5432
    DATABASE_NAME       default: smart_irrigation
    DATABASE_USER       default: postgres
    DATABASE_PASSWORD   (required — no default)

Usage:
    from database.config import DATABASE_URL
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# --------------------------------------------------
# Load .env from project root (if it exists)
# --------------------------------------------------

_project_root = Path(__file__).resolve().parent.parent
_env_path = _project_root / ".env"

if _env_path.exists():
    load_dotenv(_env_path)


# --------------------------------------------------
# Read individual variables
# --------------------------------------------------

DATABASE_HOST = os.getenv("DATABASE_HOST", "localhost")
DATABASE_PORT = os.getenv("DATABASE_PORT", "5432")
DATABASE_NAME = os.getenv("DATABASE_NAME", "smart_irrigation")
DATABASE_USER = os.getenv("DATABASE_USER", "postgres")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", "")

if not DATABASE_PASSWORD:
    print(
        "[WARNING] DATABASE_PASSWORD is not set. "
        "Set it in .env or as an environment variable."
    )


# --------------------------------------------------
# Build the SQLAlchemy connection URL
# --------------------------------------------------

DATABASE_URL = (
    f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}"
    f"@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
)
