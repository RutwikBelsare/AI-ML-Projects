"""
Every ORM model inherits from this Base. Alembic's env.py points at
Base.metadata to auto-generate migrations, so every model file must be
imported somewhere that eventually gets imported before autogenerate runs
(see app/models/__init__.py).
"""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
