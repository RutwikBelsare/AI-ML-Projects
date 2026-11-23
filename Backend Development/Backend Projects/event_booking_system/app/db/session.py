"""
Async DB engine + session factory.

pool_pre_ping=True checks a connection is still alive before handing it
out, which avoids the classic "MySQL/Postgres server has gone away" error
on long-lived connection pools. Pool size settings here are what you'd
tune under real load (Week 4: connection pooling).
"""
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,          # set True temporarily to see generated SQL while debugging
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncSession:
    """
    FastAPI dependency. Yields one session per request and guarantees it's
    closed afterwards, even if the request raised an exception.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
