"""
Minimal entrypoint for step 9 (verify the skeleton runs end-to-end).
Routers for auth/events/bookings get added here once they exist —
this file intentionally stays thin; it wires things together, it
doesn't contain logic.
"""
from fastapi import FastAPI
from sqlalchemy import text

from app.db.session import engine

app = FastAPI(title="Event Booking System", version="0.1.0")


@app.get("/health")
async def health_check():
    """
    Confirms config -> engine -> DB connection all actually work.
    This is the checkpoint to hit before writing any business logic.
    """
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as exc:  # noqa: BLE001 - health check reports any failure
        db_status = f"error: {exc}"

    return {"status": "ok", "database": db_status}
