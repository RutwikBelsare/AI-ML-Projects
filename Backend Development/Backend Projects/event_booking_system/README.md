# Event Booking System — backend skeleton

Current status: steps 1–9 of the build plan (config, models, migrations,
health check). Auth/events/bookings/tasks come next.

## Prerequisites
- Python 3.11+
- PostgreSQL running locally
- Redis running locally

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# edit .env with your real DATABASE_URL, JWT_SECRET_KEY, REDIS_URL

# 4. Create the database (adjust user/db name to match your .env)
createdb event_booking

# 5. Run the first migration
alembic revision --autogenerate -m "initial tables"
alembic upgrade head

# 6. Run the app
uvicorn app.main:app --reload
```

## Verify it worked
Open http://localhost:8000/health — you should see:
```json
{"status": "ok", "database": "ok"}
```
If `database` shows an error instead, check `DATABASE_URL` in `.env` and
that Postgres is actually running and the `event_booking` DB exists.

Swagger docs: http://localhost:8000/docs

## Project layout
```
app/
  config.py          # Pydantic Settings, reads .env
  main.py            # FastAPI app + health check
  db/
    base.py           # Declarative Base
    session.py         # async engine, sessionmaker, get_db()
  models/             # SQLAlchemy ORM: organizer, customer, event, ticket, booking
  schemas/            # (next) Pydantic request/response models
  repositories/       # (next) DB access layer
  services/           # (next) business logic, transactions
  core/               # (next) security (JWT), dependencies (role guards)
  api/v1/             # (next) route handlers
  tasks/              # (next) Celery tasks
alembic/              # migrations
tests/                # unit + integration
```

## Design decisions worth knowing
- **Async everywhere**: engine, sessions, and routes are all async so the
  API doesn't block on DB I/O under load.
- **Booking concurrency**: ticket availability is protected with a
  `SELECT ... FOR UPDATE` row lock inside a transaction (in the upcoming
  `booking_service`) so two simultaneous bookings can never oversell the
  last ticket.
- **No Docker/CI**: intentionally out of scope for this assignment — run
  Postgres/Redis natively and use the venv directly.
