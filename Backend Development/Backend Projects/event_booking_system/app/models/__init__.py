"""
Import every model here. Alembic's env.py imports app.models (this file),
which registers all tables on Base.metadata before autogenerate compares
metadata against the live DB schema. Forgetting to import a model here is
the #1 reason `alembic revision --autogenerate` silently misses a table.
"""
from app.models.booking import Booking
from app.models.customer import Customer
from app.models.event import Event
from app.models.organizer import EventOrganizer
from app.models.ticket import Ticket

__all__ = ["Booking", "Customer", "Event", "EventOrganizer", "Ticket"]
