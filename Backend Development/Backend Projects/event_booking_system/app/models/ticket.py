from sqlalchemy import CheckConstraint, ForeignKey, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Ticket(Base):
    __tablename__ = "tickets"
    __table_args__ = (
        # Speeds up "find this ticket type for this event" lookups, and the
        # booking service's row-lock query hits this index directly.
        Index("ix_tickets_event_type", "event_id", "type"),
        CheckConstraint("available_quantity >= 0", name="ck_tickets_available_nonneg"),
        CheckConstraint("available_quantity <= total_quantity", name="ck_tickets_available_le_total"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[int] = mapped_column(
        ForeignKey("events.id", ondelete="CASCADE"), index=True, nullable=False
    )
    type: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g. "General", "VIP"
    price: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    total_quantity: Mapped[int] = mapped_column(nullable=False)
    available_quantity: Mapped[int] = mapped_column(nullable=False)

    event: Mapped["Event"] = relationship(back_populates="tickets")
    bookings: Mapped[list["Booking"]] = relationship(back_populates="ticket")
