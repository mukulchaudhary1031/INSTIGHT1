from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.sql import func
from database import Base


class User(Base):
    """One row per registered user."""
    __tablename__ = "users"

    id                  = Column(Integer, primary_key=True, index=True)
    email               = Column(String(255), unique=True, index=True, nullable=False)
    password_hash       = Column(String(255), nullable=False)
    token               = Column(String(255), unique=True, nullable=True)  # auth token (stored in browser)

    # ── Upload tracking ──────────────────────────────────────────────────────
    upload_count        = Column(Integer, default=0)   # resets each month on renewal

    # ── Subscription ─────────────────────────────────────────────────────────
    is_subscribed       = Column(Boolean, default=False)
    subscription_end    = Column(DateTime, nullable=True)  # None = never subscribed

    # ── Razorpay ─────────────────────────────────────────────────────────────
    razorpay_order_id   = Column(String(255), nullable=True)
    razorpay_payment_id = Column(String(255), nullable=True)

    created_at          = Column(DateTime, default=func.now())


class SavedDataset(Base):
    """Datasets saved by users for comparison — stored in DB, not on disk."""
    __tablename__ = "saved_datasets"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, nullable=True, index=True)
    label      = Column(String(100), index=True)
    data_json  = Column(Text)        # df.to_json() — entire dataframe as JSON string
    filename   = Column(String(200))
    created_at = Column(DateTime, default=func.now())