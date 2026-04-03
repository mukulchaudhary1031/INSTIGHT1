from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ── Change YOUR_PASSWORD to your MySQL password ──────────────────────────────
DATABASE_URL = "mysql+pymysql://root:Mukul123@localhost/insights"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # auto-reconnect if connection drops
    pool_recycle=3600,    # recycle connections every 1 hour
    pool_size=10,         # 10 concurrent connections
    max_overflow=20       # allow 20 extra when busy
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """FastAPI dependency — gives a DB session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()