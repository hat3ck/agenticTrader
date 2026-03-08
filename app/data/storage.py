"""SQLite persistence layer (SQLAlchemy async).

Stores analysis history, past recommendations, and tracks performance.
MVP: SQLite via aiosqlite.  Production: swap DATABASE_URL to Postgres.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class RecommendationRecord(Base):
    """Persisted recommendation for performance tracking."""
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, server_default=func.now())
    ticker = Column(String(10), nullable=False, index=True)
    allocation_usd = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    strategy = Column(String(64), nullable=False)
    rationale = Column(Text, default="")
    horizon = Column(String(20), nullable=False)
    # Filled in later when we track performance
    price_at_recommendation = Column(Float, nullable=True)
    price_at_evaluation = Column(Float, nullable=True)
    return_pct = Column(Float, nullable=True)


class AnalysisRecord(Base):
    """Persisted analysis snapshot."""
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, server_default=func.now())
    ticker = Column(String(10), nullable=False, index=True)
    analysis_json = Column(Text, nullable=False)
    rating = Column(String(16), default="hold")


async def init_db() -> None:
    """Create all tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def save_recommendations(records: list[dict]) -> None:
    """Persist a batch of recommendation dicts."""
    async with async_session() as session:
        for rec in records:
            session.add(RecommendationRecord(**rec))
        await session.commit()


async def save_analysis(ticker: str, analysis_json: str, rating: str) -> None:
    """Persist a single analysis snapshot."""
    async with async_session() as session:
        session.add(AnalysisRecord(ticker=ticker, analysis_json=analysis_json, rating=rating))
        await session.commit()
