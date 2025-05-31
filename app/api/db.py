# db.py - Database setup and models for feedback API
import os
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from typing import Generator


def get_database_url() -> str:
    """
    Get the database URL from environment variables or use default.

    Returns:
        str: The database URL string.
    """
    return os.getenv("FEEDBACK_DB_URL", "sqlite:///./feedback.db")


def create_db_engine() -> any:
    """
    Create a SQLAlchemy engine with appropriate connect args.

    Returns:
        Engine: SQLAlchemy engine instance.
    """
    db_url = get_database_url()
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    return create_engine(db_url, connect_args=connect_args)


engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Feedback(Base):
    """
    SQLAlchemy model for the feedback table.

    Attributes:
        id (int): Primary key.
        user_input (str): User's input text.
        api_output (str): Output returned by the API.
        feedback (str): Feedback type (e.g., positive/negative).
    """
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    user_input = Column(Text, nullable=False)
    api_output = Column(Text, nullable=False)
    feedback = Column(String(20), nullable=False)


Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Yield a database session and ensure it is closed after use.

    Yields:
        Session: SQLAlchemy database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
