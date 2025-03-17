from fastapi import Depends
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base
from database import InvestmentRecommendation

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
