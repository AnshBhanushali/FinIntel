from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import relationship
from database import Base
import datetime

class StockData(Base):
    __tablename__ = "stock_data"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False)
    date = Column(DateTime, default=datetime.datetime.utcnow)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=True)
    sentiment_score = Column(Float, nullable=True)
