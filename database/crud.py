from sqlalchemy.orm import Session
from models import StockData

def create_stock_data(db: Session, ticker: str, close_price: float, volume: int, sentiment_score: float):
    stock_entry = StockData(
        ticker=ticker,
        close_price=close_price,
        volume=volume,
        sentiment_score=sentiment_score
    )
    db.add(stock_entry)
    db.commit()
    db.refresh(stock_entry)
    return stock_entry

def get_stock_data(db: Session, ticker: str):
    return db.query(StockData).filter(StockData.ticker == ticker).all()

def delete_stock_data(db: Session, stock_id: int):
    stock = db.query(StockData).filter(StockData.id == stock_id).first()
    if stock:
        db.delete(stock)
        db.commit()
    return stock
