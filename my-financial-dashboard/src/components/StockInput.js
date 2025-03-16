// components/StockInput.js
import { useState } from 'react';

export default function StockInput({ onSubmit }) {
  const [ticker, setTicker] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (ticker.trim() !== '') {
      onSubmit(ticker.trim().toUpperCase());
      setTicker('');
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        placeholder="Enter stock symbol (e.g., AAPL)"
        value={ticker}
        onChange={(e) => setTicker(e.target.value)}
      />
      <button type="submit">Analyze</button>
    </form>
  );
}
