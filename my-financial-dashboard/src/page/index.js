import { useState } from 'react';
import StockInput from '../components/StockInput';

export default function Home() {
  const [result, setResult] = useState(null);

  const handleStockSubmit = async (ticker) => {
    // Call your backend API
    const res = await fetch(`http://localhost:8000/investment-guidance/get-investment-guidance`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tickers: [ticker] })
    });
    const data = await res.json();
    setResult(data);
  };

  return (
    <div>
      <h1>Financial Dashboard</h1>
      <StockInput onSubmit={handleStockSubmit} />
      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}
