"use client";
import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const FinancialDashboard = () => {
  const [stockData, setStockData] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("http://localhost:8000/get-investment-guidance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tickers: ["AAPL", "TSLA"],
        initial_capital: 10000,
        risk_tolerance: "medium",
      }),
    })
      .then(response => response.json())
      .then(data => {
        if (data.stock_prices) {
          setStockData(data.stock_prices);
        } else {
          setError("No stock data available.");
        }
      })
      .catch(error => setError("Error fetching stock data: " + error.message));
  }, []);

  return (
    <div className="dashboard-container">
      <h2 className="dashboard-title">Financial Dashboard</h2>
      {error && <p className="error">{error}</p>}
      <div className="charts-grid">
        {stockData.map((stock, index) => (
          <div key={index} className="stock-chart">
            <h3>{stock.ticker} Stock Price History</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={stock.data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="price" stroke="#8884d8" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>
    </div>
  );
};

// Styles
const styles = `
  .dashboard-container {
    padding: 2rem;
    background: #f5f5f5;
    border-radius: 8px;
    margin: 2rem auto;
    max-width: 1200px;
  }
  .dashboard-title {
    color: #333;
    text-align: center;
    margin-bottom: 2rem;
  }
  .error {
    color: #ff4444;
    text-align: center;
  }
  .charts-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 2rem;
  }
  .stock-chart {
    background: white;
    padding: 1rem;
    border-radius: 4px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
  }
  .stock-chart h3 {
    color: #555;
    margin-bottom: 1rem;
  }
`;

// ✅ Only one default export wrapping the component
export default function FinancialDashboardWithStyles() {
  return (
    <>
      <style>{styles}</style>
      <FinancialDashboard />
    </>
  );
}
