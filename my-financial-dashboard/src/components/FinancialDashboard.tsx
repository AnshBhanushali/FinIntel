"use client"; // Required in Next.js for client-side rendering

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
        risk_tolerance: "medium"
      })
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
      <h2>Financial Dashboard</h2>
      {error && <p style={{ color: "red" }}>{error}</p>}

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
  );
};

export default FinancialDashboard;
