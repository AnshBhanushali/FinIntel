"use client";
import React, { useState } from "react";
import styles from "./StockInput.module.css";

// Define the shape of props for this component
interface StockInputProps {
  onSubmit: (params: {
    tickers: string[];
    riskTolerance: string;
    totalCapital: number;
  }) => void;
}

// Define your component using TypeScript
export default function StockInput({ onSubmit }: StockInputProps) {
  // Four default dropdowns
  const [tickers, setTickers] = useState<string[]>(["", "", "", ""]);
  // Additional tickers
  const [extraTickers, setExtraTickers] = useState<string[]>([]);
  // Risk tolerance (default: "medium")
  const [riskTolerance, setRiskTolerance] = useState<string>("medium");
  // Total capital in dollars
  const [totalCapital, setTotalCapital] = useState<string>("");

  // Predefined stock options
  const stockOptions = [
    { value: "AAPL", label: "Apple Inc." },
    { value: "TSLA", label: "Tesla Inc." },
    { value: "MSFT", label: "Microsoft Corp." },
    { value: "GOOGL", label: "Alphabet Inc." },
    { value: "AMZN", label: "Amazon.com Inc." },
    { value: "NVDA", label: "NVIDIA Corp." },
    { value: "META", label: "Meta Platforms Inc." },
  ];

  // Handle changes to either default or extra tickers
  const handleTickerChange = (
    index: number,
    value: string,
    isExtra: boolean = false
  ) => {
    if (isExtra) {
      const newExtraTickers = [...extraTickers];
      newExtraTickers[index] = value.toUpperCase();
      setExtraTickers(newExtraTickers);
    } else {
      const newTickers = [...tickers];
      newTickers[index] = value.toUpperCase();
      setTickers(newTickers);
    }
  };

  // Add a new ticker field
  const addTickerField = () => {
    setExtraTickers([...extraTickers, ""]);
  };

  // Remove an extra ticker field
  const removeExtraTicker = (index: number) => {
    setExtraTickers(extraTickers.filter((_, i) => i !== index));
  };

  // When the form is submitted
  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    // Combine default and extra tickers, removing any blank strings
    const allTickers = [...tickers, ...extraTickers].filter(
      (t) => t.trim() !== ""
    );

    // Basic validation
    if (allTickers.length === 0 || !totalCapital) {
      alert("Please select at least one stock and enter total capital.");
      return;
    }

    // Call the parent-provided onSubmit callback
    onSubmit({
      tickers: allTickers,
      riskTolerance,
      totalCapital: parseFloat(totalCapital),
    });
  };

  return (
    <form className={styles.form} onSubmit={handleSubmit}>
      <div className={styles.tickerSection}>
        <h3>Select Stocks</h3>

        {/* Default 4 dropdowns */}
        {tickers.map((ticker, index) => (
          <select
            key={`default-${index}`}
            value={ticker}
            onChange={(e) => handleTickerChange(index, e.target.value)}
            className={styles.dropdown}
          >
            <option value="">Select a stock</option>
            {stockOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label} ({option.value})
              </option>
            ))}
          </select>
        ))}

        {/* Extra ticker fields */}
        {extraTickers.map((ticker, index) => (
          <div key={`extra-${index}`} className={styles.extraTicker}>
            <select
              value={ticker}
              onChange={(e) => handleTickerChange(index, e.target.value, true)}
              className={styles.dropdown}
            >
              <option value="">Select a stock</option>
              {stockOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label} ({option.value})
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => removeExtraTicker(index)}
              className={styles.removeButton}
            >
              Remove
            </button>
          </div>
        ))}

        <button
          type="button"
          onClick={addTickerField}
          className={styles.addButton}
        >
          Add Another Stock
        </button>
      </div>

      <div className={styles.inputSection}>
        <label>Risk Tolerance</label>
        <select
          value={riskTolerance}
          onChange={(e) => setRiskTolerance(e.target.value)}
          className={styles.dropdown}
        >
          <option value="low">Low</option>
          <option value="medium">Medium</option>
          <option value="high">High</option>
        </select>
      </div>

      <div className={styles.inputSection}>
        <label>Total Capital ($)</label>
        <input
          type="number"
          value={totalCapital}
          onChange={(e) => setTotalCapital(e.target.value)}
          placeholder="Enter total capital"
          className={styles.input}
          min="0"
          step="100"
        />
      </div>

      <button type="submit" className={styles.submitButton}>
        Analyze Portfolio
      </button>
    </form>
  );
}
