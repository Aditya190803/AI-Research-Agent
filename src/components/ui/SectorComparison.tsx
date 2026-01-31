"use client";

import { FinanceData } from "@/types";
import { motion } from "framer-motion";
import { BarChart3, AlertCircle } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface SectorComparisonProps {
  data: FinanceData[];
}

export function SectorComparison({ data }: SectorComparisonProps) {
  // Filter out stocks with no data (price = 0)
  const validData = data.filter(stock => stock.price > 0);

  if (validData.length < 2) {
    // Not enough valid data to compare
    if (data.length >= 2 && validData.length < 2) {
      return (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass rounded-2xl p-6"
        >
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="w-5 h-5 text-accent" />
            <h3 className="font-semibold text-text-primary">Stock Comparison</h3>
          </div>
          <div className="flex items-center justify-center py-8 text-center">
            <div>
              <AlertCircle className="w-8 h-8 text-warning mx-auto mb-3" />
              <p className="text-text-secondary text-sm">
                Insufficient data available for comparison chart.
              </p>
              <p className="text-text-muted text-xs mt-1">
                Real-time data temporarily unavailable for most stocks.
              </p>
            </div>
          </div>
        </motion.div>
      );
    }
    return null;
  }

  const chartData = validData.map((stock) => ({
    name: stock.symbol,
    price: stock.price,
    change: stock.changePercent.replace("%", ""),
    sector: stock.sector,
    isPositive: !stock.changePercent.startsWith("-"),
  }));

  const currencySymbol = validData[0]?.currencySymbol || "₹";

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-2xl p-6"
    >
      <div className="flex items-center gap-2 mb-6">
        <BarChart3 className="w-5 h-5 text-accent" />
        <h3 className="font-semibold text-text-primary">Stock Comparison</h3>
      </div>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#334155" />
            <XAxis
              dataKey="name"
              axisLine={false}
              tickLine={false}
              tick={{ fill: "#94a3b8", fontSize: 12 }}
            />
            <YAxis
              axisLine={false}
              tickLine={false}
              tick={{ fill: "#94a3b8", fontSize: 12 }}
              tickFormatter={(value) => `${currencySymbol}${value.toLocaleString('en-IN')}`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1e293b",
                border: "1px solid #334155",
                borderRadius: "12px",
                color: "#f8fafc",
              }}
              itemStyle={{ color: "#f8fafc" }}
            />
            <Bar dataKey="price" radius={[4, 4, 0, 0]}>
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.isPositive ? "#10b981" : "#ef4444"}
                  fillOpacity={0.8}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4">
        {chartData.map((stock) => (
          <div key={stock.name} className="flex flex-col">
            <span className="text-xs text-text-muted uppercase tracking-wider">{stock.name}</span>
            <span className="text-sm font-medium text-text-secondary truncate">{stock.sector}</span>
          </div>
        ))}
      </div>
    </motion.div>
  );
}
