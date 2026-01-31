"use client";

import { motion } from "framer-motion";
import { FinanceData } from "@/types";
import { TrendingUp, TrendingDown, Building, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface FinanceCardProps {
  data: FinanceData;
}

export function FinanceCard({ data }: FinanceCardProps) {
  const isDataAvailable = data.price > 0;
  const isPositive = data.change >= 0;
  const changeColor = isPositive ? "text-success" : "text-error";
  const changeBg = isPositive ? "bg-success/10" : "bg-error/10";
  const chartColor = isPositive ? "#10b981" : "#ef4444";

  const formatNumber = (num: number) => {
    if (num === 0 || isNaN(num)) return "—";
    return new Intl.NumberFormat("en-IN", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(num);
  };

  const formatPrice = (num: number) => {
    if (num === 0 || isNaN(num)) return "N/A";
    return new Intl.NumberFormat("en-IN", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(num);
  };

  const formatMarketCap = (cap: string) => {
    if (!cap || cap === "N/A") return "N/A";
    const num = parseFloat(cap);
    if (isNaN(num)) return cap;
    if (num >= 1e12) return `${(num / 1e12).toFixed(2)}T`;
    if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e7) return `₹${(num / 1e7).toFixed(2)}Cr`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
    return cap;
  };

  const chartData = data.historicalData || [];

  // Show error state if data is unavailable
  if (!isDataAvailable) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass rounded-2xl p-5 hover:border-accent/30 transition-all duration-300 flex flex-col"
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-lg font-bold text-text-primary">{data.symbol}</span>
              <span className="px-2 py-0.5 rounded-full bg-surface text-text-muted text-xs">
                {data.exchange !== "N/A" ? data.exchange : "Unknown"}
              </span>
            </div>
            <p className="text-text-secondary text-sm truncate">{data.companyName}</p>
          </div>
        </div>

        {/* Error State */}
        <div className="flex-1 flex flex-col items-center justify-center py-8 text-center">
          <div className="p-3 rounded-full bg-warning/10 mb-4">
            <AlertCircle className="w-8 h-8 text-warning" />
          </div>
          <h4 className="text-text-primary font-medium mb-2">Data Temporarily Unavailable</h4>
          <p className="text-text-muted text-sm max-w-[200px]">
            Unable to fetch real-time data for this stock. This may be due to API rate limits or market hours.
          </p>
        </div>

        {/* Retry hint */}
        <div className="mt-4 pt-4 border-t border-border-subtle text-center">
          <p className="text-xs text-text-muted">
            Data will be retried automatically on the next research query.
          </p>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-2xl p-5 hover:border-accent/30 transition-all duration-300 flex flex-col"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-lg font-bold text-text-primary">{data.symbol}</span>
            <span className="px-2 py-0.5 rounded-full bg-surface text-text-muted text-xs">
              {data.exchange}
            </span>
          </div>
          <p className="text-text-secondary text-sm truncate">{data.companyName}</p>
        </div>
        <div
          className={cn(
            "flex items-center gap-1 px-3 py-1.5 rounded-full text-sm font-medium",
            changeBg,
            changeColor
          )}
        >
          {isPositive ? (
            <TrendingUp className="w-4 h-4" />
          ) : (
            <TrendingDown className="w-4 h-4" />
          )}
          <span>{data.changePercent}</span>
        </div>
      </div>

      {/* Price and Overview */}
      <div className="flex items-end justify-between mb-6">
        <div>
          <span className="text-3xl font-bold text-text-primary">
            {data.currencySymbol}
            {formatPrice(data.price)}
          </span>
          <div className={cn("mt-1 text-sm font-medium", changeColor)}>
            {isPositive ? "+" : ""}
            {formatNumber(data.change)} ({data.currency})
          </div>
        </div>
      </div>

      {/* Interactive Chart */}
      <div className="w-full h-48 mb-6 -mx-2">
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id={`colorPrice-${data.symbol}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={chartColor} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={chartColor} stopOpacity={0} />
                </linearGradient>
              </defs>
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="bg-surface-elevated border border-border-subtle p-3 rounded-xl shadow-2xl text-xs backdrop-blur-md">
                        <p className="text-text-muted mb-1 font-medium">{payload[0].payload.date}</p>
                        <p className="text-lg font-bold text-text-primary">
                          {data.currencySymbol}{formatPrice(payload[0].value as number)}
                        </p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <XAxis dataKey="date" hide />
              <YAxis hide domain={["auto", "auto"]} />
              <Area
                type="monotone"
                dataKey="price"
                stroke={chartColor}
                strokeWidth={2.5}
                fillOpacity={1}
                fill={`url(#colorPrice-${data.symbol})`}
                animationDuration={1500}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="w-full h-full flex items-center justify-center bg-surface/30 rounded-xl border border-dashed border-border-subtle">
            <p className="text-xs text-text-muted italic">Historical data unavailable</p>
          </div>
        )}
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-3">
        <StatItem
          label="Day High"
          value={data.high > 0 ? `${data.currencySymbol}${formatPrice(data.high)}` : "N/A"}
        />
        <StatItem
          label="Day Low"
          value={data.low > 0 ? `${data.currencySymbol}${formatPrice(data.low)}` : "N/A"}
        />
        <StatItem
          label="52W High"
          value={data.fiftyTwoWeekHigh > 0 ? `${data.currencySymbol}${formatPrice(data.fiftyTwoWeekHigh)}` : "N/A"}
        />
        <StatItem
          label="52W Low"
          value={data.fiftyTwoWeekLow > 0 ? `${data.currencySymbol}${formatPrice(data.fiftyTwoWeekLow)}` : "N/A"}
        />
      </div>

      {/* Additional Info */}
      <div className="mt-4 pt-4 border-t border-border-subtle">
        <div className="grid grid-cols-3 gap-3 text-center">
          <div>
            <p className="text-xs text-text-muted mb-1">Market Cap</p>
            <p className="text-sm font-medium text-text-primary">
              {formatMarketCap(data.marketCap)}
            </p>
          </div>
          <div>
            <p className="text-xs text-text-muted mb-1">P/E Ratio</p>
            <p className="text-sm font-medium text-text-primary">{data.peRatio}</p>
          </div>
          <div>
            <p className="text-xs text-text-muted mb-1">Div Yield</p>
            <p className="text-sm font-medium text-text-primary">
              {data.dividendYield && data.dividendYield !== "N/A"
                ? `${(parseFloat(data.dividendYield) * 100).toFixed(2)}%`
                : "N/A"}
            </p>
          </div>
        </div>
      </div>

      {/* Sector Info */}
      {data.sector && data.sector !== "N/A" && (
        <div className="mt-4 flex items-center gap-2">
          <Building className="w-4 h-4 text-text-muted" />
          <span className="text-sm text-text-muted">
            {data.sector} • {data.industry}
          </span>
        </div>
      )}
    </motion.div>
  );
}

function StatItem({ label, value }: { label: string; value: string }) {
  const isUnavailable = value === "N/A" || value.includes("—");
  return (
    <div className="p-3 rounded-xl bg-surface/50">
      <p className="text-xs text-text-muted mb-1">{label}</p>
      <p className={cn(
        "text-sm font-medium",
        isUnavailable ? "text-text-muted italic" : "text-text-primary"
      )}>
        {value}
      </p>
    </div>
  );
}
