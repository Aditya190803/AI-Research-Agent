"use client";

import { motion } from "framer-motion";
import { FinanceData } from "@/types";
import { TrendingUp, TrendingDown, Building, BarChart3 } from "lucide-react";
import { cn } from "@/lib/utils";

interface FinanceCardProps {
  data: FinanceData;
}

export function FinanceCard({ data }: FinanceCardProps) {
  const isPositive = data.change >= 0;
  const changeColor = isPositive ? "text-success" : "text-error";
  const changeBg = isPositive ? "bg-success/10" : "bg-error/10";

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat("en-IN", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(num);
  };

  const formatMarketCap = (cap: string) => {
    const num = parseFloat(cap);
    if (isNaN(num)) return cap;
    if (num >= 1e12) return `${(num / 1e12).toFixed(2)}T`;
    if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
    return cap;
  };

  // Generate mock sparkline data
  const generateSparklineData = () => {
    const points = 20;
    const dataPoints = [];
    let current = 50;
    const trend = isPositive ? 0.5 : -0.5;
    
    for (let i = 0; i < points; i++) {
      current += (Math.random() - 0.5) * 10 + trend;
      dataPoints.push(current);
    }
    
    const min = Math.min(...dataPoints);
    const max = Math.max(...dataPoints);
    const range = max - min;
    
    return dataPoints.map((p, i) => ({
      x: (i / (points - 1)) * 100,
      y: 100 - ((p - min) / range) * 100
    }));
  };

  const sparklinePoints = generateSparklineData();
  const pathData = `M ${sparklinePoints.map(p => `${p.x},${p.y}`).join(' L ')}`;

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

      {/* Price and Sparkline */}
      <div className="flex items-end justify-between mb-6">
        <div>
          <span className="text-3xl font-bold text-text-primary">
            {data.currencySymbol}
            {formatNumber(data.price)}
          </span>
          <div className={cn("mt-1 text-sm font-medium", changeColor)}>
            {isPositive ? "+" : ""}
            {formatNumber(data.change)} ({data.currency})
          </div>
        </div>
        
        <div className="w-24 h-12">
          <svg viewBox="0 0 100 100" className="w-full h-full overflow-visible">
            <defs>
              <linearGradient id={`gradient-${data.symbol}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={isPositive ? "var(--color-success)" : "var(--color-error)"} stopOpacity="0.2" />
                <stop offset="100%" stopColor={isPositive ? "var(--color-success)" : "var(--color-error)"} stopOpacity="0" />
              </linearGradient>
            </defs>
            <path
              d={`${pathData} L 100,100 L 0,100 Z`}
              fill={`url(#gradient-${data.symbol})`}
            />
            <motion.path
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 1.5, ease: "easeInOut" }}
              d={pathData}
              fill="none"
              stroke={isPositive ? "var(--color-success)" : "var(--color-error)"}
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-3">
        <StatItem label="Day High" value={`${data.currencySymbol}${formatNumber(data.high)}`} />
        <StatItem label="Day Low" value={`${data.currencySymbol}${formatNumber(data.low)}`} />
        <StatItem
          label="52W High"
          value={`${data.currencySymbol}${formatNumber(data.fiftyTwoWeekHigh)}`}
        />
        <StatItem
          label="52W Low"
          value={`${data.currencySymbol}${formatNumber(data.fiftyTwoWeekLow)}`}
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
              {data.dividendYield !== "N/A"
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
  return (
    <div className="p-3 rounded-xl bg-surface/50">
      <p className="text-xs text-text-muted mb-1">{label}</p>
      <p className="text-sm font-medium text-text-primary">{value}</p>
    </div>
  );
}
