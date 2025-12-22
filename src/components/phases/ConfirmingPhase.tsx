"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { useResearch } from "@/lib/context/ResearchContext";
import {
  Target,
  Route,
  Search,
  CheckCircle,
  Edit3,
  Play,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";

export function ConfirmingPhase() {
  const { query, strategy, executeDeepResearch, cancelResearch } = useResearch();
  const [isExecuting, setIsExecuting] = useState(false);

  const handleExecute = async () => {
    setIsExecuting(true);
    await executeDeepResearch();
  };

  if (!strategy) {
    return (
      <div className="max-w-2xl mx-auto">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass rounded-2xl p-8 text-center"
        >
          <Loader2 className="w-12 h-12 text-accent animate-spin mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-text-primary mb-2">
            Building Research Strategy
          </h2>
          <p className="text-text-secondary">
            Crafting the optimal approach for your research...
          </p>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8"
      >
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-accent/10 mb-4">
          <Target className="w-8 h-8 text-accent" />
        </div>
        <h2 className="text-2xl font-semibold text-text-primary mb-2">
          Research Strategy Ready
        </h2>
        <p className="text-text-secondary">
          Review your research plan before we begin the deep investigation
        </p>
      </motion.div>

      {/* Strategy Cards */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Scope */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass rounded-2xl p-6"
        >
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-accent/10">
              <Target className="w-5 h-5 text-accent" />
            </div>
            <h3 className="font-semibold text-text-primary">Research Scope</h3>
          </div>
          <p className="text-text-secondary">{strategy.scope}</p>
        </motion.div>

        {/* Optimized Query */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
          className="glass rounded-2xl p-6"
        >
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-success/10">
              <Edit3 className="w-5 h-5 text-success" />
            </div>
            <h3 className="font-semibold text-text-primary">Optimized Query</h3>
          </div>
          <p className="text-text-secondary">{strategy.optimizedQuery}</p>
        </motion.div>
      </div>

      {/* Sub-queries */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="glass rounded-2xl p-6"
      >
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-warning/10">
            <Search className="w-5 h-5 text-warning" />
          </div>
          <h3 className="font-semibold text-text-primary">Search Queries</h3>
        </div>
        <div className="space-y-2">
          {strategy.subQueries.map((subQuery, index) => (
            <div
              key={index}
              className="flex items-start gap-3 p-3 rounded-xl bg-surface/50"
            >
              <span className="flex items-center justify-center w-6 h-6 rounded-full bg-accent/10 text-accent text-xs font-medium">
                {index + 1}
              </span>
              <p className="text-text-secondary flex-1">{subQuery}</p>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Investigation Path */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35 }}
        className="glass rounded-2xl p-6"
      >
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-accent/10">
            <Route className="w-5 h-5 text-accent" />
          </div>
          <h3 className="font-semibold text-text-primary">Investigation Path</h3>
        </div>
        <div className="space-y-3">
          {strategy.investigationPath.map((step, index) => (
            <div key={index} className="flex items-start gap-3">
              <div className="flex flex-col items-center">
                <div className="w-8 h-8 rounded-full bg-accent/10 flex items-center justify-center">
                  <CheckCircle className="w-4 h-4 text-accent" />
                </div>
                {index < strategy.investigationPath.length - 1 && (
                  <div className="w-0.5 h-6 bg-border-subtle mt-1" />
                )}
              </div>
              <p className="text-text-secondary pt-1">{step}</p>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="flex items-center justify-center gap-4 pt-4"
      >
        <button
          onClick={cancelResearch}
          disabled={isExecuting}
          className="px-6 py-2.5 rounded-xl text-text-secondary hover:text-text-primary 
                   hover:bg-surface transition-all duration-200 disabled:opacity-50"
        >
          Cancel
        </button>

        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleExecute}
          disabled={isExecuting}
          className={cn(
            "flex items-center gap-2 px-8 py-3 rounded-xl font-medium transition-all duration-200",
            !isExecuting
              ? "bg-accent text-white hover:bg-accent-hover shadow-lg shadow-accent/20 pulse-glow"
              : "bg-accent/50 text-white/50 cursor-not-allowed"
          )}
        >
          {isExecuting ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Starting Research...</span>
            </>
          ) : (
            <>
              <Play className="w-5 h-5" />
              <span>Start Deep Research</span>
            </>
          )}
        </motion.button>
      </motion.div>
    </div>
  );
}
