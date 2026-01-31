"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useResearch } from "@/lib/context/ResearchContext";
import {
  Search,
  Brain,
  FileText,
  TrendingUp,
  Newspaper,
  Sparkles,
  Loader2,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { FinanceCardSkeleton, SourceCardSkeleton } from "@/components/ui/Skeleton";

const researchSteps = [
  { icon: Search, label: "Searching the web", duration: 3000 },
  { icon: TrendingUp, label: "Gathering real-time data", duration: 2500 },
  { icon: Newspaper, label: "Checking latest news", duration: 2000 },
  { icon: Brain, label: "Analyzing information", duration: 4000 },
  { icon: FileText, label: "Cross-referencing sources", duration: 3500 },
  { icon: Sparkles, label: "Synthesizing insights", duration: 3000 },
];

export function ResearchingPhase() {
  const { strategy, cancelResearch } = useResearch();
  const [currentStep, setCurrentStep] = useState(0);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [currentSubQueryIndex, setCurrentSubQueryIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setElapsedTime((prev) => prev + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (currentStep < researchSteps.length - 1) {
      const timer = setTimeout(() => {
        setCurrentStep((prev) => prev + 1);
      }, researchSteps[currentStep].duration);

      return () => clearTimeout(timer);
    }
  }, [currentStep]);

  // Cycle through sub-queries
  useEffect(() => {
    if (strategy && strategy.subQueries.length > 0) {
      const interval = setInterval(() => {
        setCurrentSubQueryIndex((prev) => (prev + 1) % strategy.subQueries.length);
      }, 3000);
      return () => clearInterval(interval);
    }
  }, [strategy]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
  };

  return (
    <div className="max-w-2xl mx-auto">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="glass rounded-2xl p-8"
      >
        {/* Header */}
        <div className="text-center mb-8">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
            className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-accent/10 mb-6"
          >
            <Brain className="w-10 h-10 text-accent" />
          </motion.div>

          <h2 className="text-2xl font-semibold text-text-primary mb-2">
            Research in Progress
          </h2>
          <p className="text-text-secondary">
            Analyzing multiple sources to bring you comprehensive insights
          </p>

          {/* Timer */}
          <div className="mt-4 inline-flex items-center gap-2 px-4 py-2 rounded-full bg-surface text-text-muted text-sm">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Elapsed: {formatTime(elapsedTime)}</span>
          </div>
        </div>

        {/* Progress Steps */}
        <div className="space-y-3">
          {researchSteps.map((step, index) => {
            const Icon = step.icon;
            const isActive = index === currentStep;
            const isComplete = index < currentStep;

            return (
              <motion.div
                key={step.label}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`flex items-center gap-4 p-4 rounded-xl transition-all duration-300 ${
                  isActive
                    ? "bg-accent/10 border border-accent/30"
                    : isComplete
                    ? "bg-success/5"
                    : "bg-surface/30"
                }`}
              >
                <div
                  className={`flex items-center justify-center w-10 h-10 rounded-xl transition-all duration-300 ${
                    isActive
                      ? "bg-accent text-white"
                      : isComplete
                      ? "bg-success/10 text-success"
                      : "bg-surface text-text-muted"
                  }`}
                >
                  {isActive ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Icon className="w-5 h-5" />
                  )}
                </div>

                <div className="flex-1">
                  <p
                    className={`font-medium transition-colors ${
                      isActive
                        ? "text-text-primary"
                        : isComplete
                        ? "text-success"
                        : "text-text-muted"
                    }`}
                  >
                    {step.label}
                    {isActive && <span className="typing-dots ml-1" />}
                  </p>
                </div>

                {isComplete && (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="w-6 h-6 rounded-full bg-success/10 flex items-center justify-center"
                  >
                    <svg
                      className="w-4 h-4 text-success"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                  </motion.div>
                )}
              </motion.div>
            );
          })}
        </div>

        {/* Sub-queries being processed */}
        {strategy && strategy.subQueries.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="mt-8 p-4 rounded-xl bg-surface/30 border border-border-subtle"
          >
            <div className="flex items-center justify-between mb-3">
              <p className="text-sm text-text-muted">Processing research tasks:</p>
              <span className="text-xs text-accent font-medium">
                {currentSubQueryIndex + 1} / {strategy.subQueries.length}
              </span>
            </div>
            
            <AnimatePresence mode="wait">
              <motion.div
                key={currentSubQueryIndex}
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -5 }}
                className="flex items-center gap-3 text-text-primary"
              >
                <div className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
                <p className="text-sm font-medium line-clamp-1">
                  {strategy.subQueries[currentSubQueryIndex]}
                </p>
              </motion.div>
            </AnimatePresence>

            <div className="mt-4 flex flex-wrap gap-2">
              {strategy.subQueries.map((query, index) => (
                <div
                  key={index}
                  className={cn(
                    "h-1 flex-1 rounded-full transition-all duration-500",
                    index === currentSubQueryIndex ? "bg-accent" : 
                    index < currentSubQueryIndex ? "bg-success/50" : "bg-surface"
                  )}
                />
              ))}
            </div>
          </motion.div>
        )}

        {/* Cancel Button */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="mt-8 text-center"
        >
          <button
            onClick={cancelResearch}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-text-muted 
                     hover:text-text-primary hover:bg-surface transition-all duration-200"
          >
            <X className="w-4 h-4" />
            <span>Cancel Research</span>
          </button>
        </motion.div>
      </motion.div>

      {/* Preview Skeletons */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="mt-12 space-y-8"
      >
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-accent/50" />
            <div className="h-5 w-32 bg-surface-hover/50 rounded animate-pulse" />
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            <FinanceCardSkeleton />
            <FinanceCardSkeleton />
          </div>
        </div>

        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-accent/50" />
            <div className="h-5 w-32 bg-surface-hover/50 rounded animate-pulse" />
          </div>
          <div className="grid gap-3">
            <SourceCardSkeleton />
            <SourceCardSkeleton />
            <SourceCardSkeleton />
          </div>
        </div>
      </motion.div>
    </div>
  );
}
