"use client";

import { useResearch } from "@/lib/context/ResearchContext";
import { motion } from "framer-motion";
import { Search, ArrowRight, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { KeyboardEvent } from "react";

interface SearchInputProps {
  variant?: "home" | "compact";
}

export function SearchInput({ variant = "home" }: SearchInputProps) {
  const { query, setQuery, handleSearch, phase, result } = useResearch();
  const isLoading = ["exploring", "clarifying", "confirming", "researching"].includes(phase);
  const isSameQuery = phase === "complete" && query.trim() === result?.query?.trim();

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (query.trim() && !isLoading && !isSameQuery) {
        handleSearch();
      }
    }
  };

  const handleSubmit = () => {
    if (query.trim() && !isLoading && !isSameQuery) {
      handleSearch();
    }
  };

  if (variant === "compact") {
    // During research, show as a read-only display
    if (isLoading) {
      return (
        <div className="relative">
          <div className="flex items-center gap-3 p-3 rounded-xl glass">
            <Loader2 className="w-5 h-5 text-accent animate-spin flex-shrink-0" />
            <p className="flex-1 text-text-primary truncate">{query}</p>
          </div>
        </div>
      );
    }

    return (
      <div className="relative">
        <div className="relative flex items-center gap-3 p-3 rounded-xl glass gradient-border">
          <Search className="w-5 h-5 text-text-muted flex-shrink-0" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                handleSubmit();
              }
            }}
            placeholder="Ask anything..."
            className="flex-1 bg-transparent text-text-primary placeholder:text-text-muted outline-none"
          />
          <motion.button
            whileHover={{ scale: isSameQuery ? 1 : 1.05 }}
            whileTap={{ scale: isSameQuery ? 1 : 0.95 }}
            onClick={handleSubmit}
            disabled={!query.trim() || isSameQuery}
            className={cn(
              "p-2 rounded-lg transition-all duration-200",
              query.trim() && !isSameQuery
                ? "bg-accent text-white hover:bg-accent-hover"
                : "bg-surface text-text-muted cursor-not-allowed"
            )}
          >
            <ArrowRight className="w-5 h-5" />
          </motion.button>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ scale: 0.98, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="relative"
    >
      <div
        className={cn(
          "relative rounded-2xl glass gradient-border overflow-hidden",
          "focus-within:ring-2 focus-within:ring-accent/30 transition-all duration-300"
        )}
      >
        <div className="flex items-start gap-4 p-4 md:p-5">
          <Search className="w-6 h-6 text-text-muted flex-shrink-0 mt-1" />
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="What would you like to research today?"
            rows={3}
            className="flex-1 bg-transparent text-lg text-text-primary placeholder:text-text-muted 
                     outline-none resize-none min-h-[80px]"
          />
        </div>

        <div className="flex items-center justify-between px-5 py-3 border-t border-border-subtle bg-surface/30">
          <p className="text-sm text-text-muted">
            Press <kbd className="px-1.5 py-0.5 rounded bg-surface text-text-secondary text-xs font-mono">Enter</kbd> to search
          </p>
          <motion.button
            whileHover={{ scale: (isLoading || isSameQuery) ? 1 : 1.02 }}
            whileTap={{ scale: (isLoading || isSameQuery) ? 1 : 0.98 }}
            onClick={handleSubmit}
            disabled={!query.trim() || isLoading || isSameQuery}
            className={cn(
              "flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium transition-all duration-200",
              query.trim() && !isLoading && !isSameQuery
                ? "bg-accent text-white hover:bg-accent-hover shadow-lg shadow-accent/20"
                : "bg-surface text-text-muted cursor-not-allowed"
            )}
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Researching...</span>
              </>
            ) : (
              <>
                <span>Research</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
}
