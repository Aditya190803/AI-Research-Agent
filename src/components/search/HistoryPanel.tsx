"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useResearch } from "@/lib/context/ResearchContext";
import { 
  History, 
  X, 
  Search, 
  Clock, 
  ChevronRight, 
  Trash2,
  MessageSquare,
  TrendingUp,
  FileText
} from "lucide-react";
import { cn } from "@/lib/utils";
import { formatDistanceToNow } from "date-fns";

export function HistoryPanel() {
  const { 
    history, 
    showHistory, 
    setShowHistory, 
    loadFromHistory, 
    currentResearchId,
    setHistory 
  } = useResearch();

  const deleteHistoryItem = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    setHistory((prev) => prev.filter((item) => item.id !== id));
  };

  const clearAllHistory = () => {
    if (confirm("Are you sure you want to clear all research history?")) {
      setHistory([]);
    }
  };

  return (
    <AnimatePresence>
      {showHistory && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setShowHistory(false)}
            className="fixed inset-0 bg-background/80 backdrop-blur-sm z-40"
          />

          {/* Panel */}
          <motion.div
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="fixed right-0 top-0 bottom-0 w-full max-w-md bg-surface border-l border-border z-50 flex flex-col shadow-2xl"
          >
            {/* Header */}
            <div className="p-6 border-b border-border flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-accent/10 text-accent">
                  <History className="w-5 h-5" />
                </div>
                <h2 className="text-xl font-semibold text-text-primary">Research History</h2>
              </div>
              <button
                onClick={() => setShowHistory(false)}
                className="p-2 rounded-lg hover:bg-surface-hover text-text-muted hover:text-text-primary transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {history.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-center px-6">
                  <div className="relative mb-6">
                    <div className="absolute inset-0 bg-accent/20 blur-3xl rounded-full" />
                    <div className="relative p-6 rounded-2xl bg-surface border border-border-subtle shadow-xl">
                      <History className="w-12 h-12 text-accent/40" />
                    </div>
                  </div>
                  <h3 className="text-lg font-semibold text-text-primary mb-2">No research history</h3>
                  <p className="text-text-secondary text-sm max-w-[240px]">
                    Start a new search to see your research history appear here.
                  </p>
                  <button
                    onClick={() => setShowHistory(false)}
                    className="mt-8 px-6 py-2 rounded-full bg-accent text-white font-medium hover:bg-accent-hover transition-colors shadow-lg shadow-accent/20"
                  >
                    Start Researching
                  </button>
                </div>
              ) : (
                <div className="space-y-2">
                  {history.map((item) => (
                    <motion.div
                      key={item.id}
                      layout
                      onClick={() => loadFromHistory(item)}
                      className={cn(
                        "group relative p-4 rounded-xl border transition-all duration-200 cursor-pointer",
                        currentResearchId === item.id
                          ? "bg-accent/5 border-accent/30"
                          : "bg-surface-hover/50 border-transparent hover:border-border hover:bg-surface-hover"
                      )}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex-1 min-w-0">
                          <p className={cn(
                            "text-sm font-medium truncate mb-1",
                            currentResearchId === item.id ? "text-accent" : "text-text-primary"
                          )}>
                            {item.query}
                          </p>
                          <div className="flex items-center gap-3 text-xs text-text-muted">
                            <span className="flex items-center gap-1">
                              <Clock className="w-3 h-3" />
                              {formatDistanceToNow(new Date(item.timestamp), { addSuffix: true })}
                            </span>
                            <span className="flex items-center gap-1 capitalize">
                              {item.type === 'finance' ? (
                                <TrendingUp className="w-3 h-3 text-accent" />
                              ) : (
                                <FileText className="w-3 h-3 text-success" />
                              )}
                              {item.type}
                            </span>
                          </div>
                        </div>
                        <button
                          onClick={(e) => deleteHistoryItem(e, item.id)}
                          className="opacity-0 group-hover:opacity-100 p-1.5 rounded-md hover:bg-error/10 hover:text-error text-text-muted transition-all"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </div>

            {/* Footer */}
            {history.length > 0 && (
              <div className="p-4 border-t border-border">
                <button
                  onClick={clearAllHistory}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-xl text-sm font-medium text-error hover:bg-error/10 transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                  Clear All History
                </button>
              </div>
            )}
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
