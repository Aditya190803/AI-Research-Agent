"use client";

import { ResearchResult } from "@/types";
import { useResearch } from "@/lib/context/ResearchContext";
import { motion, AnimatePresence } from "framer-motion";
import { X, Clock, CheckCircle, AlertCircle, Loader2, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface HistoryPanelProps {
  isOpen: boolean;
  onClose: () => void;
  history: ResearchResult[];
}

function formatTimeAgo(date: Date): string {
  const now = new Date();
  const seconds = Math.floor((now.getTime() - new Date(date).getTime()) / 1000);

  if (seconds < 60) return "Just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;
  return new Date(date).toLocaleDateString();
}

function getPhaseIcon(phase: string) {
  switch (phase) {
    case "complete":
      return <CheckCircle className="w-4 h-4 text-success" />;
    case "error":
      return <AlertCircle className="w-4 h-4 text-error" />;
    case "exploring":
    case "clarifying":
    case "confirming":
    case "researching":
      return <Loader2 className="w-4 h-4 text-accent animate-spin" />;
    default:
      return <Clock className="w-4 h-4 text-text-muted" />;
  }
}

export function HistoryPanel({ isOpen, onClose, history }: HistoryPanelProps) {
  const { loadFromHistory } = useResearch();

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
          />

          {/* Panel */}
          <motion.div
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="fixed top-0 right-0 bottom-0 w-full max-w-md bg-background border-l border-border z-50 
                     flex flex-col shadow-2xl"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h2 className="text-lg font-semibold text-text-primary">Research History</h2>
              <button
                onClick={onClose}
                className="p-2 rounded-lg text-text-muted hover:text-text-primary hover:bg-surface 
                         transition-all duration-200"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* History List */}
            <div className="flex-1 overflow-y-auto">
              {history.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full p-8 text-center">
                  <Clock className="w-12 h-12 text-text-muted mb-4" />
                  <p className="text-text-secondary font-medium">No research history</p>
                  <p className="text-sm text-text-muted mt-1">
                    Your research queries will appear here
                  </p>
                </div>
              ) : (
                <div className="p-2">
                  {history.map((item, index) => (
                    <motion.button
                      key={item.id}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                      onClick={() => loadFromHistory(item)}
                      className={cn(
                        "w-full p-4 rounded-xl text-left transition-all duration-200",
                        "hover:bg-surface group"
                      )}
                    >
                      <div className="flex items-start gap-3">
                        <div className="mt-1">{getPhaseIcon(item.phase)}</div>
                        <div className="flex-1 min-w-0">
                          <p className="text-text-primary font-medium line-clamp-2 group-hover:text-accent transition-colors">
                            {item.query}
                          </p>
                          <div className="flex items-center gap-2 mt-2">
                            <span
                              className={cn(
                                "text-xs px-2 py-0.5 rounded-full",
                                item.type === "finance"
                                  ? "bg-success/10 text-success"
                                  : "bg-accent/10 text-accent"
                              )}
                            >
                              {item.type === "finance" ? "Finance" : "General"}
                            </span>
                            <span className="text-xs text-text-muted">
                              {formatTimeAgo(item.timestamp)}
                            </span>
                          </div>
                        </div>
                        <ChevronRight className="w-4 h-4 text-text-muted group-hover:text-accent transition-colors" />
                      </div>
                    </motion.button>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
