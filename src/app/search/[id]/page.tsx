"use client";

import { useParams } from "next/navigation";
import { useResearch } from "@/lib/context/ResearchContext";
import {
  ExploringPhase,
  ClarifyingPhase,
  ConfirmingPhase,
  ResearchingPhase,
  CompletePhase,
  ErrorPhase,
} from "@/components/phases";
import { motion, AnimatePresence } from "framer-motion";
import { useEffect } from "react";

export default function SearchPage() {
  const params = useParams();
  const {
    query,
    phase,
    history,
    currentResearchId,
    loadFromHistory,
  } = useResearch();

  // Load from history if visiting a different research ID
  useEffect(() => {
    const researchId = params.id as string;
    if (researchId && researchId !== currentResearchId) {
      const historyItem = history.find((h) => h.id === researchId);
      if (historyItem) {
        loadFromHistory(historyItem);
      }
    }
  }, [params.id, currentResearchId, history, loadFromHistory]);

  const renderPhase = () => {
    switch (phase) {
      case "exploring":
        return <ExploringPhase />;
      case "clarifying":
        return <ClarifyingPhase />;
      case "confirming":
        return <ConfirmingPhase />;
      case "researching":
        return <ResearchingPhase />;
      case "complete":
        return <CompletePhase />;
      case "error":
        return <ErrorPhase />;
      default:
        return (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex flex-col items-center justify-center py-20"
          >
            <p className="text-text-muted">Start a new research query above</p>
          </motion.div>
        );
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <main className="flex-1 pt-24 pb-8">
        <div className="max-w-5xl mx-auto px-4">
          {/* Static Query Title */}
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-12 text-center"
          >
            <h1 className="text-2xl md:text-3xl font-bold text-text-primary tracking-tight max-w-3xl mx-auto leading-tight">
              {query}
            </h1>
            <div className="mt-4 flex items-center justify-center gap-2">
              <div className="h-1 w-12 rounded-full bg-accent/30" />
              <div className="h-1 w-1 rounded-full bg-accent/50" />
              <div className="h-1 w-12 rounded-full bg-accent/30" />
            </div>
          </motion.div>

          {/* Phase Content */}
          <AnimatePresence mode="wait">
            <motion.div
              key={phase}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              {renderPhase()}
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}
