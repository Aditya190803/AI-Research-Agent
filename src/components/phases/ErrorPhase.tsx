"use client";

import { motion } from "framer-motion";
import { useResearch } from "@/lib/context/ResearchContext";
import { AlertCircle, RefreshCw, Home } from "lucide-react";
import Link from "next/link";

export function ErrorPhase() {
  const { error, query, resetState, handleSearch } = useResearch();

  const handleRetry = () => {
    handleSearch();
  };

  return (
    <div className="max-w-xl mx-auto">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="glass rounded-2xl p-8 text-center"
      >
        {/* Error Icon */}
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", damping: 10, stiffness: 100 }}
          className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-error/10 mb-6"
        >
          <AlertCircle className="w-10 h-10 text-error" />
        </motion.div>

        <h2 className="text-2xl font-semibold text-text-primary mb-2">
          Something Went Wrong
        </h2>
        <p className="text-text-secondary mb-6 max-w-md mx-auto">
          {error || "We encountered an error while processing your research request."}
        </p>

        {/* Query that failed */}
        {query && (
          <div className="p-4 rounded-xl bg-surface/50 border border-border-subtle mb-8">
            <p className="text-sm text-text-muted mb-1">Your query:</p>
            <p className="text-text-primary">{query}</p>
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center justify-center gap-4">
          <Link
            href="/"
            onClick={() => resetState()}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-text-secondary 
                     hover:text-text-primary hover:bg-surface transition-all duration-200"
          >
            <Home className="w-4 h-4" />
            <span>Go Home</span>
          </Link>

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleRetry}
            className="flex items-center gap-2 px-6 py-2.5 rounded-xl font-medium 
                     bg-accent text-white hover:bg-accent-hover shadow-lg shadow-accent/20 
                     transition-all duration-200"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Try Again</span>
          </motion.button>
        </div>

        {/* Help Text */}
        <p className="text-sm text-text-muted mt-8">
          If this problem persists, try simplifying your query or check your internet connection.
        </p>
      </motion.div>
    </div>
  );
}
