"use client";

import { motion } from "framer-motion";
import { useResearch } from "@/lib/context/ResearchContext";
import { Loader2, Search, Brain, Sparkles } from "lucide-react";

export function ExploringPhase() {
  const { query } = useResearch();

  return (
    <div className="max-w-2xl mx-auto">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="glass rounded-2xl p-8 text-center"
      >
        {/* Animated Icon */}
        <motion.div
          animate={{
            scale: [1, 1.1, 1],
            rotate: [0, 5, -5, 0],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut",
          }}
          className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-accent/10 mb-6"
        >
          <Brain className="w-10 h-10 text-accent" />
        </motion.div>

        <h2 className="text-2xl font-semibold text-text-primary mb-2">
          Exploring Your Query
        </h2>
        <p className="text-text-secondary mb-8 max-w-md mx-auto">
          Analyzing your research topic and identifying key areas to investigate
        </p>

        {/* Progress Steps */}
        <div className="space-y-3">
          <ProgressStep
            icon={<Search className="w-4 h-4" />}
            text="Understanding your query"
            isActive
          />
          <ProgressStep
            icon={<Brain className="w-4 h-4" />}
            text="Identifying key research areas"
            isActive={false}
          />
          <ProgressStep
            icon={<Sparkles className="w-4 h-4" />}
            text="Preparing clarifying questions"
            isActive={false}
          />
        </div>
      </motion.div>
    </div>
  );
}

function ProgressStep({
  icon,
  text,
  isActive,
}: {
  icon: React.ReactNode;
  text: string;
  isActive: boolean;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      className={`flex items-center gap-3 p-3 rounded-lg transition-all duration-200 ${
        isActive ? "bg-accent/10" : "bg-surface/50"
      }`}
    >
      <div
        className={`flex items-center justify-center w-8 h-8 rounded-lg ${
          isActive ? "bg-accent text-white" : "bg-surface text-text-muted"
        }`}
      >
        {isActive ? <Loader2 className="w-4 h-4 animate-spin" /> : icon}
      </div>
      <span
        className={`font-medium ${isActive ? "text-text-primary" : "text-text-muted"}`}
      >
        {text}
      </span>
    </motion.div>
  );
}
