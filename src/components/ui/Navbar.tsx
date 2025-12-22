"use client";

import { motion } from "framer-motion";
import { useResearch } from "@/lib/context/ResearchContext";
import { History, Search, Sparkles } from "lucide-react";
import Link from "next/link";
import { cn } from "@/lib/utils";

export function Navbar() {
  const { setShowHistory } = useResearch();

  return (
    <nav className="fixed top-0 left-0 right-0 z-30 px-6 py-4">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <Link 
          href="/" 
          className="flex items-center gap-2 group"
        >
          <div className="p-2 rounded-xl bg-accent/10 text-accent group-hover:bg-accent group-hover:text-white transition-all duration-300">
            <Sparkles className="w-5 h-5" />
          </div>
          <span className="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-text-primary to-text-secondary">
            Deep Research
          </span>
        </Link>

        <div className="flex items-center gap-2">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setShowHistory(true)}
            className="flex items-center gap-2 px-3 py-2 md:px-4 md:py-2 rounded-xl glass hover:bg-surface-hover transition-all duration-200 text-text-secondary hover:text-text-primary"
          >
            <History className="w-4 h-4" />
            <span className="text-sm font-medium hidden sm:inline">History</span>
          </motion.button>
        </div>
      </div>
    </nav>
  );
}
