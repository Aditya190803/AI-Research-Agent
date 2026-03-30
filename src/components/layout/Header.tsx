"use client";

import Link from "next/link";
import { useResearch } from "@/lib/context/ResearchContext";
import { motion } from "framer-motion";
import { History, Menu, X, Home } from "lucide-react";
import { useState } from "react";

export function Header() {
  const { showHistory, setShowHistory, resetState } = useResearch();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 glass border-b border-border-subtle">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link
            href="/"
            onClick={() => resetState()}
            className="flex items-center gap-2 group"
          >
            <motion.div
              whileHover={{ rotate: 180 }}
              transition={{ duration: 0.3 }}
              className="p-2 rounded-lg bg-accent/10"
            >
              <img
                src="/site-icon.svg"
                alt=""
                aria-hidden="true"
                className="w-5 h-5"
              />
            </motion.div>
            <span className="font-semibold text-lg text-text-primary group-hover:text-accent transition-colors">
              AI Research Agent
            </span>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-2">
            <Link
              href="/"
              onClick={() => resetState()}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-text-secondary 
                       hover:text-text-primary hover:bg-surface transition-all duration-200"
            >
              <Home className="w-4 h-4" />
              <span>Home</span>
            </Link>
            <button
              onClick={() => setShowHistory(!showHistory)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-text-secondary 
                       hover:text-text-primary hover:bg-surface transition-all duration-200"
            >
              <History className="w-4 h-4" />
              <span>History</span>
            </button>
          </nav>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 rounded-lg text-text-secondary hover:text-text-primary 
                     hover:bg-surface transition-all"
          >
            {mobileMenuOpen ? (
              <X className="w-5 h-5" />
            ) : (
              <Menu className="w-5 h-5" />
            )}
          </button>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden border-t border-border-subtle py-4"
          >
            <nav className="flex flex-col gap-2">
              <Link
                href="/"
                onClick={() => {
                  resetState();
                  setMobileMenuOpen(false);
                }}
                className="flex items-center gap-2 px-4 py-3 rounded-lg text-text-secondary 
                         hover:text-text-primary hover:bg-surface transition-all"
              >
                <Home className="w-4 h-4" />
                <span>Home</span>
              </Link>
              <button
                onClick={() => {
                  setShowHistory(!showHistory);
                  setMobileMenuOpen(false);
                }}
                className="flex items-center gap-2 px-4 py-3 rounded-lg text-text-secondary 
                         hover:text-text-primary hover:bg-surface transition-all text-left"
              >
                <History className="w-4 h-4" />
                <span>History</span>
              </button>
            </nav>
          </motion.div>
        )}
      </div>
    </header>
  );
}
