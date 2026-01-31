"use client";

import { useEffect } from "react";
import { useResearch } from "@/lib/context/ResearchContext";
import { SearchInput } from "@/components/search/SearchInput";
import { motion } from "framer-motion";
import {
  Sparkles,
  TrendingUp,
  Globe,
  Zap,
  Shield,
  Brain,
} from "lucide-react";

const features = [
  {
    icon: Brain,
    title: "AI-Powered Analysis",
    description: "Advanced reasoning for comprehensive research synthesis",
  },
  {
    icon: Globe,
    title: "Real-Time Web Search",
    description: "Access the latest information from across the internet",
  },
  {
    icon: TrendingUp,
    title: "Financial Intelligence",
    description: "Live stock data and market analysis at your fingertips",
  },
  {
    icon: Zap,
    title: "Lightning Fast",
    description: "Get deep insights in seconds, not hours",
  },
  {
    icon: Shield,
    title: "Privacy First",
    description: "Your research stays private and secure",
  },
  {
    icon: Sparkles,
    title: "Smart Follow-ups",
    description: "Refine and explore with contextual questions",
  },
];

const exampleQueries = [
  "What are the best mutual funds to invest in India for 2026?",
  "Compare Tesla vs Rivian stock performance",
  "How does quantum computing affect cybersecurity?",
  "Analyze the impact of AI on Indian IT sector",
];

export default function HomePage() {
  const { setQuery, handleSearch, resetState } = useResearch();

  // Reset state when landing on home page to ensure a fresh start
  // and avoid race conditions with search page history loading
  useEffect(() => {
    resetState();
  }, [resetState]);

  const handleExampleClick = (query: string) => {
    setQuery(query);
    setTimeout(() => handleSearch(), 100);
  };

  return (
    <div className="min-h-screen flex flex-col">
      <main className="flex-1 flex flex-col items-center justify-center px-4 pt-32 pb-20">
        <div className="w-full max-w-4xl mx-auto">
          {/* Hero Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.1, duration: 0.5 }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass mb-6"
            >
              <Sparkles className="w-4 h-4 text-accent" />
              <span className="text-sm text-text-secondary">
                AI-Powered Research Agent
              </span>
            </motion.div>

            <h1 className="text-4xl md:text-6xl font-bold mb-4 tracking-tight">
              <span className="gradient-text">Research Smarter,</span>
              <br />
              <span className="text-text-primary">Not Harder</span>
            </h1>

            <p className="text-lg md:text-xl text-text-secondary max-w-2xl mx-auto">
              Get comprehensive, AI-synthesized research with real-time data,
              expert analysis, and actionable insights.
            </p>
          </motion.div>

          {/* Search Input */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
          >
            <SearchInput variant="home" />
          </motion.div>

          {/* Example Queries */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4, duration: 0.5 }}
            className="mt-8"
          >
            <p className="text-sm text-text-muted text-center mb-4">
              Try these examples:
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {exampleQueries.map((query, index) => (
                <motion.button
                  key={query}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.5 + index * 0.1 }}
                  onClick={() => handleExampleClick(query)}
                  className="px-4 py-2 rounded-full glass glass-hover text-sm text-text-secondary
                           hover:text-text-primary transition-all duration-200 cursor-pointer"
                >
                  {query.length > 40 ? query.slice(0, 40) + "..." : query}
                </motion.button>
              ))}
            </div>
          </motion.div>

          {/* Features Grid */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6, duration: 0.6 }}
            className="mt-20 grid grid-cols-2 md:grid-cols-3 gap-4"
          >
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 + index * 0.1 }}
                className="p-5 rounded-xl glass group hover:border-accent/30 transition-all duration-300"
              >
                <feature.icon className="w-8 h-8 text-accent mb-3 group-hover:scale-110 transition-transform" />
                <h3 className="font-semibold text-text-primary mb-1">
                  {feature.title}
                </h3>
                <p className="text-sm text-text-muted">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </main>
    </div>
  );
}
