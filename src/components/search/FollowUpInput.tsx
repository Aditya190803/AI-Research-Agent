"use client";

import { useState, KeyboardEvent } from "react";
import { motion } from "framer-motion";
import { useResearch } from "@/lib/context/ResearchContext";
import { Send, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

export function FollowUpInput() {
  const { handleFollowUp, phase } = useResearch();
  const [followUpQuery, setFollowUpQuery] = useState("");
  const isLoading = ["exploring", "clarifying", "confirming", "researching"].includes(phase);

  const handleSubmit = async () => {
    if (!followUpQuery.trim() || isLoading) return;
    const query = followUpQuery;
    setFollowUpQuery("");
    await handleFollowUp(query);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="relative">
      <div className="flex items-center gap-3 p-3 rounded-xl glass gradient-border">
        <input
          type="text"
          value={followUpQuery}
          onChange={(e) => setFollowUpQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a follow-up question..."
          disabled={isLoading}
          className="flex-1 bg-transparent text-text-primary placeholder:text-text-muted 
                   outline-none disabled:opacity-50"
        />
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={handleSubmit}
          disabled={!followUpQuery.trim() || isLoading}
          className={cn(
            "p-2.5 rounded-lg transition-all duration-200",
            followUpQuery.trim() && !isLoading
              ? "bg-accent text-white hover:bg-accent-hover"
              : "bg-surface text-text-muted cursor-not-allowed"
          )}
        >
          {isLoading ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Send className="w-5 h-5" />
          )}
        </motion.button>
      </div>

      {/* Suggestions */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="mt-3 flex flex-wrap gap-2"
      >
        <SuggestionChip
          text="Explain in more detail"
          onClick={() => setFollowUpQuery("Can you explain this in more detail?")}
        />
        <SuggestionChip
          text="Compare alternatives"
          onClick={() => setFollowUpQuery("What are the alternatives and how do they compare?")}
        />
        <SuggestionChip
          text="Practical implications"
          onClick={() => setFollowUpQuery("What are the practical implications of this?")}
        />
      </motion.div>
    </div>
  );
}

function SuggestionChip({
  text,
  onClick,
}: {
  text: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="px-3 py-1.5 rounded-full text-sm text-text-muted bg-surface/50 
               hover:bg-surface hover:text-text-secondary transition-all duration-200"
    >
      {text}
    </button>
  );
}
