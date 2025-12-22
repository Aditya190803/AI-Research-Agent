"use client";

import { motion } from "framer-motion";
import { Source } from "@/types";
import { ExternalLink, Globe } from "lucide-react";

interface SourceCardProps {
  source: Source;
  index: number;
}

export function SourceCard({ source, index }: SourceCardProps) {
  const getDomain = (url: string) => {
    try {
      const urlObj = new URL(url);
      return urlObj.hostname.replace("www.", "");
    } catch {
      return url;
    }
  };

  const getFaviconUrl = (url: string) => {
    const domain = getDomain(url);
    return `https://www.google.com/s2/favicons?domain=${domain}&sz=32`;
  };

  return (
    <motion.a
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className="block p-4 rounded-xl glass hover:border-accent/30 transition-all duration-200 group"
    >
      <div className="flex items-start gap-3">
        {/* Favicon */}
        <div className="w-8 h-8 rounded-lg bg-surface flex items-center justify-center flex-shrink-0 overflow-hidden">
          <img
            src={getFaviconUrl(source.url)}
            alt=""
            className="w-4 h-4"
            onError={(e) => {
              e.currentTarget.style.display = "none";
              e.currentTarget.nextElementSibling?.classList.remove("hidden");
            }}
          />
          <Globe className="w-4 h-4 text-text-muted hidden" />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <h4 className="font-medium text-text-primary group-hover:text-accent transition-colors line-clamp-2">
              {source.title}
            </h4>
            <ExternalLink className="w-4 h-4 text-text-muted flex-shrink-0 group-hover:text-accent transition-colors" />
          </div>
          <p className="text-sm text-accent/70 mt-1">{getDomain(source.url)}</p>
          <p className="text-sm text-text-muted mt-2 line-clamp-2">{source.snippet}</p>
        </div>
      </div>
    </motion.a>
  );
}
