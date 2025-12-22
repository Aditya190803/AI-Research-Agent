"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { useResearch } from "@/lib/context/ResearchContext";
import { useRouter } from "next/navigation";
import {
  Share,
  Download,
  CheckCircle,
  BookOpen,
  TrendingUp,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  RefreshCw,
  MessageSquarePlus,
  Sparkles,
  FileText,
  BarChart3,
  Plus,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { FinanceCard } from "@/components/ui/FinanceCard";
import { SourceCard } from "@/components/ui/SourceCard";
import { FollowUpInput } from "@/components/search/FollowUpInput";
import ReactMarkdown from "react-markdown";

export function CompletePhase() {
  const { result, handleRefine, resetState } = useResearch();
  const router = useRouter();
  const [showFullAnalysis, setShowFullAnalysis] = useState(false);
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState<"summary" | "analysis" | "sources">("summary");

  if (!result) return null;

  const handleNewResearch = () => {
    resetState();
    router.push("/");
  };

  const handleCopy = async () => {
    const content = `# ${result.query}\n\n## Summary\n${result.summary}\n\n## Key Findings\n${result.keyFindings?.map((f) => `- ${f}`).join("\n")}\n\n## Detailed Analysis\n${result.detailedAnalysis}`;
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  /*
  const handleShare = async () => {
    await navigator.clipboard.writeText(window.location.href);
    alert("Link copied to clipboard!");
  };
  */

  const handleCopySection = async (title: string, content: string) => {
    await navigator.clipboard.writeText(`${title}\n\n${content}`);
    alert(`${title} copied to clipboard!`);
  };

  const handleExport = () => {
    const content = `# ${result.query}\n\n## Summary\n${result.summary}\n\n## Key Findings\n${result.keyFindings?.map((f) => `- ${f}`).join("\n")}\n\n## Detailed Analysis\n${result.detailedAnalysis}`;
    const blob = new Blob([content], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `research-${result.query.slice(0, 20).replace(/\s+/g, "-")}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Success Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-success/10">
            <CheckCircle className="w-6 h-6 text-success" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-text-primary">
              Research Complete
            </h2>
            <p className="text-sm text-text-muted">
              Analyzed {result.sources?.length || 0} sources
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleNewResearch}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-accent text-white 
                     hover:bg-accent-hover transition-all duration-200 shadow-lg shadow-accent/20"
          >
            <Plus className="w-4 h-4" />
            <span className="text-sm font-medium">New Research</span>
          </motion.button>

          <div className="flex items-center gap-1 ml-2 bg-surface/50 p-1 rounded-xl border border-border-subtle">
            {/* 
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleShare}
              className="p-2 rounded-lg text-text-secondary hover:text-text-primary hover:bg-white/5 transition-all"
              title="Share Link"
            >
              <Share className="w-4 h-4" />
            </motion.button>
            */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleExport}
              className="p-2 rounded-lg text-text-secondary hover:text-text-primary hover:bg-white/5 transition-all"
              title="Export as Markdown"
            >
              <Download className="w-4 h-4" />
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleCopy}
              className="p-2 rounded-lg text-text-secondary hover:text-text-primary hover:bg-white/5 transition-all"
              title="Copy to Clipboard"
            >
              {copied ? (
                <Check className="w-4 h-4 text-success" />
              ) : (
                <Copy className="w-4 h-4" />
              )}
            </motion.button>
            <div className="w-px h-4 bg-border-subtle mx-1" />
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleRefine}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-text-secondary hover:text-text-primary hover:bg-white/5 transition-all"
            >
              <RefreshCw className="w-4 h-4" />
              <span className="text-sm font-medium hidden sm:inline">Refine</span>
            </motion.button>
          </div>
        </div>
      </motion.div>

      {/* Finance Data (if available) */}
      {result.financeData && result.financeData.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
        >
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-accent" />
            <h3 className="font-semibold text-text-primary">Financial Data</h3>
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            {result.financeData.map((data) => (
              <FinanceCard key={data.symbol} data={data} />
            ))}
          </div>
        </motion.div>
      )}

      {/* Tabs */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="flex items-center gap-2 border-b border-border-subtle pb-2 overflow-x-auto no-scrollbar"
      >
        <TabButton
          active={activeTab === "summary"}
          onClick={() => setActiveTab("summary")}
          icon={<Sparkles className="w-4 h-4" />}
          label="Summary"
        />
        <TabButton
          active={activeTab === "analysis"}
          onClick={() => setActiveTab("analysis")}
          icon={<FileText className="w-4 h-4" />}
          label="Full Analysis"
        />
        <TabButton
          active={activeTab === "sources"}
          onClick={() => setActiveTab("sources")}
          icon={<BookOpen className="w-4 h-4" />}
          label={`Sources (${result.sources?.length || 0})`}
        />
      </motion.div>

      {/* Tab Content */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
      >
        {activeTab === "summary" && (
          <div className="space-y-6">
            {/* Summary */}
            <div className="glass rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-accent" />
                  <h3 className="font-semibold text-text-primary">AI Summary</h3>
                </div>
                <button 
                  onClick={() => handleCopySection("AI Summary", result.summary || "")}
                  className="p-2 rounded-lg hover:bg-surface-hover text-text-muted hover:text-text-primary transition-colors"
                  title="Copy Summary"
                >
                  <Copy className="w-4 h-4" />
                </button>
              </div>
              <div className="prose prose-invert prose-research max-w-none">
                <ReactMarkdown>{result.summary}</ReactMarkdown>
              </div>
            </div>

            {/* Key Findings */}
            {result.keyFindings && result.keyFindings.length > 0 && (
              <div className="glass rounded-2xl p-6">
                <div className="flex items-center gap-2 mb-4">
                  <BarChart3 className="w-5 h-5 text-success" />
                  <h3 className="font-semibold text-text-primary">Key Findings</h3>
                </div>
                <div className="space-y-3">
                  {result.keyFindings.map((finding, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="flex items-start gap-3 p-4 rounded-xl bg-surface/50"
                    >
                      <span className="flex items-center justify-center w-7 h-7 rounded-full bg-accent/10 text-accent text-sm font-medium flex-shrink-0">
                        {index + 1}
                      </span>
                      <p className="text-text-secondary">{finding}</p>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === "analysis" && (
          <div className="glass rounded-2xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <FileText className="w-5 h-5 text-accent" />
                  <h3 className="font-semibold text-text-primary">Detailed Analysis</h3>
                </div>
                <button 
                  onClick={() => handleCopySection("Detailed Analysis", result.detailedAnalysis || "")}
                  className="p-2 rounded-lg hover:bg-surface-hover text-text-muted hover:text-text-primary transition-colors"
                  title="Copy Analysis"
                >
                  <Copy className="w-4 h-4" />
                </button>
              </div>
              <button
                onClick={() => setShowFullAnalysis(!showFullAnalysis)}
                className="flex items-center gap-1 text-sm text-text-muted hover:text-text-primary transition-colors"
              >
                {showFullAnalysis ? (
                  <>
                    <ChevronUp className="w-4 h-4" />
                    <span>Show less</span>
                  </>
                ) : (
                  <>
                    <ChevronDown className="w-4 h-4" />
                    <span>Show full</span>
                  </>
                )}
              </button>
            </div>
            <div
              className={cn(
                "prose prose-invert prose-research max-w-none transition-all duration-300",
                !showFullAnalysis && "max-h-[400px] overflow-hidden relative"
              )}
            >
              <ReactMarkdown>{result.detailedAnalysis}</ReactMarkdown>
              {!showFullAnalysis && (
                <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-surface to-transparent pointer-events-none" />
              )}
            </div>
          </div>
        )}

        {activeTab === "sources" && (
          <div className="space-y-4">
            {result.sources && result.sources.length > 0 ? (
              <div className="grid gap-3">
                {result.sources.map((source, index) => (
                  <SourceCard key={source.url} source={source} index={index} />
                ))}
              </div>
            ) : (
              <div className="glass rounded-2xl p-8 text-center">
                <BookOpen className="w-12 h-12 text-text-muted mx-auto mb-3" />
                <p className="text-text-secondary">No sources available</p>
              </div>
            )}
          </div>
        )}
      </motion.div>

      {/* Follow-up */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="pt-6 border-t border-border-subtle"
      >
        <div className="flex items-center gap-2 mb-4">
          <MessageSquarePlus className="w-5 h-5 text-accent" />
          <h3 className="font-semibold text-text-primary">Follow-up Research</h3>
        </div>
        <FollowUpInput />
      </motion.div>

      {/* Bottom Actions */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="mt-12 flex flex-col items-center gap-4 pb-12"
      >
        <div className="h-px w-24 bg-border-subtle" />
        <p className="text-text-muted text-sm">Want to start over?</p>
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleNewResearch}
          className="flex items-center gap-2 px-8 py-3 rounded-2xl bg-surface border border-border-subtle
                   text-text-primary hover:bg-surface-hover hover:border-accent/30 transition-all duration-200"
        >
          <Plus className="w-5 h-5 text-accent" />
          <span className="font-medium">Start New Research</span>
        </motion.button>
      </motion.div>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200",
        active
          ? "bg-accent/10 text-accent"
          : "text-text-muted hover:text-text-primary hover:bg-surface"
      )}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}
