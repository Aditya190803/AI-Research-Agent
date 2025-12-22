"use client";

import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  ResearchPhase,
  ResearchResult,
  ResearchStrategy,
  ClarificationQuestion,
  SearchResult,
} from "@/types";
import {
  generateResearchId,
  saveToLocalStorage,
  loadFromLocalStorage,
  getFriendlyErrorMessage,
} from "@/lib/utils";
import { MAX_HISTORY_ITEMS } from "@/lib/constants";
import { researchClient } from "@/lib/api/client";

interface ResearchContextType {
  query: string;
  setQuery: (query: string) => void;
  phase: ResearchPhase;
  error: string | null;
  clarificationQuestions: ClarificationQuestion[];
  initialAssessment: string;
  strategy: ResearchStrategy | null;
  result: ResearchResult | null;
  currentResearchId: string | null;
  isRefining: boolean;
  setIsRefining: (isRefining: boolean) => void;
  refinedContext: string;
  setRefinedContext: (context: string) => void;
  history: ResearchResult[];
  setHistory: React.Dispatch<React.SetStateAction<ResearchResult[]>>;
  showHistory: boolean;
  setShowHistory: (show: boolean) => void;
  handleSearch: () => Promise<void>;
  handleClarificationAnswer: (answers: Record<string, string>) => Promise<void>;
  handleClarificationSkip: () => Promise<void>;
  executeDeepResearch: () => Promise<void>;
  handleRefine: () => void;
  handleRefineSubmit: () => Promise<void>;
  resetState: () => void;
  loadFromHistory: (result: ResearchResult) => void;
  cancelResearch: () => void;
  handleFollowUp: (followUpQuery: string) => Promise<void>;
}

const ResearchContext = createContext<ResearchContextType | undefined>(undefined);

export function ResearchProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [phase, setPhase] = useState<ResearchPhase>("idle");
  const [error, setError] = useState<string | null>(null);

  const [clarificationQuestions, setClarificationQuestions] = useState<ClarificationQuestion[]>([]);
  const [initialAssessment, setInitialAssessment] = useState("");
  const [strategy, setStrategy] = useState<ResearchStrategy | null>(null);
  const [result, setResult] = useState<ResearchResult | null>(null);
  const [currentResearchId, setCurrentResearchId] = useState<string | null>(null);
  const [isRefining, setIsRefining] = useState(false);
  const [refinedContext, setRefinedContext] = useState("");

  const [history, setHistory] = useState<ResearchResult[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  
  const abortControllerRef = useRef<AbortController | null>(null);

  // Load history from localStorage on mount (client-side only)
  useEffect(() => {
    const savedHistory = loadFromLocalStorage();
    setHistory(savedHistory);
  }, []);

  const updateHistory = useCallback((updates: Partial<ResearchResult>) => {
    setHistory((prev) => {
      const index = prev.findIndex((h) => h.id === currentResearchId);
      if (index === -1) return prev;
      
      const newHistory = [...prev];
      newHistory[index] = { ...newHistory[index], ...updates };
      return newHistory;
    });
  }, [currentResearchId]);

  // Save history to localStorage whenever it changes
  useEffect(() => {
    saveToLocalStorage(history);
  }, [history]);

  const resetState = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setQuery("");
    setPhase("idle");
    setError(null);
    setClarificationQuestions([]);
    setInitialAssessment("");
    setStrategy(null);
    setResult(null);
    setCurrentResearchId(null);
    setIsRefining(false);
    setRefinedContext("");
  }, []);

  const cancelResearch = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setPhase("idle");
    updateHistory({ phase: "idle" });
  }, [updateHistory]);

  const generateStrategy = useCallback(async (q: string, context: string) => {
    setPhase("confirming");
    updateHistory({ phase: "confirming" });
    
    abortControllerRef.current = new AbortController();

    try {
      const strategyData = await researchClient.generateStrategy(q, context, abortControllerRef.current.signal);
      setStrategy(strategyData);
      updateHistory({ strategy: strategyData });
    } catch (err) {
      const errorMessage = getFriendlyErrorMessage(err instanceof Error ? err.message : "An error occurred");
      setError(errorMessage);
      setPhase("error");
      updateHistory({ phase: "error", error: errorMessage });
    }
  }, [updateHistory]);

  const handleFollowUp = useCallback(async (followUpQuery: string) => {
    if (!followUpQuery.trim() || !result) return;

    const researchId = generateResearchId();
    setCurrentResearchId(researchId);
    
    const initialResearch: ResearchResult = {
      id: researchId,
      type: "general",
      query: followUpQuery,
      timestamp: new Date(),
      phase: "exploring",
    };

    setHistory((prev) => [initialResearch, ...prev].slice(0, MAX_HISTORY_ITEMS));
    router.push(`/search/${researchId}`);

    setPhase("exploring");
    
    const context = `Previous research on: "${result.query}". Summary: ${result.summary || ""}. Key findings: ${(result.keyFindings || []).join(", ")}`;

    abortControllerRef.current = new AbortController();

    try {
      const exploreData = await researchClient.explore(followUpQuery, context, abortControllerRef.current.signal);
      setInitialAssessment(exploreData.assessment);

      if (exploreData.needsClarification && exploreData.clarificationQuestions?.length > 0) {
        setPhase("clarifying");
        setClarificationQuestions(exploreData.clarificationQuestions);
        updateHistory({ 
          phase: "clarifying", 
          initialAssessment: exploreData.assessment,
          clarificationQuestions: exploreData.clarificationQuestions 
        });
      } else {
        updateHistory({ 
          phase: "confirming", 
          initialAssessment: exploreData.assessment 
        });
        await generateStrategy(followUpQuery, context);
      }
    } catch (err) {
      const errorMessage = getFriendlyErrorMessage(err instanceof Error ? err.message : "An error occurred");
      setError(errorMessage);
      setPhase("error");
      updateHistory({ phase: "error", error: errorMessage });
    }
  }, [result, router, generateStrategy, updateHistory]);

  const handleSearch = useCallback(async () => {
    if (!query.trim()) return;

    const researchId = generateResearchId();
    setCurrentResearchId(researchId);
    
    const initialResearch: ResearchResult = {
      id: researchId,
      type: "general", // Default type, will be updated later
      query: query,
      timestamp: new Date(),
      phase: "exploring",
    };

    setHistory((prev) => [initialResearch, ...prev].slice(0, MAX_HISTORY_ITEMS));
    router.push(`/search/${researchId}`);

    setPhase("exploring");

    abortControllerRef.current = new AbortController();

    try {
      const exploreData = await researchClient.explore(query, undefined, abortControllerRef.current.signal);
      setInitialAssessment(exploreData.assessment);

      if (exploreData.needsClarification && exploreData.clarificationQuestions?.length > 0) {
        setPhase("clarifying");
        setClarificationQuestions(exploreData.clarificationQuestions);
        updateHistory({ 
          phase: "clarifying", 
          initialAssessment: exploreData.assessment,
          clarificationQuestions: exploreData.clarificationQuestions 
        });
      } else {
        updateHistory({ 
          phase: "confirming", 
          initialAssessment: exploreData.assessment 
        });
        await generateStrategy(query, "");
      }
    } catch (err) {
      const errorMessage = getFriendlyErrorMessage(err instanceof Error ? err.message : "An error occurred");
      setError(errorMessage);
      setPhase("error");
      updateHistory({ phase: "error", error: errorMessage });
    }
  }, [query, router, generateStrategy, updateHistory]);

  const handleClarificationAnswer = useCallback(async (answers: Record<string, string>) => {
    const context = Object.values(answers).filter(Boolean).join(". ");
    await generateStrategy(query, context);
  }, [query, generateStrategy]);

  const handleClarificationSkip = useCallback(async () => {
    await generateStrategy(query, "");
  }, [query, generateStrategy]);

  const executeDeepResearch = useCallback(async () => {
    if (!strategy) return;

    setPhase("researching");
    updateHistory({ phase: "researching" });
    
    abortControllerRef.current = new AbortController();

    try {
      const data = await researchClient.executeResearch(strategy, abortControllerRef.current.signal);

      const researchResult: Partial<ResearchResult> = {
        type: data.type,
        query: strategy.optimizedQuery,
        summary: data.summary,
        keyFindings: data.keyFindings || [],
        detailedAnalysis: data.detailedAnalysis,
        sources: (data.searchResults || []).map((r: SearchResult) => ({
          title: r.title,
          url: r.url,
          snippet: r.content.slice(0, 200),
        })),
        financeData: data.financeData,
        phase: "complete",
      };

      // Update local state
      setResult({ ...researchResult, id: currentResearchId!, timestamp: new Date() } as ResearchResult);
      setPhase("complete");
      
      // Update history
      updateHistory(researchResult);
    } catch (err) {
      const errorMessage = getFriendlyErrorMessage(err instanceof Error ? err.message : "An error occurred");
      setError(errorMessage);
      setPhase("error");
      updateHistory({ phase: "error", error: errorMessage });
    }
  }, [strategy, currentResearchId, updateHistory]);

  const handleRefine = useCallback(() => {
    setIsRefining(true);
    updateHistory({ phase: "idle" }); // Or a new "refining" phase
  }, [updateHistory]);

  const handleRefineSubmit = useCallback(async () => {
    if (!refinedContext.trim() || !strategy) return;

    const newQuery = `${strategy.optimizedQuery}. Additional context: ${refinedContext}`;
    setQuery(newQuery);
    setRefinedContext("");
    setIsRefining(false);

    setPhase("exploring");
    updateHistory({ query: newQuery, phase: "exploring" });
    
    abortControllerRef.current = new AbortController();

    try {
      const exploreData = await researchClient.explore(newQuery, undefined, abortControllerRef.current.signal);
      setInitialAssessment(exploreData.assessment);
      updateHistory({ initialAssessment: exploreData.assessment });
      await generateStrategy(newQuery, "");
    } catch (err) {
      const errorMessage = getFriendlyErrorMessage(err instanceof Error ? err.message : "An error occurred");
      setError(errorMessage);
      setPhase("error");
      updateHistory({ phase: "error", error: errorMessage });
    }
  }, [refinedContext, strategy, generateStrategy, updateHistory]);

  const loadFromHistory = useCallback((historyResult: ResearchResult) => {
    setShowHistory(false);
    
    // Set context state from history
    setQuery(historyResult.query);
    setPhase(historyResult.phase);
    setError(historyResult.error || null);
    setClarificationQuestions(historyResult.clarificationQuestions || []);
    setInitialAssessment(historyResult.initialAssessment || "");
    setStrategy(historyResult.strategy || null);
    setResult(historyResult.phase === "complete" ? historyResult : null);
    setCurrentResearchId(historyResult.id);
    setIsRefining(false);
    setRefinedContext("");

    router.push(`/search/${historyResult.id}`);
  }, [router]);

  const value = {
    query,
    setQuery,
    phase,
    error,
    clarificationQuestions,
    initialAssessment,
    strategy,
    result,
    currentResearchId,
    isRefining,
    setIsRefining,
    refinedContext,
    setRefinedContext,
    history,
    setHistory,
    showHistory,
    setShowHistory,
    handleSearch,
    handleClarificationAnswer,
    handleClarificationSkip,
    executeDeepResearch,
    handleRefine,
    handleRefineSubmit,
    resetState,
    loadFromHistory,
    cancelResearch,
    handleFollowUp,
  };

  return <ResearchContext.Provider value={value}>{children}</ResearchContext.Provider>;
}

export function useResearch() {
  const context = useContext(ResearchContext);
  if (context === undefined) {
    throw new Error("useResearch must be used within a ResearchProvider");
  }
  return context;
}
