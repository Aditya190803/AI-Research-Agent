"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { useResearch } from "@/lib/context/ResearchContext";
import { MessageCircle, CheckCircle, SkipForward, ArrowRight } from "lucide-react";
import { cn } from "@/lib/utils";

export function ClarifyingPhase() {
  const {
    query,
    initialAssessment,
    clarificationQuestions,
    handleClarificationAnswer,
    handleClarificationSkip,
  } = useResearch();

  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleOptionSelect = (questionId: string, option: string) => {
    setAnswers((prev) => ({ ...prev, [questionId]: option }));
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);
    await handleClarificationAnswer(answers);
  };

  const handleSkip = async () => {
    setIsSubmitting(true);
    await handleClarificationSkip();
  };

  const answeredCount = Object.keys(answers).length;
  const totalQuestions = clarificationQuestions.length;

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass rounded-2xl p-6"
      >
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-xl bg-accent/10">
            <MessageCircle className="w-6 h-6 text-accent" />
          </div>
          <div className="flex-1">
            <h2 className="text-xl font-semibold text-text-primary mb-2">
              Let&apos;s Clarify Your Research
            </h2>
            <p className="text-text-secondary mb-4">
              To provide more tailored results, please answer a few quick questions.
            </p>
            
            {/* Initial Assessment */}
            {initialAssessment && (
              <div className="p-4 rounded-xl bg-surface border border-border-subtle">
                <p className="text-sm text-text-muted mb-1">Initial Assessment:</p>
                <p className="text-text-primary">{initialAssessment}</p>
              </div>
            )}
          </div>
        </div>
      </motion.div>

      {/* Questions */}
      <div className="space-y-4">
        {clarificationQuestions.map((question, index) => (
          <motion.div
            key={question.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 + index * 0.1 }}
            className="glass rounded-2xl p-6"
          >
            <div className="flex items-start gap-3 mb-4">
              <span className="flex items-center justify-center w-7 h-7 rounded-full bg-accent/10 text-accent text-sm font-medium">
                {index + 1}
              </span>
              <h3 className="text-text-primary font-medium flex-1">
                {question.question}
              </h3>
              {answers[question.id] && (
                <CheckCircle className="w-5 h-5 text-success" />
              )}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pl-10">
              {question.options.map((option) => (
                <button
                  key={option}
                  onClick={() => handleOptionSelect(question.id, option)}
                  className={cn(
                    "p-3 rounded-xl text-left text-sm transition-all duration-200",
                    answers[question.id] === option
                      ? "bg-accent text-white"
                      : "bg-surface hover:bg-surface-hover text-text-secondary hover:text-text-primary border border-border-subtle"
                  )}
                >
                  {option}
                </button>
              ))}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="flex items-center justify-between gap-4 pt-4"
      >
        <button
          onClick={handleSkip}
          disabled={isSubmitting}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-text-secondary 
                   hover:text-text-primary hover:bg-surface transition-all duration-200 
                   disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <SkipForward className="w-4 h-4" />
          <span>Skip Questions</span>
        </button>

        <div className="flex items-center gap-4">
          <span className="text-sm text-text-muted">
            {answeredCount} of {totalQuestions} answered
          </span>
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleSubmit}
            disabled={isSubmitting || answeredCount === 0}
            className={cn(
              "flex items-center gap-2 px-6 py-2.5 rounded-xl font-medium transition-all duration-200",
              answeredCount > 0 && !isSubmitting
                ? "bg-accent text-white hover:bg-accent-hover shadow-lg shadow-accent/20"
                : "bg-surface text-text-muted cursor-not-allowed"
            )}
          >
            <span>Continue</span>
            <ArrowRight className="w-4 h-4" />
          </motion.button>
        </div>
      </motion.div>
    </div>
  );
}
