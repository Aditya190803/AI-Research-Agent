"use client";

import { cn } from "@/lib/utils";

interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className }: SkeletonProps) {
  return (
    <div
      className={cn(
        "shimmer rounded-lg bg-surface",
        className
      )}
    />
  );
}

export function SkeletonText({ className }: SkeletonProps) {
  return (
    <div className={cn("space-y-2", className)}>
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-5/6" />
      <Skeleton className="h-4 w-4/6" />
    </div>
  );
}

export function SkeletonCard({ className }: SkeletonProps) {
  return (
    <div className={cn("p-6 rounded-2xl bg-surface/50 border border-border-subtle", className)}>
      <div className="flex items-center gap-3 mb-4">
        <Skeleton className="w-10 h-10 rounded-xl" />
        <div className="flex-1 space-y-2">
          <Skeleton className="h-4 w-1/3" />
          <Skeleton className="h-3 w-1/4" />
        </div>
      </div>
      <SkeletonText />
    </div>
  );
}

export function FinanceCardSkeleton() {
  return (
    <div className="glass rounded-2xl p-5 space-y-4">
      <div className="flex items-start justify-between">
        <div className="space-y-2 flex-1">
          <Skeleton className="h-6 w-20" />
          <Skeleton className="h-4 w-32" />
        </div>
        <Skeleton className="h-8 w-16 rounded-full" />
      </div>
      <Skeleton className="h-10 w-32" />
      <div className="grid grid-cols-2 gap-3">
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-12 w-full" />
      </div>
    </div>
  );
}

export function SourceCardSkeleton() {
  return (
    <div className="glass rounded-xl p-4 space-y-3">
      <div className="flex items-center gap-3">
        <Skeleton className="w-5 h-5 rounded-md" />
        <Skeleton className="h-5 w-48" />
      </div>
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-2/3" />
      <div className="flex items-center gap-2">
        <Skeleton className="h-3 w-32" />
      </div>
    </div>
  );
}
