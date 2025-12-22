"use client";

import { useKeyboardShortcuts } from "@/lib/hooks/useKeyboardShortcuts";
import { useResearch } from "@/lib/context/ResearchContext";
import { useRouter, usePathname } from "next/navigation";

export function GlobalShortcuts() {
  const { setShowHistory, showHistory } = useResearch();
  const router = useRouter();
  const pathname = usePathname();

  useKeyboardShortcuts({
    "mod+k": () => {
      if (pathname !== "/") {
        router.push("/");
      } else {
        const searchInput = document.querySelector('input[type="text"]') as HTMLInputElement;
        if (searchInput) {
          searchInput.focus();
        }
      }
    },
    "mod+h": () => {
      setShowHistory(!showHistory);
    },
    "escape": () => {
      setShowHistory(false);
    },
  });

  return null;
}
