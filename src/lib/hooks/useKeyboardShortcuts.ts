"use client";

import { useEffect } from "react";

type ShortcutHandler = (e: KeyboardEvent) => void;

export function useKeyboardShortcuts(shortcuts: Record<string, ShortcutHandler>) {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const isMac = navigator.platform.toUpperCase().indexOf("MAC") >= 0;
      const modifier = isMac ? event.metaKey : event.ctrlKey;

      for (const [key, handler] of Object.entries(shortcuts)) {
        const parts = key.split("+");
        const targetKey = parts[parts.length - 1].toLowerCase();
        const needsModifier = parts.includes("mod");
        const needsShift = parts.includes("shift");

        if (
          event.key.toLowerCase() === targetKey &&
          (!needsModifier || modifier) &&
          (!needsShift || event.shiftKey)
        ) {
          event.preventDefault();
          handler(event);
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [shortcuts]);
}
