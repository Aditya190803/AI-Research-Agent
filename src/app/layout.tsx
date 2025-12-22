import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { ResearchProvider } from "@/lib/context/ResearchContext";
import { Navbar } from "@/components/ui/Navbar";
import { HistoryPanel } from "@/components/search/HistoryPanel";
import { GlobalShortcuts } from "@/components/ui/GlobalShortcuts";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Deep Research | AI-Powered Research Assistant",
  description:
    "Experience the future of research with AI-powered deep analysis, real-time data synthesis, and comprehensive insights.",
  keywords: [
    "AI research",
    "deep research",
    "research assistant",
    "AI analysis",
    "stock research",
    "market analysis",
  ],
  authors: [{ name: "Deep Research" }],
  openGraph: {
    title: "Deep Research | AI-Powered Research Assistant",
    description:
      "Experience the future of research with AI-powered deep analysis",
    type: "website",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#1a1a2e",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body className="antialiased">
        <ResearchProvider>
          <GlobalShortcuts />
          <div className="relative min-h-screen">
            <Navbar />
            <HistoryPanel />
            {/* Ambient background effects */}
            <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
              <div className="absolute top-0 left-1/4 w-[500px] h-[500px] bg-accent/5 rounded-full blur-[120px]" />
              <div className="absolute bottom-0 right-1/4 w-[400px] h-[400px] bg-accent/3 rounded-full blur-[100px]" />
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-accent/2 rounded-full blur-[150px]" />
            </div>
            {children}
          </div>
        </ResearchProvider>
      </body>
    </html>
  );
}
