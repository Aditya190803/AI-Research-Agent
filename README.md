# Deep Research Agent

A professional-grade AI research platform that delivers comprehensive, multi-source synthesized reports with real-time financial data integration.

## 🚀 Overview

Deep Research is designed for high-stakes information gathering. Unlike standard chat AI, it follows a rigorous research methodology: analyzing the knowledge space, clarifying intent, planning the investigation, and synthesizing findings from across the web and financial markets.

## 🛠️ Tech Stack

- **Framework**: [Next.js 15+](https://nextjs.org/) (App Router, Turbopack)
- **LLM Orchestration**: [OpenRouter](https://openrouter.ai/) (utilizing `xiaomi/mimo-v2-flash:free` for high-speed synthesis).
- **Web Search**: [LangSearch](https://langsearch.com/) for deep web crawling and information retrieval.
- **Financial Data**: [Alpha Vantage](https://www.alphavantage.co/) for real-time stock quotes and company overviews.
- **Runtime**: [Bun](https://bun.sh/) for ultra-fast development and build cycles.

## 🖥️ Interactive Research Interface

The application features a dynamic, state-aware interface that guides users through the research process:

- **Real-time Feedback**: Displays specific loading states (Exploring, Clarifying, Confirming, Researching) so users know exactly what the agent is doing.
- **Contextual Search Box**: The search input adapts to the current state. During the "Refine Scope" phase, it acts as a context-injector rather than a new search trigger.
- **History Management**: Automatically saves research sessions to local storage for quick reference.
- **Source Transparency**: Every report includes direct links to the original sources, ensuring all findings are verifiable.

## �🔄 Detailed Research Workflow

Deep Research follows a sophisticated multi-stage process to ensure accuracy and depth.

### 1. Exploration & Initial Analysis
- **Knowledge Mapping**: The agent performs an initial exploratory search to map the knowledge space and identify key themes.
- **Ambiguity Detection**: Analyzes the query for missing context or multiple interpretations.
- **Financial Trigger**: Automatically detects if the query involves companies or market data requiring Alpha Vantage integration.

### 2. Intent Alignment (Clarification)
- **Targeted Questions**: If the query is broad, the agent generates 3-5 specific questions to narrow the focus.
- **User Preferences**: Users can answer these questions or "Skip" to let the AI use its best judgment based on initial findings.
- **Initial Synthesis**: Provides a "Research Scope" summary based on the initial exploration.

### 3. Strategy & Scope Confirmation
After clarification, the agent presents a **Research Strategy** which includes:
- **Confirmed Scope**: A concise summary of what will be investigated.
- **Optimized Query**: The refined search string that will be used for deep retrieval.
- **Investigation Path**: A step-by-step plan showing the specific topics to be covered.

#### 🖱️ User Actions at this Stage:
- **Execute Deep Research ("Go")**: 
    - Confirms the current strategy.
    - Triggers the full multi-source investigation.
    - Moves directly to the final synthesis phase.
- **Refine Scope**:
    - Allows the user to provide additional instructions or context.
    - **Logic**: The system combines the previous refined query with the new user input (e.g., *"Previous Query + Additional context: [User Input]"*).
    - **Restart**: The workflow loops back to Phase 1 with the newly enriched query to re-evaluate the strategy.

### 4. Deep Investigation & Synthesis
- **Query Decomposition**: The main query is broken down into 5-7 specialized sub-queries for maximum coverage.
- **Parallel Retrieval**: Executes simultaneous searches across the web using LangSearch.
- **Data Augmentation**: Merges real-time financial metrics (if applicable) with web findings.
- **Professional Synthesis**: OpenRouter's LLM processes hundreds of data points to generate a structured, cited report.

## 📊 Report Structure

Every research report is generated with a professional layout:
- **Executive Summary**: High-level overview of the findings.
- **Key Findings**: Bulleted list of critical insights.
- **Detailed Analysis**: Deep dive into specific aspects of the research.
- **Financial Overview**: (Optional) Real-time stock data and company metrics.
- **Sources**: Numbered citations linking directly to the original web sources.

## ⚙️ Environment Setup

Create a `.env.local` file in the root directory:

```bash
# Core API Keys
OPENROUTER_API_KEY=your_key_here
LANGSEARCH_API_KEY=your_key_here
ALPHA_VANTAGE_API_KEY=your_key_here

# App Configuration
NEXT_PUBLIC_APP_URL=https://your-deployment-url.com
```

## 🏃 Getting Started

1. **Install Dependencies**:
    ```bash
    bun install
    ```

2. **Run Development Server**:
    ```bash
    bun dev
    ```

3. **Build for Production**:
    ```bash
    bun run build
    ```
