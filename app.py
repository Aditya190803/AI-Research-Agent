import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize search tool
search_tool = DuckDuckGoSearchRun()

# Replace Ollama with Gemini 2.0 Flash
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Define agents
researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI and data science',
    backstory="""You are an expert at a technology research group, 
    skilled in identifying trends and analyzing complex data.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=gemini_llm
)

writer = Agent(
    role='Tech Content Strategist',
    goal='Craft compelling content on tech advancements',
    backstory="""You are a content strategist known for 
    making complex tech topics interesting and easy to understand.""",
    verbose=True,
    allow_delegation=True,
    llm=gemini_llm
)

# Define tasks
task1 = Task(
    description="""Analyze 2024's AI advancements. 
    Find major trends, new technologies, and their effects. 
    Provide a detailed report.""",
    agent=researcher
)

task2 = Task(
    description="""Create a blog post about major AI advancements using your insights. 
    Make it interesting, clear, and suited for tech enthusiasts. 
    It should be at least 4 paragraphs long.""",
    agent=writer
)

# Instantiate Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2
)

# Streamlit UI
st.title("AI Research Crew")
if st.button("Start AI Research Crew"):
    result = crew.kickoff()
    st.subheader("Crew Results")
    st.write(result)
