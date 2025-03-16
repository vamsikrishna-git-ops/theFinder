import os
import requests
import logging
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from langgraph.graph import StateGraph
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, TypedDict

# Set up logging to a file
logging.basicConfig(
    filename="serpapi_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Get SerpAPI key from environment
SERPAPI_API_KEY = "fac73487772ade675f8d6cc41b8c3df45a7e4f74ee45751be4f376983eaf5741"
if not SERPAPI_API_KEY:
    raise ValueError("Please set SERPAPI_API_KEY environment variable.")

# Define the state structure using TypedDict
class AgentState(TypedDict):
    query: str
    urls: list
    text: str
    summary: str

# Search function using SerpAPI (no location, whole web)
def search(state: AgentState) -> AgentState:
    params = {
        "q": state["query"],
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "api_key": SERPAPI_API_KEY,
        "num": 3  # Limit to top 3 results
    }
    search = GoogleSearch(params)
    results = search.get_dict().get("organic_results", [])
    state["urls"] = [result["link"] for result in results if "link" in result]
    
    # Log SerpAPI results
    logging.info(f"Query: {state['query']}, URLs: {state['urls']}")
    return state

# Scrape function using BeautifulSoup
def scrape(state: AgentState) -> AgentState:
    scraped_text = ""
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in state["urls"]:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator=" ")
            scraped_text += f"\n\nContent from {url}:\n{text.strip()}"
        except Exception as e:
            scraped_text += f"\n\nFailed to scrape {url}: {str(e)}"
    state["text"] = scraped_text
    return state

# Summarize function using DeepSeek-V3
def summarize(state: AgentState) -> AgentState:
    # Load model and tokenizer (downloads on first run)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
    
    # Prepare prompt for summarization
    prompt = (
        f"Summarize the following text to answer the query '{state['query']}':\n\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    
    # Generate summary
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,  # Limit summary length
        do_sample=False,     # Greedy decoding for consistency
        pad_token_id=tokenizer.eos_token_id
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the summary part (after the prompt)
    summary = summary[len(prompt):].strip()
    state["summary"] = summary
    return state

# Define the nodes
def search_node(state: AgentState) -> AgentState:
    return search(state)

def scrape_node(state: AgentState) -> AgentState:
    return scrape(state)

def summarize_node(state: AgentState) -> AgentState:
    return summarize(state)

# Build the LangGraph with state_schema
graph = StateGraph(state_schema=AgentState)
graph.add_node("search", search_node)
graph.add_node("scrape", scrape_node)
graph.add_node("summarize", summarize_node)
graph.add_edge("search", "scrape")
graph.add_edge("scrape", "summarize")
graph.set_entry_point("search")
compiled_graph = graph.compile()

# Function to run the agent
def run_agent(query: str) -> str:
    initial_state = AgentState(query=query, urls=[], text="", summary="")
    final_state = compiled_graph.invoke(initial_state)
    return final_state["summary"]

# Main loop to take user input and process queries
if __name__ == "__main__":
    print("Web Scraper Agent: Enter a query to search the web (or 'exit' to stop)")
    while True:
        query = input("Enter your query: ").strip()
        if query.lower() == "exit":
            print("Exiting Web Scraper Agent. Goodbye!")
            break
        if not query:
            print("Please enter a valid query.")
            continue
        
        print(f"\nProcessing Query: {query}")
        try:
            summary = run_agent(query)
            print(f"Results:\n{summary}\n")
        except Exception as e:
            print(f"Error processing query: {str(e)}\n")
        
        print("Ready for next query...")