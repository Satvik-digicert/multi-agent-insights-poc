# trends_agent_langgraph.py

import os
import json
from datetime import datetime, timedelta
import pandas as pd
from langchain.chat_models import AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from databricks import sql  # <-- Databricks SQL connector
from rich.console import Console
from rich.markdown import Markdown

### ---- Configuration ---- ###
STATE_FILE = "trend_agent_state.json"
AZURE_OPENAI_API_KEY = ''
DATABRICKS_HOST = ''
DATABRICKS_HTTP_PATH = ''
DATABRICKS_TOKEN = ''

DATABRICKS_CONFIG = {
    "server_hostname": DATABRICKS_HOST,
    "http_path": DATABRICKS_HTTP_PATH,
    "access_token":  DATABRICKS_TOKEN
}

DATABRICKS_TABLE = "lake_prod.dcone_enterprise_gp2.tag"


### ---- Define Graph State ---- ###
class GraphState(TypedDict):
    previous_run_timestamp: Annotated[str, "Timestamp of last run"]
    new_data: Annotated[str, "JSON representation of new data"]
    trend_summary: Annotated[str, "AI-generated trend insights"]


### ---- Persistent Long-Term Memory ---- ###
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {"previous_run_timestamp": "2024-06-01T00:00:00"}


def update_state(new_state: dict):
    with open(STATE_FILE, 'w') as f:
        json.dump(new_state, f)


### ---- Databricks SQL Query ---- ###
def fetch_databricks_data(start: str, end: str) -> pd.DataFrame:
    query = f"""
    SELECT * FROM {DATABRICKS_TABLE}
    WHERE created_at > '{start}' AND created_at <= '{end}'
    """
    with sql.connect(
            server_hostname=DATABRICKS_CONFIG["server_hostname"],
            http_path=DATABRICKS_CONFIG["http_path"],
            access_token=DATABRICKS_CONFIG["access_token"],
            _tls_no_verify=True
    ) as connection:
        cursor = connection.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns)


### ---- LLM Setup ---- ###
llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    model="gpt-4o",
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint="https://navya-m5xfkegc-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview",
    api_version="2024-02-15-preview",
    temperature=0
    )


### ---- Define Nodes ---- ###
def load_previous_state(_: dict) -> GraphState:
    state = load_state()
    return {"previous_run_timestamp": state["previous_run_timestamp"]}


def fetch_new_data(state: GraphState) -> GraphState:
    start = datetime.fromisoformat(state['previous_run_timestamp'])
    end = start + timedelta(days=90)
    df = fetch_databricks_data(start.isoformat(), end.isoformat())
    return {
        **state,
        "new_data": df.to_json(orient='records')
    }


def analyze_trends(state: GraphState) -> GraphState:
    input_text = f"""
    You are a BI analyst.
    Given the new dataset below, identify any trends or anomalies:

    Data:
    {state['new_data']}
    """
    result = llm.invoke(input_text)
    return {
        **state,
        "trend_summary": result.content
    }


def update_and_end(state: GraphState) -> GraphState:
    start = datetime.fromisoformat(state["previous_run_timestamp"])
    new_timestamp = (start + timedelta(days=90)).isoformat()
    update_state({
        "previous_run_timestamp": new_timestamp,
        "trend_summary": state["trend_summary"]
    })
    return state


### ---- LangGraph Wiring ---- ###
graph = StateGraph(GraphState)
graph.add_node("load_state", load_previous_state)
graph.add_node("fetch_data", fetch_new_data)
graph.add_node("analyze", analyze_trends)
graph.add_node("update", update_and_end)

graph.set_entry_point("load_state")
graph.add_edge("load_state", "fetch_data")
graph.add_edge("fetch_data", "analyze")
graph.add_edge("analyze", "update")
graph.add_edge("update", END)

trend_graph = graph.compile()

# Setup rich consoles for HTML and Markdown logs
console = Console(record=True)
html_log_path = "rich_langgraph_log2.html"
md_log_path = "rich_langgraph_log2.md"

def persist_rich_logs():
    # Export and append HTML log
    html = console.export_html(clear=False)
    with open(html_log_path, "a") as f:
        f.write(html)
        f.write("\n<!-- --- End of Run --- -->\n")
    # Export and append Markdown log
    md = console.export_text(clear=False)
    with open(md_log_path, "a") as f:
        f.write(md)
        f.write("\n--- End of Run ---\n")


if __name__ == "__main__":
    from dateutil.relativedelta import relativedelta

    max_batches = 5
    for _ in range(max_batches):
        console.rule("[bold blue]===== Executing Batch Run =====")
        result = trend_graph.invoke({})
        console.rule("[bold blue]===== Trend Summary =====")
        console.print(Markdown(f"**Trend Summary:**\n{result['trend_summary']}"))
        console.rule("[bold green]===== State Updated To =====")
        console.print(Markdown(f"**State Updated To:**\n{result['previous_run_timestamp']}"))
        persist_rich_logs()
        if datetime.fromisoformat(result["previous_run_timestamp"]) > datetime.now():
            break