# multi_agent_insights.py

import os
from dotenv import load_dotenv  # Add this import

import pandas as pd
from typing import TypedDict, Annotated, List
from databricks import sql
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from app.utils import (
    generate_visualization,
    image_to_base64,
)
from app.prompts import (TABLES_INFO , QUERY_AGENT_PROMPT, INSIGHTS_PROMPT)
import gradio as gr
import base64


### ---- Configuration ---- ###
load_dotenv()  # Load environment variables from .env file

AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
DATABRICKS_HOST = os.environ.get('DATABRICKS_HOST')
DATABRICKS_HTTP_PATH = os.environ.get('DATABRICKS_HTTP_PATH')
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')

TEMPERATURE= os.environ.get('TEMPERATURE')
AZURE_DEPLOYMENT= os.environ.get('AZURE_DEPLOYMENT')
AZURE_ENDPOINT= os.environ.get('AZURE_ENDPOINT')
AZURE_API_VERSION= os.environ.get('AZURE_API_VERSION')


DATABRICKS_CONFIG = {
    "server_hostname": DATABRICKS_HOST,
    "http_path": DATABRICKS_HTTP_PATH,
    "access_token":  DATABRICKS_TOKEN
}

### ---- Graph State ---- ###
class GraphState(TypedDict):
    queries_executed: Annotated[List[str], "List of executed SQL queries"]
    insights: Annotated[List[dict], "List of dicts: {'query': str, 'visualization': str, 'shape': tuple}"]

### ---- LLM Setup ---- ###

llm = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
    temperature=TEMPERATURE
)


def execute_query(query: str, DATABRICKS_CONFIG=None) -> pd.DataFrame:
    if DATABRICKS_CONFIG is None:
        raise ValueError("DATABRICKS_CONFIG must be provided")
    with sql.connect(
        server_hostname=DATABRICKS_CONFIG["server_hostname"],
        http_path=DATABRICKS_CONFIG["http_path"],
        access_token=DATABRICKS_CONFIG["access_token"],
        _tls_no_verify=True
    ) as connection:
        cursor = connection.cursor()
        print(f"[QUERY LOG] Executing SQL Query: {query}")
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        print(f"[QUERY LOG] DataFrame shape: {df.shape}")
        return df

### ---- Agent: Query Generator ---- ###
def query_agent(_: dict) -> GraphState:
    prompt = QUERY_AGENT_PROMPT
    print(f"[QUERY AGENT] Query Agent: {prompt}")
    response = llm.invoke(prompt)
    print("=== LLM Query Generation Response ===")
    queries = response.content.strip().split("#####")
    print(f' queries : {queries}')

    results = []
    executed_queries = []
    for query in queries:
        try:
            df = execute_query(query, DATABRICKS_CONFIG)
            results.append({"query": query, "result": df})
            executed_queries.append(query)
        except Exception as e:
            print(f"[QUERY ERROR] Query failed: {query}\nError: {str(e)}")
            continue

    # Only keep queries_executed; insights will be filled by later agents
    return {
        "queries_executed": executed_queries,
        "insights": [
            {"query": item["query"], "result": item["result"]} for item in results
        ]
    }

### ---- Agent: Visualization Generator ---- ###
def visualization_agent(state: GraphState) -> GraphState:
    """
    For each query in insights, generate a visualization and add the image path and df.shape.
    """
    new_insights = []
    for item in state["insights"]:
        df = item["result"]
        query = item["query"]
        query_hash = str(abs(hash(query)))[:8]
        img_filename = generate_visualization(df, query_hash)
        print(f"[VISUALIZATION AGENT] Visualization Image: {img_filename}")
        new_insights.append({
            "query": query,
            "visualization": img_filename,
            "shape": df.shape,
            "result": df  # Keep df for next step
        })
    return {
        **state,
        "insights": new_insights
    }

### ---- Agent: Insight Generator ---- ###
def insight_generator(state: GraphState) -> GraphState:
    insights = []
    for item in state["insights"]:
        query = item["query"]
        df = item["result"]
        img_filename = item.get("visualization")
        print(f' img_filename in insight_generator 1 : {img_filename}')
        shape = item.get("shape")
        print(f' img_filename in insight_generator  2 : {img_filename}')

        # Safely convert DataFrame to JSON, handling datetime overflow
        try:
            df_json = df.to_json(orient='records')[:6000]
        except OverflowError:
            # Fall back to string representation if JSON conversion fails
            df_json = df.head(20).to_string()
            print("[WARNING] DataFrame contains datetime values that couldn't be serialized to JSON")

        prompt = f"""
        
        {INSIGHTS_PROMPT}
 
        SQL Query:
        {query}

        Result:
        {df_json}

        Format:
        Insight: <your insight with Summary , Supporting Evidence (stat/metric) and Recommended Action>
        """
        print(f' insight prompt : {prompt}')
        print(f' insight : {df.head(5)} ')

        result = llm.invoke(prompt)
        try:
            print(f' insight result : {result.content}')
            insight = result.content
            print(f' insight generated : {insight}')
            insights.append({
                "query": query,
                "insight": insight,
                "visualization": img_filename,
                "shape": shape if shape is not None else df.shape
            })
            print("=== Insight Generated ===")
            """
                if img_filename and os.path.exists(img_filename):
                b64img = image_to_base64(img_filename)
                md_img = f'<img src="data:image/png;base64,{b64img}" alt="Visualization" style="max-width: 600px;"/>\n\n'
            else:
                md_img = ""
            print(
                f"{md_img}**Insight:** {insight}\n\n**Query:**\n{query}\n**Shape:** {shape if shape is not None else df.shape}"
            )
            """

        except Exception as e:
            print(f"[INSIGHT ERROR] Failed to parse insight for query: {query}\nError: {str(e)}")
            continue

    return {
        **state,
        "insights": insights  # No longer sorting by score
    }

### ---- Graph Wiring ---- ###
graph = StateGraph(GraphState)
graph.add_node("query_agent", query_agent)
graph.add_node("visualization_agent", visualization_agent)
graph.add_node("insight_agent", insight_generator)

graph.set_entry_point("query_agent")
graph.add_edge("query_agent", "visualization_agent")
graph.add_edge("visualization_agent", "insight_agent")
graph.add_edge("insight_agent", END)

multi_agent_graph = graph.compile()

def insights_to_html(insights):
    """
    Convert insights list to HTML with flexbox layout.
    """
    html = '<div style="display: flex; flex-direction: column; gap: 20px; width: 100%;">'

    for item in insights:
        img_html = ""
        img_filename = item.get("visualization")
        if img_filename and os.path.exists(img_filename):
            with open(img_filename, "rb") as img_file:
                b64img = base64.b64encode(img_file.read()).decode("utf-8")
            img_html = f'<img src="data:image/png;base64,{b64img}" style="max-width:100%;height:auto;margin-top:10px;" />'

        insight_card = f"""
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); background: white;">
            <h3>Insight</h3>
            <div style="max-height: 200px; overflow-y: auto; margin-bottom: 10px;">{item.get("insight", "")}</div>
            
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <div style="flex: 0.7; max-width: 70%;">
                    <h4>Query</h4>
                    <pre style="max-height: 150px; overflow-y: auto; background: #f5f5f5; padding: 10px; border-radius: 4px;">{item.get("query", "")}</pre>
                </div>
                <div style="flex: 0.3; max-width: 30%; margin-left: 15px; display: flex; flex-direction: column; align-items: center;">
                    <h4>Visualization</h4>
                    {img_html}
                    <div style="margin-top: 10px;">
                        <strong>Data Shape:</strong> {str(item.get("shape", ""))}
                    </div>
                </div>
            </div>
        </div>
        """
        html += insight_card

    html += '</div>'
    return html

def run_workflow_and_get_html():
    result = multi_agent_graph.invoke({})
    print_logs(result)  # Print logs to console
    return insights_to_html(result["insights"])

def print_logs(result):
    print("=== Executed Queries ===")
    for q in result["queries_executed"]:
        print(f"- `{q}`")

    print("=== Insights ===")
    for item in result["insights"]:
        img_filename = item.get("visualization")
        if img_filename and os.path.exists(img_filename):
            b64img = image_to_base64(img_filename)
            md_img = f'![Visualization](data:image/png;base64,{b64img})\n\n'
        else:
            md_img = ""
        print(
            f"**Image:**\n{md_img}**Insight:** {item.get('insight', '')}\n\n**Query:**\n{item['query']}\n**Shape:** {item.get('shape', '')}\n"
        )
    print(f' length of insights : {len(result["insights"])}')
    no_of_insights = len(result["insights"]) if "insights" in result else 0

    # Print result state without DataFrames
    def strip_dataframes(state):
        state_copy = dict(state)
        if "insights" in state_copy:
            state_copy["insights"] = [
                {k: v for k, v in item.items() if k != "result"}
                for item in state_copy["insights"]
            ]
        return state_copy

    print("=== State (without DataFrames) ===")
    print(strip_dataframes(result))

### ---- Execute ---- ###
if __name__ == "__main__":
    #print("=== Executing Multi-Agent BI Workflow ===")
    #result = multi_agent_graph.invoke({})
    """
    
    # Generate and save the graph image
    png_data = multi_agent_graph.get_graph().draw_mermaid_png()
    graph_img_path = "graph.png"
    with open(graph_img_path, "wb") as f:
        f.write(png_data)

    
    """
    # Define CSS for the overall page styling
    css = """
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f9f9f9;
    }
    
    h1, h2, h3, h4 {
        color: #333;
        margin-top: 0;
    }
    
    pre {
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.9em;
    }
    """

    # Gradio interface with flexbox layout instead of table
    with gr.Blocks(css=css) as demo:
        gr.Markdown("## AI Generated Insights")
        with gr.Row():
            run_btn = gr.Button("Generate AI generated Insights", variant="primary")


        # Content area - initially empty
        with gr.Row(visible=False) as content_row:
            insights_html = gr.HTML(label="Insights")

        def on_run():
            html_content = run_workflow_and_get_html()
            return gr.update(visible=True), html_content

        run_btn.click(
            on_run,
            outputs=[content_row, insights_html]
        )

    # Make the link shareable
    demo.launch(share=True)